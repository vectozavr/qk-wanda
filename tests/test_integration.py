import copy
import json

import numpy as np
import pytest
import torch

from qk_wanda import MaskArchive, apply_masks, prune_model
from .model_helpers import tiny_model
from qk_wanda.evaluation import perplexity
from qk_wanda.adapters import decoder_layers
from qk_wanda.masks import build_shared_qk_masks
from qk_wanda.scoring import QK_WANDA_LABELS, QKWandaAccumulator
from qk_wanda.pruning import validate_model

FAMILIES = ("llama", "qwen2", "mistral", "opt")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("attention", ("eager", "sdpa"))
def test_block_replay_matches_full_forward_reference(family, attention):
    """Reference uses full model forwards, without capture/replay adapters."""
    model = tiny_model(family)
    if attention == "sdpa" and not model._supports_sdpa:
        pytest.skip("This Transformers version uses eager attention for OPT")
    model.config._attn_implementation = attention
    model = type(model)(model.config).eval()
    reference = copy.deepcopy(model)
    tokens = torch.randint(3, 64, (3, 12))
    with torch.no_grad():
        for layer in decoder_layers(reference):
            nq = reference.config.num_attention_heads
            nk = getattr(reference.config, "num_key_value_heads", nq)
            q, k = layer.self_attn.q_proj, layer.self_attn.k_proj
            acc = QKWandaAccumulator(nq, nk, q.out_features // nq, variant="unmasked")
            handles = [
                q.register_forward_hook(acc.capture_query),
                k.register_forward_hook(acc.capture_key),
            ]
            try:
                for sequence in tokens:
                    reference(sequence[None], use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            qs, ks = acc.scores(q.weight, k.weight)
            qm, km = build_shared_qk_masks(qs, ks, 0.5)
            q.weight.masked_fill_(qm, 0)
            k.weight.masked_fill_(km, 0)
    report = prune_model(model, tokens)
    assert report["variant"] == "unmasked"
    assert report["scoring_label"] == "QK-Wanda"
    for (name, expected), (_, actual) in zip(
        reference.named_parameters(), model.named_parameters()
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "method,budget,variant",
    [
        ("qk-wanda", "shared", "unmasked"),
        ("qk-wanda", "separate", "unmasked"),
        ("qk-wanda", "shared", "causal"),
        ("qk-wanda", "separate", "causal"),
        ("qk-wanda", "row", "unmasked"),
        ("qk-wanda", "shared", "rope"),
        ("wanda", "row", "unmasked"),
        ("wanda", "separate", "unmasked"),
    ],
)
def test_prune_replay_and_scope(family, method, budget, variant, tmp_path):
    if family == "opt" and variant == "rope":
        pytest.skip("OPT has no RoPE")
    model = tiny_model(family)
    original = copy.deepcopy(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    tokens = torch.randint(3, 64, (3, 12))
    archive = MaskArchive()
    report = prune_model(
        model,
        tokens,
        method=method,
        budget=budget,
        variant=variant,
        batch_size=2,
        mask_callback=archive,
    )
    assert report["achieved_sparsity"] == 0.5
    assert report["variant"] == (variant if method == "qk-wanda" else None)
    assert report["scoring_label"] == (
        QK_WANDA_LABELS[variant] if method == "qk-wanda" else "Wanda"
    )
    assert model.config.use_cache == original.config.use_cache
    assert not model.training
    for name, parameter in model.named_parameters():
        if name.endswith(("q_proj.weight", "k_proj.weight")):
            assert torch.equal(parameter[parameter != 0], before[name][parameter != 0])
        else:
            assert torch.equal(parameter, before[name]), name
    for block in report["blocks"]:
        assert (
            sum(p["selected_for_removal"] for p in block["projections"])
            == sum(p["weights"] for p in block["projections"]) // 2
        )
    path = tmp_path / "masks.npz"
    archive.save(path)
    apply_masks(original, path)
    with torch.no_grad():
        logits = model(tokens).logits
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(logits, original(tokens).logits, rtol=0, atol=0)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("family", FAMILIES)
def test_hybrid_and_batch_invariance(family):
    first = tiny_model(family)
    second = copy.deepcopy(first)
    tokens = torch.randint(3, 64, (3, 12))
    a = prune_model(first, tokens, scope="block", batch_size=1)
    b = prune_model(second, tokens, scope="block", batch_size=2)
    assert a["achieved_sparsity"] == b["achieved_sparsity"] == 0.5
    expected = 6 if family == "opt" else 7
    assert all(len(block["projections"]) == expected for block in a["blocks"])
    for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("attention", ("eager", "sdpa"))
def test_noop_preserves_outputs(family, attention):
    model = tiny_model(family)
    if attention == "sdpa" and not model._supports_sdpa:
        pytest.skip("This Transformers version uses eager attention for OPT")
    # Reconstruct with the selected attention implementation.
    model.config._attn_implementation = attention
    model = type(model)(model.config).eval()
    tokens = torch.randint(3, 64, (3, 12))
    with torch.no_grad():
        before = model(tokens).logits.clone()
    report = prune_model(model, tokens, sparsity=0, batch_size=2)
    assert report["target_zeros"] == 0
    with torch.no_grad():
        torch.testing.assert_close(before, model(tokens).logits, rtol=0, atol=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_perplexity_matches_native_loss(family):
    model = tiny_model(family)
    tokens = torch.randint(3, 64, (37,))
    result = perplexity(model, tokens, seqlen=12, logit_chunk_size=5)
    with torch.no_grad():
        native = (
            torch.stack(
                [
                    model(tokens[i : i + 12][None], labels=tokens[i : i + 12][None]).loss
                    for i in (0, 12, 24)
                ]
            )
            .mean()
            .item()
        )
    assert result["mean_nll"] == pytest.approx(native, abs=1e-6)
    assert result["predicted_tokens"] == 33
    assert result["unused_tokens"] == 1


def test_validation_and_failure_restore_model():
    model = tiny_model()
    model.train()
    tokens = torch.randint(3, 64, (2, 8))
    with pytest.raises(ValueError, match="Wanda supports"):
        prune_model(model, tokens, method="wanda", budget="shared")
    with pytest.raises(ValueError, match="variant changes QK-Wanda only"):
        prune_model(model, tokens, method="wanda", variant="causal")
    with pytest.raises(ValueError, match="integer"):
        prune_model(model, tokens.float())
    model.config.model_type = "qwen3"
    with pytest.raises(ValueError, match="Unsupported"):
        prune_model(model, tokens)
    model.config.model_type = "llama"
    old = model.model.layers[0]
    hook = old.self_attn.q_proj.register_forward_hook(
        lambda *args: (_ for _ in ()).throw(RuntimeError("injected failure"))
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        prune_model(model, tokens)
    hook.remove()
    assert model.model.layers[0] is old
    assert model.training and model.config.use_cache
    assert all(not m._forward_hooks for m in model.modules())


def test_sliding_window_restriction_applies_only_to_masked_scores():
    model = tiny_model("mistral")
    model.config.sliding_window = 4
    tokens = torch.randint(3, 64, (3, 12))
    for variant in ("causal", "rope"):
        with pytest.raises(ValueError, match="sliding_window"):
            validate_model(model, variant=variant, seqlen=12)
    report = prune_model(model, tokens)
    assert report["variant"] == "unmasked"
    assert report["achieved_sparsity"] == 0.5


def test_invalid_archive_is_atomic(tmp_path):
    model = tiny_model()
    before = model.model.layers[0].self_attn.q_proj.weight.clone()
    name = "model.layers.0.self_attn.q_proj.weight"
    path = tmp_path / "bad.npz"
    metadata = {"schema_version": 1, "shapes": {name: list(before.shape), "missing": [1]}}
    np.savez(
        path,
        __metadata__=json.dumps(metadata),
        **{
            name: np.full((before.numel() + 7) // 8, 255, dtype=np.uint8),
            "missing": np.zeros(1, dtype=np.uint8),
        },
    )
    with pytest.raises(ValueError, match="mismatch"):
        apply_masks(model, path)
    assert torch.equal(before, model.model.layers[0].self_attn.q_proj.weight)
