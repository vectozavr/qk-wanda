"""Sequential block pruning with a shared Q/K budget by default."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable

import torch
from torch import nn

from .adapters import (
    _head_layout,
    _move_to_device,
    _rope_for_layer,
    decoder_layers,
    prepare_qk_calibration_input,
)
from .masks import build_pruning_mask, build_shared_qk_masks
from .scoring import (
    DEFAULT_QK_VARIANT,
    QK_WANDA_LABELS,
    QK_WANDA_VARIANTS,
    QKWandaAccumulator,
    WandaInputAccumulator,
)

SUPPORTED_MODEL_TYPES = ("llama", "qwen2", "mistral", "opt")


def validate_model(model, *, variant=DEFAULT_QK_VARIANT, seqlen=None):
    """Reject layouts whose projection outputs do not define this objective."""
    kind = getattr(model.config, "model_type", None)
    if kind not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type {kind!r}. Supported: {SUPPORTED_MODEL_TYPES}. "
            "Fused QKV, Q/K normalization and custom remote-code models need an adapter."
        )
    if getattr(model, "is_quantized", False):
        raise ValueError("Load floating-point weights, not a quantized checkpoint.")
    if any(p.is_meta for p in model.parameters()):
        raise ValueError(
            "Offloaded/meta weights are unsupported; keep the model resident on CPU/GPU."
        )
    if getattr(model.config, "pretraining_tp", 1) != 1:
        raise ValueError("Projection hooks require pretraining_tp == 1.")
    if kind == "opt" and variant == "rope":
        raise ValueError("OPT uses absolute positions; use variant='causal' or 'unmasked'.")
    if seqlen is not None:
        maximum = getattr(model.config, "max_position_embeddings", None)
        if maximum is not None and seqlen > maximum:
            raise ValueError(f"Calibration length {seqlen} exceeds model context {maximum}.")
        sliding = getattr(model.config, "sliding_window", None)
        if kind == "qwen2" and not getattr(model.config, "use_sliding_window", False):
            sliding = None
        if variant in ("causal", "rope") and sliding and seqlen > sliding:
            raise ValueError(
                "Use calibration length <= sliding_window: the score uses a full causal mask."
            )
    layers = decoder_layers(model)
    if not len(layers):
        raise ValueError("The model has no decoder blocks.")
    for block in layers:
        attn = getattr(block, "self_attn", None)
        if attn is None:
            raise ValueError("Each block must expose self_attn.")
        if any(getattr(attn, name, None) is not None for name in ("q_norm", "k_norm")):
            raise ValueError("Nonlinear Q/K normalization is not supported.")
        q, k = getattr(attn, "q_proj", None), getattr(attn, "k_proj", None)
        if not isinstance(q, nn.Linear) or not isinstance(k, nn.Linear):
            raise ValueError("Separate nn.Linear q_proj and k_proj modules are required.")
        devices = {p.device for p in block.parameters()}
        if any(d.type == "meta" for d in devices):
            raise ValueError(
                "CPU/disk weight offload is not supported. Keep complete blocks resident "
                "on CPU or GPU; use more GPU memory or a smaller model."
            )
        if len(devices) != 1:
            raise ValueError("Keep each complete decoder block on a single device.")
        if not q.weight.is_floating_point() or q.in_features != k.in_features:
            raise ValueError("Q/K must be floating-point projections with the same input width.")
        nq, nk, _ = _head_layout(model, block, q, k)
        if nq % nk:
            raise ValueError("Regular contiguous GQA groups are required.")
    return layers


def _output_hidden(output):
    # Transformers 4.45 blocks return a tuple; newer releases may return a tensor.
    return output if torch.is_tensor(output) else output[0]


def _batch_context(context, batch_size):
    # OPT checks the exact mask shape instead of broadcasting its batch axis.
    result = dict(context)
    mask = result.get("attention_mask")
    if mask is not None and mask.ndim == 4 and mask.shape[0] == 1:
        result["attention_mask"] = mask.expand(batch_size, *mask.shape[1:])
    return result


@torch.no_grad()
def prune_model(
    model,
    calibration_tokens: torch.Tensor,
    *,
    sparsity: float = 0.5,
    method: str = "qk-wanda",
    budget: str | None = None,
    variant: str = DEFAULT_QK_VARIANT,
    scope: str = "qk",
    rounding: str = "ceil",
    batch_size: int = 1,
    storage_device: str = "cpu",
    mask_callback: Callable | None = None,
    progress: Callable | None = None,
) -> dict:
    """Prune a loaded model in place and return a JSON-serializable report.

    ``calibration_tokens`` is an integer tensor [sequences, tokens] with no
    padding. Each row is an independent attention context. Hidden states are
    captured once and replayed block by block. A block's sparse output becomes
    the next block's input. Both Q/K masks are chosen before either is applied.

    ``variant`` defaults to the paper's full QK objective, ``'unmasked'``.
    ``'causal'`` selects QK-Wanda-M; ``'rope'`` selects QK-Wanda-MR, including
    both causal masking and RoPE. These switches change the scoring objective,
    not the model's attention computation.

    ``budget`` defaults to shared for QK-Wanda and row for Wanda. ``scope='block'``
    additionally prunes V/O and MLP projections with row-wise Wanda. Biases,
    embeddings, normalization parameters and the output head remain unchanged.

    The optional mask callback receives (full_parameter_name, boolean_mask).
    True means deletion. Keep a copy if retaining the tensor after the callback.
    """
    if method not in ("qk-wanda", "wanda"):
        raise ValueError("method must be 'qk-wanda' or 'wanda'")
    budget = budget or ("shared" if method == "qk-wanda" else "row")
    if budget not in ("shared", "separate", "row"):
        raise ValueError("budget must be shared, separate or row")
    if method == "wanda" and budget == "shared":
        raise ValueError("Wanda supports row or separate matrix budgets, not shared Q/K scores.")
    if not math.isfinite(sparsity) or not 0 <= sparsity < 1:
        raise ValueError("sparsity must be finite and in [0, 1)")
    if variant not in QK_WANDA_VARIANTS:
        raise ValueError("variant must be causal, unmasked or rope")
    if method == "wanda" and variant != DEFAULT_QK_VARIANT:
        raise ValueError("variant changes QK-Wanda only; leave it at its default for Wanda.")
    if scope not in ("qk", "block") or rounding not in ("ceil", "floor"):
        raise ValueError("scope must be qk/block and rounding must be ceil/floor")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not torch.is_tensor(calibration_tokens) or calibration_tokens.ndim != 2:
        raise ValueError("calibration_tokens must be an unpadded tensor [N, T]")
    if calibration_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("calibration_tokens must contain integer token IDs")
    nsamples, seqlen = calibration_tokens.shape
    if nsamples < 1 or seqlen < 2:
        raise ValueError("Use at least one sequence and at least two tokens per sequence.")
    if (
        calibration_tokens.min() < 0
        or calibration_tokens.max() >= model.get_input_embeddings().num_embeddings
    ):
        raise ValueError("Calibration token IDs are outside the model vocabulary.")
    layers = validate_model(model, variant=variant, seqlen=seqlen)
    was_training, old_cache = model.training, model.config.use_cache
    model.eval()
    model.config.use_cache = False
    names = {id(p): n for n, p in model.named_parameters()}
    report = {
        "schema_version": 1,
        "method": method,
        "variant": variant if method == "qk-wanda" else None,
        "scoring_label": QK_WANDA_LABELS[variant] if method == "qk-wanda" else "Wanda",
        "budget": budget,
        "scope": scope,
        "rounding": rounding,
        "requested_sparsity": sparsity,
        "model_type": model.config.model_type,
        "calibration_sequences": nsamples,
        "calibration_sequence_length": seqlen,
        "calibration_token_sha256": hashlib.sha256(
            calibration_tokens.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
        ).hexdigest(),
        "blocks": [],
    }
    try:
        batches = calibration_tokens.split(batch_size)
        inputs, outputs, kwargs = prepare_qk_calibration_input(
            model,
            batches,
            nsamples,
            torch.device("cpu"),
            seqlen=seqlen,
            storage_device=storage_device,
        )
        for block_index, layer in enumerate(layers):
            device = next(layer.parameters()).device
            context = _move_to_device(kwargs, device)
            q, k = layer.self_attn.q_proj, layer.self_attn.k_proj
            nq, nk, width = _head_layout(model, layer, q, k)
            selected = {"self_attn.q_proj": q, "self_attn.k_proj": k}
            if scope == "block":
                expected = (
                    {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
                    if model.config.model_type == "opt"
                    else {
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    }
                )
                selected = {
                    n: m
                    for n, m in layer.named_modules()
                    if isinstance(m, nn.Linear) and n.split(".")[-1] in expected
                }
            wanda = {
                name: WandaInputAccumulator(module, sequence_length=seqlen)
                for name, module in selected.items()
                if method == "wanda" or module not in (q, k)
            }
            accumulator = (
                QKWandaAccumulator(nq, nk, width, variant=variant) if method == "qk-wanda" else None
            )
            hooks = []
            try:
                hooks += [a.layer.register_forward_hook(a.capture) for a in wanda.values()]
                if accumulator is not None:
                    hooks += [
                        q.register_forward_hook(accumulator.capture_query),
                        k.register_forward_hook(accumulator.capture_key),
                    ]
                for start in range(0, nsamples, batch_size):
                    sample = inputs[start : start + batch_size].to(device)
                    if accumulator is not None and variant == "rope":
                        accumulator.set_next_rope(*_rope_for_layer(layer, sample, context))
                    _output_hidden(layer(sample, **_batch_context(context, sample.shape[0])))
                if accumulator is not None:
                    accumulator.assert_hooks_drained()
                    if accumulator.num_sequences != nsamples:
                        raise RuntimeError(
                            "Q/K projection hooks did not see every calibration sequence."
                        )
                if any(a.num_sequences != nsamples for a in wanda.values()):
                    raise RuntimeError(
                        "Wanda projection hooks did not see every calibration sequence."
                    )
            finally:
                for hook in hooks:
                    hook.remove()

            masks = {}
            if accumulator is not None:
                qs, ks = accumulator.scores(q.weight, k.weight)
                if budget == "shared":
                    qm, km = build_shared_qk_masks(qs, ks, sparsity, rounding)
                else:
                    granularity = "matrix" if budget == "separate" else "row"
                    qm, km = [
                        build_pruning_mask(s, sparsity, granularity=granularity, rounding=rounding)
                        for s in (qs, ks)
                    ]
                masks.update({"self_attn.q_proj": qm, "self_attn.k_proj": km})
                del qs, ks, qm, km, accumulator
            for name, a in wanda.items():
                granularity = "matrix" if budget == "separate" and a.layer in (q, k) else "row"
                masks[name] = build_pruning_mask(
                    a.score(), sparsity, granularity=granularity, rounding=rounding
                )
            projections = []
            for name, mask in masks.items():
                parameter = selected[name].weight
                before = int((parameter == 0).sum())
                parameter.masked_fill_(mask, 0)
                if mask_callback is not None:
                    mask_callback(names[id(parameter)], mask)
                zeros = int((parameter == 0).sum())
                projections.append(
                    {
                        "name": names[id(parameter)],
                        "weights": parameter.numel(),
                        "selected_for_removal": int(mask.sum()),
                        "zeros_before": before,
                        "zeros_after": zeros,
                        "sparsity": zeros / parameter.numel(),
                    }
                )
            del masks, wanda
            for start in range(0, nsamples, batch_size):
                stop = min(start + batch_size, nsamples)
                hidden = _output_hidden(
                    layer(inputs[start:stop].to(device), **_batch_context(context, stop - start))
                )
                if not torch.isfinite(hidden).all():
                    raise FloatingPointError(
                        f"Nonfinite output in block {block_index}; try bfloat16 or float32."
                    )
                outputs[start:stop].copy_(hidden)
            inputs, outputs = outputs, inputs
            block_report = {
                "index": block_index,
                "query_heads": nq,
                "key_value_heads": nk,
                "head_dim": width,
                "projections": projections,
            }
            report["blocks"].append(block_report)
            if progress is not None:
                progress(block_index + 1, len(layers), block_report)
        entries = [p for b in report["blocks"] for p in b["projections"]]
        report["target_weights"] = sum(p["weights"] for p in entries)
        report["target_zeros"] = sum(p["zeros_after"] for p in entries)
        report["achieved_sparsity"] = report["target_zeros"] / report["target_weights"]
        return report
    finally:
        model.config.use_cache = old_cache
        model.train(was_training)
