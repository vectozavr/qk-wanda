import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from qk_wanda.cli import main
from qk_wanda.data import sample_documents
from .model_helpers import tiny_model


@pytest.mark.parametrize("method", ("qk-wanda", "wanda"))
def test_cli_offline_round_trip(method, tmp_path, capsys):
    model_path = tmp_path / "original"
    tiny_model("qwen2").save_pretrained(model_path)
    vocab = {"[UNK]": 0, **{f"w{i}": i for i in range(1, 64)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(model_path)
    tokens = tmp_path / "tokens.npy"
    np.save(tokens, np.random.default_rng(0).integers(3, 64, (3, 12)))
    evaluation = tmp_path / "eval.txt"
    evaluation.write_text(" ".join(f"w{i % 63 + 1}" for i in range(40)))
    output = tmp_path / "run"
    shared = [
        "--device",
        "cpu",
        "--local-files-only",
        "--eval-text",
        str(evaluation),
        "--eval-seqlen",
        "12",
        "--logit-chunk-size",
        "5",
    ]
    main(
        [
            "prune",
            "--model",
            str(model_path),
            "--output",
            str(output),
            "--calibration-tokens",
            str(tokens),
            "--save-model",
            "--save-calibration",
            "--eval",
            "--batch-size",
            "2",
            *(["--method", "wanda"] if method == "wanda" else []),
            *shared,
        ]
    )
    report = json.loads((output / "report.json").read_text())
    assert report["achieved_sparsity"] == 0.5
    assert report["variant"] == ("unmasked" if method == "qk-wanda" else None)
    assert report["scoring_label"] == ("QK-Wanda" if method == "qk-wanda" else "Wanda")
    assert report["budget"] == ("shared" if method == "qk-wanda" else "row")
    for name, source in (("saved", output / "model"), ("replayed", model_path)):
        args = [
            "evaluate",
            "--model",
            str(source),
            "--output",
            str(tmp_path / f"{name}.json"),
            *shared,
        ]
        if name == "replayed":
            args += ["--masks", str(output / "masks.npz")]
        main(args)
        result = json.loads((tmp_path / f"{name}.json").read_text())
        assert result["perplexity"] == pytest.approx(
            report["evaluation"]["pruned"]["perplexity"], abs=1e-6
        )
    sparse = AutoModelForCausalLM.from_pretrained(output / "model", local_files_only=True)
    assert torch.isfinite(sparse(torch.tensor([[3, 4, 5]])).logits).all()
    with pytest.raises(ValueError, match="existing results"):
        main(["prune", "--model", "must-not-download", "--output", str(output)])


@pytest.mark.parametrize("variant", ("causal", "rope"))
def test_wanda_rejects_qk_variant_before_loading(variant, tmp_path):
    with pytest.raises(ValueError, match="--variant applies to QK-Wanda only"):
        main(
            [
                "prune",
                "--model",
                "must-not-download",
                "--output",
                str(tmp_path / "run"),
                "--method",
                "wanda",
                "--variant",
                variant,
            ]
        )


def test_calibration_windows_stay_independent():
    class Numbers:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[int(x) for x in text.split()]]))

    docs = [" ".join(str(i * 100 + j) for j in range(30)) for i in range(5)]
    tokens, records = sample_documents(docs, Numbers(), nsamples=6, seqlen=4, window_length=8)
    repeated, other = sample_documents(docs, Numbers(), nsamples=6, seqlen=4, window_length=8)
    assert torch.equal(tokens, repeated) and records == other
    assert len({r["document_index"] for r in records}) == 3
    assert ((tokens[:, -1] - tokens[:, 0]) == 3).all()
    for i, record in enumerate(records):
        assert tokens[i * 2, 0] == record["document_index"] * 100 + record["token_offset"]
    with pytest.raises(ValueError, match="long documents"):
        sample_documents(["1 2", "3 4"], Numbers(), nsamples=1, seqlen=4)
