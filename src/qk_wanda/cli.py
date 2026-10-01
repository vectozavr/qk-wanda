"""Command-line entry points for pruning and evaluating Hugging Face models."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .data import calibration_data, evaluation_tokens, save_calibration
from .evaluation import perplexity
from .pruning import prune_model, validate_model
from .scoring import DEFAULT_QK_VARIANT, QK_WANDA_VARIANTS
from .serialization import MaskArchive, apply_masks


def _model_arguments(parser):
    parser.add_argument(
        "--model", required=True, help="Hugging Face model ID or local save_pretrained directory"
    )
    parser.add_argument("--revision", help="Pin a model/tokenizer revision for reproducible runs")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...; auto may distribute complete blocks",
    )
    parser.add_argument(
        "--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16")
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=("auto", "sdpa", "eager"),
        help="auto lets Transformers select a supported attention implementation",
    )
    parser.add_argument("--local-files-only", action="store_true")


def _eval_arguments(parser):
    parser.add_argument("--eval-dataset", choices=("wikitext2", "c4"), default="wikitext2")
    parser.add_argument("--eval-text", help="UTF-8 evaluation text instead of a downloaded dataset")
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument(
        "--eval-max-sequences",
        type=int,
        help="Optional smoke-test limit; omitted evaluates all full windows",
    )
    parser.add_argument("--logit-chunk-size", type=int, default=128)
    parser.add_argument("--eval-dataset-revision")


def build_parser():
    parser = argparse.ArgumentParser(description="QK-Wanda: Q/K-aware unstructured pruning")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    prune = sub.add_parser("prune", help="Prune and optionally evaluate a pretrained model")
    _model_arguments(prune)
    prune.add_argument("--output", type=Path, required=True, help="New or empty run directory")
    prune.add_argument("--sparsity", type=float, default=0.5)
    prune.add_argument("--method", choices=("qk-wanda", "wanda"), default="qk-wanda")
    prune.add_argument(
        "--budget",
        choices=("shared", "separate", "row"),
        help="Default: shared for QK-Wanda; row for Wanda",
    )
    prune.add_argument(
        "--variant",
        choices=QK_WANDA_VARIANTS,
        default=DEFAULT_QK_VARIANT,
        help="Default: unmasked QK-Wanda; causal: QK-Wanda-M; rope: QK-Wanda-MR",
    )
    prune.add_argument("--scope", choices=("qk", "block"), default="qk")
    prune.add_argument("--rounding", choices=("ceil", "floor"), default="ceil")
    source = prune.add_mutually_exclusive_group()
    source.add_argument("--calibration", choices=("wikitext2", "c4"), default="wikitext2")
    source.add_argument("--calibration-text", help="UTF-8 file; one document per nonempty line")
    source.add_argument(
        "--calibration-tokens",
        help="Previously saved integer .npy [N, T]; overrides nsamples/seqlen",
    )
    prune.add_argument("--nsamples", type=int, default=128)
    prune.add_argument("--seqlen", type=int, default=64)
    prune.add_argument(
        "--window-length",
        type=int,
        help="Sample longer document windows, then segment independently",
    )
    prune.add_argument("--seed", type=int, default=0)
    prune.add_argument("--dataset-revision")
    prune.add_argument("--batch-size", type=int, default=1)
    prune.add_argument(
        "--cpu-mask-sort",
        action="store_true",
        help="Trade RAM/transfer time for less GPU sorting memory",
    )
    prune.add_argument(
        "--save-calibration",
        action="store_true",
        help="Save token IDs for exact reuse; may contain private text",
    )
    prune.add_argument(
        "--save-model",
        action="store_true",
        help="Also save a full sparse HF checkpoint; dense storage size",
    )
    prune.add_argument(
        "--eval",
        action="store_true",
        help="Compare dense and sparse perplexity on identical held-out tokens",
    )
    _eval_arguments(prune)
    evaluate = sub.add_parser(
        "evaluate", help="Evaluate a saved model or replay masks on its original checkpoint"
    )
    _model_arguments(evaluate)
    _eval_arguments(evaluate)
    evaluate.add_argument("--masks", help="Optional masks.npz to apply to the original checkpoint")
    evaluate.add_argument("--output", type=Path, help="Write evaluation JSON to a new file")
    return parser


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "auto" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype
    if dtype == "auto":
        dtype = (
            "float32"
            if device == "cpu"
            else ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
        )
    common = dict(
        revision=args.revision, local_files_only=args.local_files_only, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    attention = (
        {}
        if args.attn_implementation == "auto"
        else {"attn_implementation": args.attn_implementation}
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, dtype),
        device_map=device,
        **attention,
        **common,
    ).eval()
    validate_model(model, variant=getattr(args, "variant", DEFAULT_QK_VARIANT))
    return model, tokenizer


def _evaluate(model, tokens, args):
    return perplexity(
        model,
        tokens,
        seqlen=args.eval_seqlen,
        max_sequences=args.eval_max_sequences,
        logit_chunk_size=args.logit_chunk_size,
    )


def _eval_data(tokenizer, args):
    return evaluation_tokens(
        tokenizer,
        source=args.eval_dataset,
        text_file=args.eval_text,
        dataset_revision=args.eval_dataset_revision,
    )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prune":
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError(
                "--output must be a new or empty directory; existing results are never overwritten"
            )
        if not 0 <= args.sparsity < 1 or args.batch_size < 1:
            raise ValueError("Require 0 <= sparsity < 1 and a positive batch-size")
        if args.method == "wanda" and args.budget == "shared":
            raise ValueError("Wanda supports row or separate budgets")
        if args.method == "wanda" and args.variant != DEFAULT_QK_VARIANT:
            raise ValueError("--variant applies to QK-Wanda only")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # Match the research implementation's FP32 accumulation policy.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.cpu_mask_sort:
            import os

            os.environ["QK_WANDA_CPU_MASK_SORT"] = "1"
    elif args.output and args.output.exists():
        raise ValueError("Evaluation output already exists; choose a new path")
    model, tokenizer = load_model(args)
    if args.command == "evaluate":
        if args.masks:
            apply_masks(model, args.masks)
        result = _evaluate(model, _eval_data(tokenizer, args), args)
        if args.output:
            _write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return

    tokens, calibration = calibration_data(
        tokenizer,
        source=args.calibration,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        window_length=args.window_length,
        text_file=args.calibration_text,
        token_file=args.calibration_tokens,
        dataset_revision=args.dataset_revision,
    )
    eval_tokens = _eval_data(tokenizer, args) if args.eval else None
    dense = _evaluate(model, eval_tokens, args) if args.eval else None
    archive = MaskArchive()
    report = prune_model(
        model,
        tokens,
        sparsity=args.sparsity,
        method=args.method,
        budget=args.budget,
        variant=args.variant,
        scope=args.scope,
        rounding=args.rounding,
        batch_size=args.batch_size,
        mask_callback=archive,
        progress=lambda i, n, _: print(f"Pruned block {i}/{n}", flush=True),
    )
    report.update(
        {
            "calibration": calibration,
            "model": args.model,
            "revision": args.revision,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "weight_dtype": str(next(model.parameters()).dtype),
            "attention_implementation": model.config._attn_implementation,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("qk-wanda", "torch", "transformers", "accelerate")
            },
        }
    )
    if args.eval:
        report["evaluation"] = {
            "dataset": "text_file" if args.eval_text else args.eval_dataset,
            "dataset_revision": args.eval_dataset_revision,
            "dense": dense,
            "pruned": _evaluate(model, eval_tokens, args),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    archive.save(args.output / "masks.npz")
    if args.save_calibration:
        save_calibration(tokens, args.output / "calibration.npy")
    if args.save_model:
        model.save_pretrained(args.output / "model", safe_serialization=True)
        tokenizer.save_pretrained(args.output / "model")
    _write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "achieved_target_sparsity": report["achieved_sparsity"],
                "output": str(args.output),
                "evaluation": report.get("evaluation"),
            },
            indent=2,
        )
    )
