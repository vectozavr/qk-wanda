"""Deterministic, unpadded calibration sequences and held-out evaluation text."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

C4_TRAIN = "en/c4-train.00000-of-01024.json.gz"
C4_VALIDATION = "en/c4-validation.00000-of-00008.json.gz"


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("Install dataset support with: pip install '.[data]'") from error
    return load_dataset(*args, **kwargs)


def _encode(tokenizer, text):
    # Tokenize the full stream here; calibration/evaluation select short windows
    # before any model forward, so the tokenizer's full-stream length warning
    # does not describe the inputs actually sent to the model.
    return tokenizer(text, return_tensors="pt", add_special_tokens=True, verbose=False).input_ids[0]


def sample_documents(documents, tokenizer, *, nsamples, seqlen, seed=0, window_length=None):
    """Sample unique documents, then split each sampled window independently.

    With window_length=2048 and seqlen=64, 8192 sequences come from 256
    distinct document windows, as in the main experimental protocol.
    """
    window_length = window_length or seqlen
    if nsamples <= 0 or seqlen < 2 or window_length < seqlen or window_length % seqlen:
        raise ValueError(
            "Positive sample count and window_length divisible by seqlen are required."
        )
    total = nsamples * seqlen
    if total % window_length:
        raise ValueError("nsamples * seqlen must be divisible by window_length")
    count = total // window_length
    if not len(documents) or count > len(documents):
        raise ValueError("Not enough documents for the requested number of distinct windows")
    rng = random.Random(seed)
    used, sampled, records = set(), [], []
    # Bounded rejection sampling preserves the paper's random-window convention
    # without hanging when the supplied text has too few sufficiently long rows.
    for _ in range(max(1000, count * 1000)):
        if len(sampled) == count:
            break
        index = rng.randint(0, len(documents) - 1)
        if index in used:
            continue
        item = documents[index]
        text = item["text"] if isinstance(item, dict) else item
        tokens = _encode(tokenizer, text)
        if len(tokens) < window_length:
            continue
        offset = rng.randint(0, len(tokens) - window_length)
        sampled.append(tokens[offset : offset + window_length])
        used.add(index)
        records.append({"document_index": index, "token_offset": offset})
    if len(sampled) != count:
        raise ValueError(
            "Not enough long documents found. Reduce sample count/length or provide more text."
        )
    return torch.stack(sampled).reshape(nsamples, seqlen), records


def calibration_data(
    tokenizer,
    *,
    source="wikitext2",
    nsamples=128,
    seqlen=64,
    seed=0,
    window_length=None,
    text_file=None,
    token_file=None,
    dataset_revision=None,
):
    if token_file:
        array = np.load(token_file, allow_pickle=False)
        if array.ndim != 2 or array.dtype.kind not in "iu":
            raise ValueError("Calibration .npy must contain an integer array [N, T]")
        return torch.from_numpy(array.astype(np.int64)), {"source": "token_file"}
    if text_file:
        # One document per non-empty line: no cross-document causal attention.
        documents = [
            line
            for line in Path(text_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source = "text_file"
    elif source == "c4":
        documents = _load_dataset(
            "allenai/c4",
            "en",
            data_files={"train": C4_TRAIN},
            split="train",
            revision=dataset_revision,
            verification_mode="no_checks",
        )
    elif source == "wikitext2":
        documents = _load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="train", revision=dataset_revision
        )
    else:
        raise ValueError("calibration source must be wikitext2 or c4")
    tokens, records = sample_documents(
        documents,
        tokenizer,
        nsamples=nsamples,
        seqlen=seqlen,
        seed=seed,
        window_length=window_length,
    )
    return tokens, {
        "source": source,
        "split": "train" if source != "text_file" else None,
        "seed": seed,
        "window_length": window_length or seqlen,
        "dataset_revision": dataset_revision,
        "dataset_fingerprint": getattr(documents, "_fingerprint", None),
        "windows": records,
    }


def evaluation_tokens(
    tokenizer, *, source="wikitext2", text_file=None, dataset_revision=None, c4_documents=1100
):
    if text_file:
        text = Path(text_file).read_text(encoding="utf-8")
    elif source == "wikitext2":
        dataset = _load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test", revision=dataset_revision
        )
        text = "\n\n".join(dataset["text"])
    elif source == "c4":
        dataset = _load_dataset(
            "allenai/c4",
            "en",
            data_files={"validation": C4_VALIDATION},
            split="validation",
            revision=dataset_revision,
            verification_mode="no_checks",
        )
        text = " ".join(dataset[:c4_documents]["text"])
    else:
        raise ValueError("evaluation source must be wikitext2 or c4")
    return _encode(tokenizer, text)


def save_calibration(tokens, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.save(file, tokens.detach().cpu().numpy(), allow_pickle=False)
