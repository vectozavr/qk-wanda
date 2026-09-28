"""Next-token perplexity with a memory-bounded vocabulary projection."""

import math

import torch
import torch.nn.functional as F

from .adapters import input_device_for_model


@torch.no_grad()
def perplexity(model, tokens, *, seqlen=2048, max_sequences=None, logit_chunk_size=128):
    """Evaluate non-overlapping, complete context windows.

    Every window predicts its last T-1 tokens. The tail shorter than a full
    window is excluded. Vocabulary logits are computed in token chunks. No
    padding, training data, sliding-window stride, or task averaging is used.
    """
    if seqlen < 2 or logit_chunk_size < 1:
        raise ValueError("seqlen >= 2 and logit_chunk_size >= 1 are required")
    if tokens.ndim != 1:
        raise ValueError("Evaluation tokens must be one flat token stream")
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be positive")
    count = tokens.numel() // seqlen
    if max_sequences is not None:
        count = min(count, max_sequences)
    if count == 0:
        raise ValueError("Not enough evaluation tokens for one full context window")
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    try:
        for index in range(count):
            ids = tokens[index * seqlen : (index + 1) * seqlen].unsqueeze(0)
            outputs = model.model(
                ids.to(input_device_for_model(model)), use_cache=False, return_dict=True
            )
            hidden = outputs.last_hidden_state
            head_device = model.lm_head.weight.device
            for start in range(0, seqlen - 1, logit_chunk_size):
                stop = min(start + logit_chunk_size, seqlen - 1)
                logits = model.lm_head(hidden[:, start:stop].to(head_device)).float()
                labels = ids[:, start + 1 : stop + 1].to(logits.device)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
                )
                if not torch.isfinite(nll):
                    raise FloatingPointError(
                        "Nonfinite evaluation loss; try bfloat16 or float32 weights"
                    )
                total_nll += float(nll)
                total_tokens += labels.numel()
    finally:
        model.train(was_training)
    mean_nll = total_nll / total_tokens
    return {
        "perplexity": math.exp(mean_nll) if mean_nll < 709 else None,
        "mean_nll": mean_nll,
        "predicted_tokens": total_tokens,
        "sequences": count,
        "sequence_length": seqlen,
        "unused_tokens": tokens.numel() - count * seqlen,
    }
