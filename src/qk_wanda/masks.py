"""Deterministic deletion masks: True means remove the weight."""

import math
import os

import torch


def build_pruning_mask(
    scores: torch.Tensor,
    sparsity_ratio: float,
    prune_n: int = 0,
    prune_m: int = 0,
    granularity: str = "row",
    rounding: str = "ceil",
) -> torch.Tensor:
    """Build a deterministic mask where ``True`` entries are pruned.

    Counts round up within the selected budget group by default, for both
    Wanda and QK-Wanda. Explicit ``rounding='floor'`` reproduces the original
    Wanda integer convention and archived experiments. N:M masks are unaffected.
    At 50% with even Llama widths the two conventions coincide.
    """

    if scores.ndim != 2:
        raise ValueError(f"scores must be a matrix, got shape {tuple(scores.shape)}")
    if not 0 <= sparsity_ratio < 1:
        raise ValueError(f"sparsity_ratio must be in [0, 1), got {sparsity_ratio}")
    if granularity not in ("row", "matrix"):
        raise ValueError("granularity must be 'row' or 'matrix'")
    if rounding not in ("floor", "ceil"):
        raise ValueError("rounding must be 'floor' or 'ceil'")

    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("scores must be finite and nonnegative")

    rows, columns = scores.shape
    cpu_sort = os.environ.get("QK_WANDA_CPU_MASK_SORT") == "1"
    sort_scores = scores.detach().cpu() if cpu_sort else scores
    sort_mask = torch.zeros_like(sort_scores, dtype=torch.bool)

    if prune_n or prune_m:
        if prune_n <= 0 or prune_m <= 0 or prune_n >= prune_m:
            raise ValueError("N:M pruning requires 0 < prune_n < prune_m")
        if columns % prune_m:
            raise ValueError(
                f"input width {columns} is not divisible by the N:M block size {prune_m}"
            )
        blocks = sort_scores.reshape(rows, columns // prune_m, prune_m)
        indices = torch.argsort(blocks, dim=-1, stable=True)[..., :prune_n]
        sort_mask.reshape_as(blocks).scatter_(dim=-1, index=indices, value=True)
        return sort_mask.to(device=scores.device) if cpu_sort else sort_mask

    round_count = math.floor if rounding == "floor" else math.ceil
    if granularity == "row":
        count = round_count(columns * sparsity_ratio)
        if count:
            indices = torch.argsort(sort_scores, dim=1, stable=True)[:, :count]
            sort_mask.scatter_(dim=1, index=indices, value=True)
    else:
        count = round_count(scores.numel() * sparsity_ratio)
        if count:
            flat_indices = torch.argsort(sort_scores.reshape(-1), stable=True)[:count]
            sort_mask.reshape(-1).scatter_(dim=0, index=flat_indices, value=True)
    return sort_mask.to(device=scores.device) if cpu_sort else sort_mask


def build_shared_qk_masks(query_scores, key_scores, sparsity_ratio, rounding="ceil"):
    """Prune one pooled Q/K budget; ties use Q then K, each in row-major order.

    Scores are already expressed in the same QK-loss units, including GQA.
    No per-projection normalization or quota is applied.
    """
    if query_scores.ndim != 2 or key_scores.ndim != 2:
        raise ValueError("shared Q/K scores must be matrices")
    if query_scores.shape[1] != key_scores.shape[1]:
        raise ValueError("shared Q/K projections must have the same input width")
    if query_scores.device != key_scores.device or query_scores.dtype != key_scores.dtype:
        raise ValueError("shared Q/K scores must share device and dtype")
    cpu_sort = os.environ.get("QK_WANDA_CPU_MASK_SORT") == "1"
    q = query_scores.detach().cpu() if cpu_sort else query_scores.detach()
    k = key_scores.detach().cpu() if cpu_sort else key_scores.detach()
    pooled = torch.cat((q.reshape(-1), k.reshape(-1))).reshape(1, -1)
    if not torch.isfinite(pooled).all() or (pooled < 0).any():
        raise ValueError("shared Q/K scores must be finite and nonnegative")
    mask = build_pruning_mask(
        pooled, sparsity_ratio, granularity="matrix", rounding=rounding
    ).reshape(-1)
    boundary = query_scores.numel()
    return (
        mask[:boundary].reshape_as(query_scores).to(query_scores.device),
        mask[boundary:].reshape_as(key_scores).to(key_scores.device),
    )
