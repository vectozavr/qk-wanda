"""Check the README's numerical example against the public pruning code."""

import json
from pathlib import Path

import torch

from qk_wanda.masks import build_pruning_mask, build_shared_qk_masks
from qk_wanda.scoring import QKWandaAccumulator


def test_readme_illustration_scores_masks_and_joint_error():
    example = json.loads((Path(__file__).parents[1] / "assets/illustration.json").read_text())
    x = torch.tensor(example["X"], dtype=torch.float64).T
    wq = torch.tensor(example["WQ"], dtype=torch.float64)
    wk = torch.tensor(example["WK"], dtype=torch.float64)
    q, k = x @ wq.T, x @ wk.T
    scorer = QKWandaAccumulator(1, 1, 2, accumulation_dtype=torch.float64)
    scorer.add_batch(x, q, k)
    qs, ks = scorer.scores(wq, wk)
    # The diagram displays square roots of unscaled costs. The scorer includes 1/m.
    for side, scores in (("Q", qs), ("K", ks)):
        displayed = torch.tensor(example[f"qk_scores_{side}"], dtype=torch.float64)
        torch.testing.assert_close(scores, displayed.square() / 2, rtol=0, atol=0)
    qm, km = build_shared_qk_masks(qs, ks, 0.5)
    input_norm = x.square().sum(0).sqrt()
    wqm = build_pruning_mask(wq.abs() * input_norm, 0.5)
    wkm = build_pruning_mask(wk.abs() * input_norm, 0.5)
    for field, mask in (
        ("qk_removed_Q", qm),
        ("qk_removed_K", km),
        ("wanda_removed_Q", wqm),
        ("wanda_removed_K", wkm),
    ):
        assert mask.tolist() == example[field]
    # Wanda retains a key weight in each column: this is unstructured pruning.
    assert (~wkm).any(dim=0).all()
    dense = q @ k.T
    errors = {}
    for name, query_mask, key_mask in (("Wanda", wqm, wkm), ("QK-Wanda", qm, km)):
        pruned_q = x @ wq.masked_fill(query_mask, 0).T
        pruned_k = x @ wk.masked_fill(key_mask, 0).T
        errors[name] = (pruned_q @ pruned_k.T - dense).square().sum().item()
        assert errors[name] == example["joint_deletion_checks"][name]["squared_error"]
    assert errors == {"Wanda": 81.0, "QK-Wanda": 9.0}
