import itertools
import math

import pytest
import torch

from qk_wanda.masks import build_pruning_mask, build_shared_qk_masks


@pytest.mark.parametrize("rounding", ("ceil", "floor"))
def test_shared_budget_matches_exhaustive_optimum(rounding):
    q = torch.tensor([[3.0, 2.0], [4.0, 1.0]])
    k = torch.tensor([[0.5, 5.0]])
    qm, km = build_shared_qk_masks(q, k, 0.4, rounding)
    values = torch.cat((q.flatten(), k.flatten()))
    mask = torch.cat((qm.flatten(), km.flatten()))
    count = getattr(math, rounding)(6 * 0.4)
    assert int(mask.sum()) == count
    optimum = min(
        sum(values[i].item() for i in indices)
        for indices in itertools.combinations(range(6), count)
    )
    assert float(values[mask].sum()) == optimum


def test_ties_row_major_q_before_k(monkeypatch):
    q, k = torch.ones(2, 4), torch.ones(1, 4)
    qm, km = build_shared_qk_masks(q, k, 0.75)
    assert qm.all() and km.tolist() == [[True, False, False, False]]
    rows = build_pruning_mask(q, 0.5)
    assert rows.tolist() == [[True, True, False, False]] * 2
    monkeypatch.setenv("QK_WANDA_CPU_MASK_SORT", "1")
    a, b = build_shared_qk_masks(q, k, 0.75)
    assert torch.equal(a, qm) and torch.equal(b, km)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -1.0))
def test_invalid_scores_fail(bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_pruning_mask(torch.tensor([[bad, 1.0]]), 0.5)
