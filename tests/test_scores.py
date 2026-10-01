"""Regression tests against explicit single-weight reconstruction errors."""

import unittest
import torch
from qk_wanda.scoring import QKWandaAccumulator, apply_llama_rope
from qk_wanda.masks import build_pruning_mask


def project(x, weight, bias=None):
    output = x @ weight.transpose(0, 1)
    return output if bias is None else output + bias


def reshape_heads(states, num_heads, head_dim):
    return states.reshape(states.shape[0], num_heads, head_dim)


def explicit_single_deletion_loss(
    kind,
    row,
    column,
    query_weight,
    key_weight,
    sequences,
    num_query_heads,
    num_key_heads,
    head_dim,
    variant="unmasked",
    query_bias=None,
    key_bias=None,
    rope_tensors=None,
):
    query_heads_per_key = num_query_heads // num_key_heads
    losses = []
    for sequence_index, x in enumerate(sequences):
        dense_query = reshape_heads(project(x, query_weight, query_bias), num_query_heads, head_dim)
        dense_key = reshape_heads(project(x, key_weight, key_bias), num_key_heads, head_dim)
        if variant == "rope":
            cos, sin = rope_tensors[sequence_index]
            dense_query = apply_llama_rope(dense_query.unsqueeze(0), cos, sin).squeeze(0)
            dense_key = apply_llama_rope(dense_key.unsqueeze(0), cos, sin).squeeze(0)

        if kind == "query":
            pruned_weight = query_weight.clone()
            pruned_weight[row, column] = 0
            pruned_query = reshape_heads(
                project(x, pruned_weight, query_bias), num_query_heads, head_dim
            )
            if variant == "rope":
                cos, sin = rope_tensors[sequence_index]
                pruned_query = apply_llama_rope(pruned_query.unsqueeze(0), cos, sin).squeeze(0)
            head = row // head_dim
            key_head = head // query_heads_per_key
            delta = pruned_query[:, head] @ dense_key[:, key_head].transpose(0, 1) - dense_query[
                :, head
            ] @ dense_key[:, key_head].transpose(0, 1)
            if variant in ("causal", "rope"):
                delta = delta * torch.tril(torch.ones_like(delta))
            loss = delta.square().sum() / head_dim
        else:
            pruned_weight = key_weight.clone()
            pruned_weight[row, column] = 0
            pruned_key = reshape_heads(project(x, pruned_weight, key_bias), num_key_heads, head_dim)
            if variant == "rope":
                cos, sin = rope_tensors[sequence_index]
                pruned_key = apply_llama_rope(pruned_key.unsqueeze(0), cos, sin).squeeze(0)
            key_head = row // head_dim
            loss = torch.zeros((), dtype=x.dtype)
            first_query_head = key_head * query_heads_per_key
            for head in range(first_query_head, first_query_head + query_heads_per_key):
                delta = dense_query[:, head] @ pruned_key[:, key_head].transpose(
                    0, 1
                ) - dense_query[:, head] @ dense_key[:, key_head].transpose(0, 1)
                if variant in ("causal", "rope"):
                    delta = delta * torch.tril(torch.ones_like(delta))
                loss = loss + delta.square().sum() / head_dim
        losses.append(loss)
    return torch.stack(losses).mean()


class QKWandaScoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.dtype = torch.float64

    def test_default_scores_include_future_token_pairs(self):
        x = torch.eye(2, dtype=self.dtype)
        wq = torch.tensor([[2.0, 1.0]], dtype=self.dtype)
        wk = torch.tensor([[1.0, 3.0]], dtype=self.dtype)
        default = QKWandaAccumulator(1, 1, 1, accumulation_dtype=self.dtype)
        default.add_batch(x, project(x, wq), project(x, wk))
        qs, ks = default.scores(wq, wk)
        # Full QK reconstruction counts both keys for each query.
        torch.testing.assert_close(qs, torch.tensor([[40.0, 10.0]], dtype=self.dtype))
        torch.testing.assert_close(ks, torch.tensor([[5.0, 45.0]], dtype=self.dtype))
        causal = QKWandaAccumulator(1, 1, 1, variant="causal", accumulation_dtype=self.dtype)
        causal.add_batch(x, project(x, wq), project(x, wk))
        cq, ck = causal.scores(wq, wk)
        torch.testing.assert_close(cq, torch.tensor([[4.0, 10.0]], dtype=self.dtype))
        torch.testing.assert_close(ck, torch.tensor([[5.0, 9.0]], dtype=self.dtype))

    def assert_all_scalar_deletions_match(
        self,
        num_query_heads,
        num_key_heads,
        head_dim,
        sequences,
        variant="unmasked",
        use_bias=True,
        rope_tensors=None,
    ):
        input_dim = sequences[0].shape[-1]
        query_weight = torch.randn(num_query_heads * head_dim, input_dim, dtype=self.dtype)
        key_weight = torch.randn(num_key_heads * head_dim, input_dim, dtype=self.dtype)
        query_weight[0, 0] = 0
        key_weight[-1, -1] = 0
        query_bias = torch.randn(num_query_heads * head_dim, dtype=self.dtype) if use_bias else None
        key_bias = torch.randn(num_key_heads * head_dim, dtype=self.dtype) if use_bias else None

        accumulator = QKWandaAccumulator(
            num_query_heads,
            num_key_heads,
            head_dim,
            variant=variant,
            accumulation_dtype=self.dtype,
        )
        for sequence_index, x in enumerate(sequences):
            kwargs = {}
            if variant == "rope":
                kwargs["cos"], kwargs["sin"] = rope_tensors[sequence_index]
            accumulator.add_batch(
                x,
                project(x, query_weight, query_bias),
                project(x, key_weight, key_bias),
                **kwargs,
            )
        query_scores, key_scores = accumulator.scores(query_weight, key_weight)

        for kind, weight, scores in (
            ("query", query_weight, query_scores),
            ("key", key_weight, key_scores),
        ):
            for row in range(weight.shape[0]):
                for column in range(weight.shape[1]):
                    measured = explicit_single_deletion_loss(
                        kind,
                        row,
                        column,
                        query_weight,
                        key_weight,
                        sequences,
                        num_query_heads,
                        num_key_heads,
                        head_dim,
                        variant=variant,
                        query_bias=query_bias,
                        key_bias=key_bias,
                        rope_tensors=rope_tensors,
                    )
                    self.assertTrue(
                        torch.allclose(measured, scores[row, column], atol=1e-10, rtol=1e-10),
                        msg=(
                            f"{variant} {kind} ({row}, {column}): "
                            f"explicit={measured.item()} analytic={scores[row, column].item()}"
                        ),
                    )

    def test_unmasked_mha_matches_every_scalar_deletion(self):
        sequences = [
            torch.randn(3, 4, dtype=self.dtype),
            torch.randn(5, 4, dtype=self.dtype),
        ]
        self.assert_all_scalar_deletions_match(2, 2, 2, sequences)

    def test_unmasked_gqa_matches_every_scalar_deletion_with_bias(self):
        sequences = [
            torch.randn(2, 3, dtype=self.dtype),
            torch.randn(4, 3, dtype=self.dtype),
        ]
        self.assert_all_scalar_deletions_match(4, 2, 2, sequences)

    def test_causal_gqa_matches_every_scalar_deletion(self):
        sequences = [torch.randn(4, 3, dtype=self.dtype)]
        self.assert_all_scalar_deletions_match(4, 2, 2, sequences, variant="causal", use_bias=False)

    def test_causal_mha_with_bias_matches_every_scalar_deletion(self):
        sequences = [torch.randn(4, 3, dtype=self.dtype)]
        self.assert_all_scalar_deletions_match(2, 2, 2, sequences, variant="causal", use_bias=True)

    def test_rope_gqa_matches_every_scalar_deletion(self):
        sequence = torch.randn(4, 3, dtype=self.dtype)
        angles = torch.randn(1, sequence.shape[0], 2, dtype=self.dtype)
        cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
        sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
        self.assert_all_scalar_deletions_match(
            4,
            2,
            4,
            [sequence],
            variant="rope",
            use_bias=False,
            rope_tensors=[(cos, sin)],
        )

    def test_sequencewise_aggregation_excludes_cross_sequence_terms(self):
        num_query_heads, num_key_heads, head_dim, input_dim = 2, 1, 2, 3
        query_weight = torch.randn(4, input_dim, dtype=self.dtype)
        key_weight = torch.randn(2, input_dim, dtype=self.dtype)
        sequences = [
            torch.tensor([[4.0, 0.0, 1.0], [0.0, 0.5, 0.0]], dtype=self.dtype),
            torch.tensor([[0.0, 3.0, 0.0], [0.0, 2.0, 5.0]], dtype=self.dtype),
        ]
        accumulator = QKWandaAccumulator(
            num_query_heads,
            num_key_heads,
            head_dim,
            variant="unmasked",
            accumulation_dtype=self.dtype,
        )
        x_energies = []
        key_energies = []
        for x in sequences:
            query = project(x, query_weight)
            key = project(x, key_weight)
            accumulator.add_batch(x, query, key)
            x_energies.append(x.square().sum(0))
            key_energies.append(key.reshape(x.shape[0], num_key_heads, head_dim).square().sum(0))
        gamma_query, _ = accumulator.interaction_factors()

        expected = sum(
            key_energy.reshape(-1, 1) * x_energy.reshape(1, -1)
            for key_energy, x_energy in zip(key_energies, x_energies)
        ) / len(sequences)
        expected = expected.repeat(num_query_heads, 1)
        global_product = (
            torch.stack(key_energies).mean(0).reshape(-1, 1)
            * torch.stack(x_energies).mean(0).reshape(1, -1)
        ).repeat(num_query_heads, 1)

        self.assertTrue(torch.allclose(gamma_query, expected))
        self.assertFalse(torch.allclose(gamma_query, global_product))

    def test_batch_elements_remain_independent_sequences(self):
        num_query_heads, num_key_heads, head_dim, input_dim = 4, 2, 2, 3
        query_weight = torch.randn(8, input_dim, dtype=self.dtype)
        key_weight = torch.randn(4, input_dim, dtype=self.dtype)
        inputs = torch.randn(2, 4, input_dim, dtype=self.dtype)
        query_outputs = project(inputs, query_weight)
        key_outputs = project(inputs, key_weight)

        batched = QKWandaAccumulator(
            num_query_heads,
            num_key_heads,
            head_dim,
            accumulation_dtype=self.dtype,
        )
        batched.add_batch(inputs, query_outputs, key_outputs)
        separate = QKWandaAccumulator(
            num_query_heads,
            num_key_heads,
            head_dim,
            accumulation_dtype=self.dtype,
        )
        for index in range(inputs.shape[0]):
            separate.add_batch(inputs[index], query_outputs[index], key_outputs[index])

        for batched_score, separate_score in zip(
            batched.scores(query_weight, key_weight),
            separate.scores(query_weight, key_weight),
        ):
            self.assertTrue(torch.allclose(batched_score, separate_score))

    def test_reciprocal_group_scaling_leaves_scores_invariant(self):
        num_query_heads, num_key_heads, head_dim, input_dim = 4, 2, 2, 3
        query_weight = torch.randn(8, input_dim, dtype=self.dtype)
        key_weight = torch.randn(4, input_dim, dtype=self.dtype)
        sequences = [
            torch.randn(3, input_dim, dtype=self.dtype),
            torch.randn(4, input_dim, dtype=self.dtype),
        ]

        def get_scores(q_weight, k_weight):
            accumulator = QKWandaAccumulator(
                num_query_heads,
                num_key_heads,
                head_dim,
                accumulation_dtype=self.dtype,
            )
            for x in sequences:
                accumulator.add_batch(x, project(x, q_weight), project(x, k_weight))
            return accumulator.scores(q_weight, k_weight)

        original_query, original_key = get_scores(query_weight, key_weight)
        scaled_query = query_weight.clone()
        scaled_key = key_weight.clone()
        scale = -2.75
        # Key group 1 serves query heads 2 and 3: rows 4:8.
        scaled_query[4:8] *= scale
        scaled_key[2:4] /= scale
        changed_query, changed_key = get_scores(scaled_query, scaled_key)

        self.assertTrue(torch.allclose(original_query, changed_query, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(original_key, changed_key, atol=1e-10, rtol=1e-10))

    def test_projection_hooks_capture_the_linear_input(self):
        input_dim, num_query_heads, num_key_heads, head_dim = 3, 4, 2, 2
        query = torch.nn.Linear(input_dim, num_query_heads * head_dim, bias=True).double()
        key = torch.nn.Linear(input_dim, num_key_heads * head_dim, bias=True).double()
        normalized_input = torch.randn(1, 5, input_dim, dtype=self.dtype)

        hooked = QKWandaAccumulator(
            num_query_heads,
            num_key_heads,
            head_dim,
            accumulation_dtype=self.dtype,
        )
        handles = [
            query.register_forward_hook(hooked.capture_query),
            key.register_forward_hook(hooked.capture_key),
        ]
        query_output = query(normalized_input)
        key_output = key(normalized_input)
        for handle in handles:
            handle.remove()
        hooked.assert_hooks_drained()

        direct = QKWandaAccumulator(
            num_query_heads,
            num_key_heads,
            head_dim,
            accumulation_dtype=self.dtype,
        )
        direct.add_batch(normalized_input, query_output, key_output)
        for hooked_score, direct_score in zip(
            hooked.scores(query.weight, key.weight),
            direct.scores(query.weight, key.weight),
        ):
            self.assertTrue(torch.equal(hooked_score, direct_score))

    def test_one_sequence_unmasked_row_masks_reduce_to_wanda(self):
        num_query_heads, num_key_heads, head_dim, input_dim = 4, 2, 2, 5
        query_weight = torch.randn(8, input_dim, dtype=self.dtype)
        key_weight = torch.randn(4, input_dim, dtype=self.dtype)
        x = torch.randn(6, input_dim, dtype=self.dtype)
        accumulator = QKWandaAccumulator(
            num_query_heads,
            num_key_heads,
            head_dim,
            variant="unmasked",
            accumulation_dtype=self.dtype,
        )
        accumulator.add_batch(x, project(x, query_weight), project(x, key_weight))
        query_score, key_score = accumulator.scores(query_weight, key_weight)
        wanda_query = query_weight.abs() * x.square().sum(0).sqrt().reshape(1, -1)
        wanda_key = key_weight.abs() * x.square().sum(0).sqrt().reshape(1, -1)
        self.assertTrue(
            torch.equal(
                build_pruning_mask(query_score, 0.4),
                build_pruning_mask(wanda_query, 0.4),
            )
        )
        self.assertTrue(
            torch.equal(
                build_pruning_mask(key_score, 0.4),
                build_pruning_mask(wanda_key, 0.4),
            )
        )
