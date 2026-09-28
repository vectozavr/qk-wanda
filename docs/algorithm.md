# Algorithm and numerical conventions

## Objective

For one attention head, the default target is the causally allowed part of

$$\widehat Q^\top\widehat K-Q^\top K,$$

before RoPE. Its squared Frobenius norm is averaged over independent calibration sequences. With GQA, the loss sums over query heads: a shared key contributes to every query head that uses it. Biases remain fixed, but are included in the opposite projection's activations.

For a query weight, its squared magnitude multiplies input-feature squares weighted by the corresponding key coordinate's causal prefix sum of squares. For a key weight, use query suffix sums and sum the contributions of all query heads sharing that key. `scoring.py` accumulates these quantities without constructing a token-by-token attention matrix.

The scores are exact for deleting **one** original weight while fixing every other weight. Selecting many weights from these scores minimizes their additive surrogate, not the full interacting multiweight loss.

## Implementation conventions

- Input tensors use `[batch, tokens, features]`; linear weights use `[output, input]`.
- A Q/K output row identifies a head and coordinate within that head. GQA query heads are contiguous groups sharing one key head.
- Each sequence has its own causal context; sequences are never concatenated before computing QK interactions. Sequence costs are averaged, and query-head contributions to a shared key are summed.
- QK scores include a common factor `1 / head_dim`, corresponding to squared reconstruction error of scaled dot products. The paper's unscaled objective differs by this positive constant. Every Q/K weight in a block has the same head width, so every supported budget produces the same ranking and mask.
- Wanda uses `abs(weight) * sqrt(mean_sequence_input_energy)`. Squaring this nonnegative score preserves its ranking within every Wanda budget pool.
- Statistics and scores use FP32, regardless of the model's weight dtype. Original weights retain their dtype.
- A Boolean mask value of `True` means deletion.
- Counts round up by default. Ties use increasing flat row-major indices, with Q preceding K in shared pools.

## Sequential block procedure

1. Capture the calibration hidden states entering the first transformer block.
2. Collect Q/K activations and compute both score arrays with the current block's Q/K weights intact.
3. Select both masks using a shared pool, separate matrix pools, or row pools.
4. Set selected weights to zero. For `scope='block'`, also apply Wanda masks to V/O and MLP projections, collected during the same pass.
5. Run the pruned block on the calibration hidden states, and use those outputs for the next block.

There is one scoring/masking decision per block. No gradients, weight compensation, or progressive re-scoring are used. The optional RoPE variant uses the actual cosine/sine tensors supplied by the model, rather than guessing its frequency schedule.
