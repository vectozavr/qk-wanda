# Algorithm and numerical conventions

## Objective

For one attention head, QK-Wanda measures each deletion by the squared Frobenius norm of

$$\widehat Q^\top\widehat K-Q^\top K,$$

over **all token pairs**, before RoPE, without causal masking or centering. Costs are averaged over independent calibration sequences. With GQA, the loss sums over query heads: a shared key contributes to every query head that uses it. Biases remain fixed, but are included in the opposite projection's activations.

For each calibration sequence, a query weight's squared magnitude multiplies its input-feature energy and the corresponding key coordinate's energy. A key weight uses the corresponding query energies, summed over the query heads sharing that key. The implementation averages these **products within sequences**, rather than multiplying separately averaged energies. `scoring.py` accumulates the required energies without constructing a token-by-token attention matrix.

The optional `causal` variant (QK-Wanda-M) restricts these products to keys at or before each query, using key-prefix and query-suffix sums. The `rope` variant (QK-Wanda-MR) additionally applies the model's RoPE tensors. In the paper, M denotes masking, R denotes RoPE, and C denotes centering; the public package exposes the base method, M, and MR. Changing the score leaves the decoder's actual causal attention and positional embeddings unchanged.

The scores are exact for deleting **one** original weight while fixing every other weight. Selecting many weights from these scores minimizes their additive surrogate, not the full interacting multiweight loss.

## Implementation conventions

- Input tensors use `[batch, tokens, features]`; linear weights use `[output, input]`.
- A QK output row identifies a head and coordinate within that head. GQA query heads are contiguous groups sharing one key head.
- Each sequence is independent; sequences are never concatenated before computing QK interactions. Sequence costs are averaged, and query-head contributions to a shared key are summed.
- QK scores include a common factor `1 / head_dim`, corresponding to squared reconstruction error of scaled dot products. The paper's unscaled objective differs by this positive constant. Every QK weight in a block has the same head width, so every supported budget produces the same ranking and mask.
- Wanda uses `abs(weight) * sqrt(mean_sequence_input_energy)`. Squaring this nonnegative score preserves its ranking within every Wanda budget pool.
- Statistics and scores use FP32, regardless of the model's weight dtype. Original weights retain their dtype.
- A Boolean mask value of `True` means deletion.
- Counts round up by default. Ties use increasing flat row-major indices, with Q preceding K in shared pools.

## Sequential block procedure

1. Capture the calibration hidden states entering the first transformer block.
2. Collect QK activations and compute both score arrays with the current block's QK weights intact.
3. Select both masks using a shared pool, separate matrix pools, or row pools.
4. Set selected weights to zero. For `scope='block'`, also apply Wanda masks to value/output and MLP projections, collected during the same pass.
5. Run the pruned block on the calibration hidden states, and use those outputs for the next block.

There is one scoring/masking decision per block. No gradients, weight compensation, or progressive re-scoring are used. QK-Wanda-MR uses the actual cosine/sine tensors supplied by the model, rather than guessing its frequency schedule.
