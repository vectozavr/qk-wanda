# Release notes

## 0.2.0 - 2026-10-01

- QK-Wanda now defaults to the paper's unmasked QK reconstruction objective in the CLI, Python pruning API, and tensor accumulator.
- `--variant causal` / `variant="causal"` selects QK-Wanda-M. Use it explicitly to preserve the scoring behavior of a 0.1.0 run. The existing `rope` option includes causal masking and is labeled QK-Wanda-MR.
- Reports include the corresponding paper name in `scoring_label`.
- Documentation and the Wanda versus QK-Wanda illustration match the final manuscript. The reproduction example uses calibration batches of 32 independent sequences.

## 0.1.0 - 2026-09-28

Initial release, with causal scoring as the default.
