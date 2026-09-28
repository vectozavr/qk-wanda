# Reproduction guide

## Main calibration protocol

The main Q/K-only experiments use **256 distinct C4 document windows of 2048 tokens**, split into **8192 independent sequences of 64 tokens**: 524,288 calibration tokens in total. Sampling uses seed 0 and the first English C4 training shard. Windows never cross document boundaries.

Use the same tokenizer and checkpoint revision for sampling, pruning, and evaluation. The script below uses the original Transformer implementation version, FP16 weights, and a shared Q/K budget. Qwen2.5-72B uses BF16 instead.

```bash
python -m pip install 'transformers==4.45.2'
qk-wanda prune \
  --model meta-llama/Llama-2-7b-hf \
  --output runs/llama2-7b-qk-s80 \
  --dtype float16 --attn-implementation eager \
  --calibration c4 --nsamples 8192 --seqlen 64 --window-length 2048 \
  --seed 0 --sparsity 0.8 --batch-size 1 \
  --save-calibration --save-model --eval
```

Add `--revision <checkpoint-commit>` and `--dataset-revision <dataset-commit>` to pin the input artifacts. `report.json` records the resolved model revision, requested dataset revision, dataset fingerprint, sampled document indices/offsets, and calibration token hash. Keep the saved token array for exact comparisons. Library, tokenizer, kernel, and precision differences can still affect results; these commands specify the protocol rather than promising bitwise agreement across environments.

The CPU quick-start example uses WikiText-2 training text with 16 × 64 calibration tokens; the general `prune` command defaults to 128 × 64. Both are smaller than the paper's calibration setting. C4's first training shard is a substantial download; reuse the Hugging Face cache across runs.

## Matched methods and allocation controls

Reuse the saved calibration file on a fresh copy of the same original checkpoint for each method:

```bash
bash examples/compare_budgets.sh \
  meta-llama/Llama-2-7b-hf \
  runs/llama2-7b-qk-s80/calibration.npy \
  runs/llama2-7b-controls 0.8 --dtype float16 --attn-implementation eager
```

The script runs row-wise Wanda, matrix-budget Wanda, separate-budget QK-Wanda, and shared-budget QK-Wanda. All use the same token IDs, sparsity, and ceiling convention. It evaluates WikiText-2 PPL and saves every mask. Shared budgets select a fraction of the **combined parameter count**, not the mean of the Q and K sparsities. For hybrid full-block pruning, add `--scope block --save-model`; non-Q/K projection budgets stay row-wise.

## Perplexity

WikiText-2 uses `Salesforce/wikitext`, `wikitext-2-raw-v1`, **test**, joined with two newlines. C4 uses the first 1100 texts in `en/c4-validation.00000-of-00008.json.gz`, joined with spaces, and the first **256 × 2048** tokens. Both use complete nonoverlapping 2048-token contexts; each predicts 2047 next tokens. For C4, explicitly retain the 256-sequence limit:

```bash
qk-wanda evaluate --model runs/llama2-7b-qk-s80/model \
  --dtype float16 --eval-dataset wikitext2 \
  --output runs/llama2-7b-qk-s80/wikitext2.json
qk-wanda evaluate --model runs/llama2-7b-qk-s80/model \
  --dtype float16 --eval-dataset c4 --eval-max-sequences 256 \
  --output runs/llama2-7b-qk-s80/c4.json
```

Dense and pruned evaluations should use the same dtype and protocol. `--eval` in `prune` evaluates both on the same token stream. An explicit `--eval-max-sequences` smaller than the full protocol is useful for smoke tests but is not a directly comparable benchmark result.

## Zero-shot evaluation

Save the sparse checkpoint with `--save-model`. Install the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.12/README.md) in a separate evaluation environment, keeping Transformers below version 5:

```bash
python -m pip install 'lm_eval[hf]==0.4.12' 'transformers>=4.45.2,<5'
lm_eval --model hf \
  --model_args pretrained=runs/llama2-7b-qk-s80/model,dtype=float16 \
  --tasks boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
  --num_fewshot 0 --batch_size 4 --device cuda:0 --seed 0 \
  --output_path runs/llama2-7b-qk-s80/zero-shot
```

Use `acc` for BoolQ, RTE, and WinoGrande; use `acc_norm` for HellaSwag, ARC-Easy, ARC-Challenge, and OpenBookQA. Report the unweighted mean of those seven task scores, multiplied by 100 for percentage accuracy. Keep the harness version and task configurations with your results. The harness controls each task's evaluation split; this suite mixes validation and test splits.

## Reproducible artifacts

The public source package contains no pretrained weights or calibration corpus. Run outputs are ignored by Git. Preserve `report.json`, `masks.npz`, the model revision, and (when appropriate to share) `calibration.npy`. Mask replay validates names and shapes; the user must also choose the same original checkpoint/revision, since a same-shaped model may contain different weights.
