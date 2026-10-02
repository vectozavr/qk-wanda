# QK-Wanda: Coupling Queries and Keys for Pruning

[![arXiv](https://img.shields.io/badge/arXiv-2610.01554-b31b1b.svg)](https://arxiv.org/abs/2610.01554)

![Wanda uses row-wise budgets; QK-Wanda adds opposite-projection factors and shares the budget across query and key weights.](assets/wanda-vs-qk-wanda.jpg)

QK-Wanda augments Wanda scores with query–key interactions, allowing a shared pruning budget across query and key weights. It requires calibration forward passes, without gradients, retraining, or updates to retained weights. The default method reconstructs all QK products before rotary position embeddings (RoPE), without a causal mask or centering, as in the paper. The illustration uses this same objective for one token and one head.

## Citation

If you use QK-Wanda in your work, please cite the [accompanying paper](https://arxiv.org/abs/2610.01554):

```bibtex
@misc{ilin2026qkwanda,
  title         = {{QK-Wanda}: Coupling Queries and Keys for Unstructured Pruning},
  author        = {Ilin, Ivan and Richt{\'a}rik, Peter},
  year          = {2026},
  eprint        = {2610.01554},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2610.01554},
  url           = {https://arxiv.org/abs/2610.01554}
}
```

See [CITATION.cff](CITATION.cff) for software metadata and the preferred paper citation.

## Install

Requires Python **3.10+** and Transformers **4.45.2–4.x**.

```bash
git clone https://github.com/vectozavr/qk-wanda.git
cd qk-wanda
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data]'
```

For GPU execution, install the [appropriate CUDA build of PyTorch](https://pytorch.org/get-started/locally/) in this environment before installing QK-Wanda.

## Quick start

Prune a pretrained Qwen2.5-0.5B model on CPU:

```bash
qk-wanda prune \
  --model Qwen/Qwen2.5-0.5B \
  --device cpu \
  --sparsity 0.5 \
  --output runs/qwen-0.5b-cpu
```

This removes **50% of the combined query and key weights in each transformer block**, leaving other parameters unchanged. Change `--model` to another supported Hugging Face ID or local checkpoint; use `--device cuda:0` for GPU execution. The CPU example needs several GB of RAM. Weights and calibration data download on the first run and are cached afterward.

In the illustration, both methods remove four of eight weights. The resulting squared QK reconstruction error is **81 for Wanda and 9 for QK-Wanda**. This example illustrates the criterion; joint reconstruction is not guaranteed to improve for every mask or model.

The output directory must be new or empty. It contains:

- `report.json`: settings, calibration provenance, and per-block removal counts.
- `masks.npz`: compressed pruning masks, without model weights.

Add `--save-model` to save a Hugging Face checkpoint in `model/`, or `--save-calibration` to save the sampled token IDs in `calibration.npy`.

## Evaluation

Evaluate the pruned model by applying its masks to the original checkpoint:

```bash
qk-wanda evaluate \
  --model Qwen/Qwen2.5-0.5B \
  --device cpu \
  --masks runs/qwen-0.5b-cpu/masks.npz
```

By default, this measures perplexity on the WikiText-2 test set with 2048-token contexts. Omit `--masks` to evaluate the dense model. For a short check, add `--eval-seqlen 256 --eval-max-sequences 4` to both evaluations. Add `--output results.json` to save the result.

Alternatively, add `--eval` to the pruning command to evaluate both dense and pruned models in one run. If you saved a checkpoint, pass its `model/` directory to `--model` without `--masks`.

See the [reproduction guide](docs/reproduction.md) for the paper's calibration settings, C4 perplexity, and seven-task zero-shot evaluation.

## Comparison with Wanda

Run the original row-wise Wanda baseline with the same model, sparsity, and calibration settings:

```bash
qk-wanda prune \
  --model Qwen/Qwen2.5-0.5B \
  --device cpu \
  --sparsity 0.5 \
  --method wanda \
  --output runs/qwen-0.5b-wanda
```

Evaluate its masks using the command above, replacing the mask path. Add `--budget separate` to test Wanda with a budget per projection matrix. Default calibration sampling is deterministic; for exact reuse across runs, save it with `--save-calibration` and pass the resulting file via `--calibration-tokens`.

## Supported models

| Model family | Examples |
| --- | --- |
| Llama | Llama 2, Llama 3/3.1/3.2, TinyLlama |
| Qwen2 | Qwen2, Qwen2.5 |
| Mistral | Standard Mistral decoders with separate QK projections |
| OPT | OPT checkpoints |

Both multi-head and grouped-query attention are supported. Mistral and OPT are implementation extensions beyond the paper's benchmarks. See [model support](docs/model-support.md) for tested versions and architecture restrictions.

## Options

- **Calibration:** defaults to 128 WikiText-2 training sequences of 64 tokens, with seed 0. Change `--calibration`, `--nsamples`, `--seqlen`, or `--seed`; use `--calibration-text file.txt` for your own text or `--calibration-tokens tokens.npy` for saved token IDs.
- **Budget:** QK-Wanda defaults to a shared QK budget per block. `--budget separate` assigns a budget to each projection matrix; `--budget row` assigns one to each output row. Wanda defaults to row-wise budgets and supports separate matrix budgets, but not shared QK budgets.
- **Scope:** `--scope block` also prunes value, output, and MLP projections using Wanda; the default prunes only query and key weights.
- **Scoring:** `--variant unmasked` is the default QK-Wanda method. `--variant causal` selects QK-Wanda-M. `--variant rope` selects QK-Wanda-MR, which includes both causal masking and RoPE (unavailable for OPT). The model's attention computation stays unchanged for every scoring variant.
- **Precision and memory:** `--dtype` overrides automatic precision; `--cpu-mask-sort` moves mask sorting to CPU to reduce GPU memory use.

Run `qk-wanda prune --help` or `qk-wanda evaluate --help` for all flags. For integration into your own code, see the [Python API](docs/python-api.md); the [algorithm notes](docs/algorithm.md) explain scoring and mask selection.

## Development tests

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Tests run offline with tiny model fixtures and check scores, pruning masks, model integrations, checkpoint reloads, and evaluation.

Version 0.2.0 changes the default from causal scoring to the paper's unmasked method. To reproduce a version 0.1.0 QK-Wanda run, explicitly use `--variant causal`. See [release notes](CHANGELOG.md).

## Credits

This implementation builds on [Wanda](https://github.com/locuslab/wanda) and its blockwise pruning workflow, which in turn builds on [SparseGPT](https://github.com/IST-DASLab/sparsegpt). The original copyright notice is retained in [LICENSE](LICENSE); see [NOTICE](NOTICE) for attribution.
