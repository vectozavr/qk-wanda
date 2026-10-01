# Model support and testing

The adapters target standard Hugging Face `LlamaForCausalLM`, `Qwen2ForCausalLM`, `MistralForCausalLM`, and `OPTForCausalLM` classes. A matching model name alone is insufficient: the model must expose the standard separate linear QK projections and regular contiguous query-to-key groups.

## Supported paths

- Default QK-Wanda and causal QK-Wanda-M scoring: Llama, Qwen2, Mistral, OPT.
- Causal RoPE-aware QK-Wanda-MR scoring: Llama, Qwen2, Mistral.
- MHA and GQA; projection biases remain fixed.
- QK-only pruning, and hybrid block pruning with Wanda on other projections.
- CPU-resident models and complete blocks placed on GPUs, including a multi-GPU `device_map`.
- Fixed-length, unpadded calibration sequences; batch sizes greater than one and final partial batches.
- OPT's embedding projection dimension may differ from its hidden dimension.

The model must fit in resident CPU/GPU memory. Accelerate disk offload and dynamic CPU weight offload are unsupported. A CPU-resident model is supported; an offload hook that replaces its weights by meta tensors is not. Use `--cpu-mask-sort` to reduce GPU sorting memory.

For QK-Wanda-M and QK-Wanda-MR, Mistral's calibration length must not exceed its sliding attention window; otherwise the causal score would include inaccessible token pairs. The base QK-Wanda objective intentionally uses all token pairs and has no such scoring restriction. Every variant retains the model's native attention window and must respect its maximum context length. See the [Transformers Mistral documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/mistral) for the architecture.

## Unsupported layouts

The package rejects unsupported model types, fused QKV, QK normalization, quantized checkpoints, and tensor-parallel projection slicing (`pretraining_tp != 1`). It does not execute custom remote code. Newer architectures such as Qwen3 and Gemma need their own objective/layout review and adapter; renaming the model type is not sufficient.

## Validation

Offline tests use tiny randomly initialized versions of all four real Hugging Face model classes. They check:

- scores against explicit reconstruction errors for every single-coordinate deletion, including causal masking, GQA, biases, and RoPE;
- shared-budget masks against exhaustive small-case optimization and deterministic tie rules;
- preservation of every retained weight and every excluded parameter;
- batch-size invariance, sparse block propagation, and valid forward outputs;
- compressed mask replay and checkpoint save/reload;
- chunked perplexity against the model's native loss;
- the CLI end to end with a local tokenizer, model, calibration IDs, and evaluation text.

The package is tested with Transformers **4.45.2 and 4.57.6**, Python 3.11, and PyTorch 2.11 on CPU. CI covers both Transformers versions. Transformers 5 is outside the supported range. CUDA, multi-GPU execution, and large pretrained-model benchmarks require suitable hardware; the offline tests do not establish their performance or memory requirements.

The CPU pruning workflow was also checked end to end with the actual pretrained `Qwen/Qwen2.5-0.5B` checkpoint (revision `060db6499f32faf8b98477b0a26969ef7d8b9987`), FP32 weights, Transformers 4.57.6, 16 × 64 calibration tokens, and four 256-token WikiText-2 evaluation windows. It pruned all 24 blocks at 50% combined QK sparsity and produced finite dense/pruned perplexities. This small experiment verifies the pretrained workflow; the paper's evaluation uses the longer protocol in the reproduction guide.
