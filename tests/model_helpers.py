"""Small, randomly initialized model fixtures for offline automated tests."""

import torch


def tiny_model(family="llama", seed=0):
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        OPTConfig,
        OPTForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    torch.manual_seed(seed)
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    if family == "opt":
        config = OPTConfig(
            **kwargs, ffn_dim=64, dropout=0, attention_dropout=0, word_embed_proj_dim=16
        )
        constructor = OPTForCausalLM
    else:
        cfg, constructor = {
            "llama": (LlamaConfig, LlamaForCausalLM),
            "qwen2": (Qwen2Config, Qwen2ForCausalLM),
            "mistral": (MistralConfig, MistralForCausalLM),
        }[family]
        config = cfg(**kwargs, intermediate_size=64, num_key_value_heads=2, attention_dropout=0)
    config._attn_implementation = "eager"
    return constructor(config).eval()
