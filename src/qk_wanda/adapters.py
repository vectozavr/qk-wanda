"""Block input capture and replay for supported Hugging Face decoder models."""

from typing import Optional, Tuple

import torch
from torch import nn

from .scoring import _ensure_batched


def model_family(model) -> str:
    """Return the supported decoder family without importing Transformers."""

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type == "opt":
        return "opt"
    if hasattr(getattr(model, "model", None), "layers"):
        return model_type or "llama_like"
    if hasattr(getattr(getattr(model, "model", None), "decoder", None), "layers"):
        return model_type or "decoder_like"
    raise ValueError(
        "QK-Wanda requires decoder blocks at model.model.layers "
        "(Llama-style) or model.model.decoder.layers (OPT-style)"
    )


def decoder_layers(model):
    """Return the mutable decoder block sequence for a supported model."""

    model_body = getattr(model, "model", None)
    if hasattr(model_body, "layers"):
        return model_body.layers
    decoder = getattr(model_body, "decoder", None)
    if hasattr(decoder, "layers"):
        return decoder.layers
    # Keep the error text and family detection in one place.
    model_family(model)
    raise AssertionError("unreachable")


def _normalize_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str) and device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def input_device_for_model(model, fallback: torch.device = torch.device("cpu")):
    """Resolve the embedding/input device for Llama and OPT device maps."""

    device_map = getattr(model, "hf_device_map", {}) or {}
    for key in ("model.embed_tokens", "model.decoder.embed_tokens", ""):
        if key in device_map:
            return _normalize_device(device_map[key])

    model_body = getattr(model, "model", None)
    embedding = getattr(model_body, "embed_tokens", None)
    if embedding is None:
        embedding = getattr(getattr(model_body, "decoder", None), "embed_tokens", None)
    if embedding is not None:
        parameter = next(embedding.parameters(), None)
        if parameter is not None and not parameter.is_meta:
            return parameter.device
    return _normalize_device(fallback)


def _replay_kwarg_names(model) -> Tuple[str, ...]:
    if model_family(model) == "opt":
        return (
            "attention_mask",
            "layer_head_mask",
            "past_key_value",
            "output_attentions",
            "use_cache",
        )
    return (
        "attention_mask",
        "position_ids",
        "cache_position",
        "position_embeddings",
    )


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _slice_batch_value(value, batch_index: int, batch_size: int):
    if torch.is_tensor(value):
        # cache_position is one-dimensional and has no batch axis.  Tensors
        # with an explicit leading batch dimension are sliced per sequence.
        if value.ndim >= 2 and value.shape[0] == batch_size:
            return value[batch_index : batch_index + 1].detach()
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_slice_batch_value(item, batch_index, batch_size) for item in value)
    if isinstance(value, list):
        return [_slice_batch_value(item, batch_index, batch_size) for item in value]
    return value


def _same_forward_context(left, right) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _same_forward_context(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_forward_context(left[key], right[key]) for key in left
        )
    return left == right


class _CalibrationCapture(Exception):
    pass


@torch.no_grad()
def prepare_qk_calibration_input(
    model,
    dataloader,
    nsamples: int,
    device: torch.device,
    seqlen: Optional[int] = None,
    storage_device=None,
):
    """Capture fixed-length inputs and positional kwargs for the first block."""

    if nsamples <= 0:
        raise ValueError("nsamples must be positive")

    capture_seqlen = getattr(model, "seqlen", None) if seqlen is None else seqlen
    if not isinstance(capture_seqlen, int) or capture_seqlen <= 0:
        raise ValueError("seqlen must be a positive integer")

    layers = decoder_layers(model)
    old_use_cache = model.config.use_cache
    model.config.use_cache = False
    embedding_device = input_device_for_model(model, device)
    replay_kwarg_names = _replay_kwarg_names(model)

    dtype = next(iter(model.parameters())).dtype
    inputs = torch.zeros(
        (nsamples, capture_seqlen, model.config.hidden_size),
        dtype=dtype,
        device=embedding_device if storage_device is None else storage_device,
    )
    cache = {"index": 0, "forward_kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            # Qwen2 >= 4.55 selects a mask using this block attribute.
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, hidden_states, **kwargs):
            hidden_states = _ensure_batched(hidden_states, "decoder hidden_states")
            if hidden_states.shape[1] != capture_seqlen:
                raise ValueError(
                    "captured decoder sequence length "
                    f"{hidden_states.shape[1]} does not match requested {capture_seqlen}"
                )
            remaining = nsamples - cache["index"]
            take = min(hidden_states.shape[0], remaining)
            inputs[cache["index"] : cache["index"] + take].copy_(hidden_states[:take])
            for batch_index in range(take):
                sample_kwargs = {
                    key: _slice_batch_value(kwargs[key], batch_index, hidden_states.shape[0])
                    for key in replay_kwarg_names
                    if key in kwargs and kwargs[key] is not None
                }
                if cache["forward_kwargs"] is None:
                    cache["forward_kwargs"] = sample_kwargs
                elif not _same_forward_context(cache["forward_kwargs"], sample_kwargs):
                    raise ValueError(
                        "QK-Wanda's model integration requires identical attention masks "
                        "and decoder position context for every calibration sequence; "
                        "use the fixed-length unpadded protocol or call the tensor scorer "
                        "directly for per-example contexts"
                    )
            cache["index"] += take
            raise _CalibrationCapture

    original_layer = layers[0]
    layers[0] = Catcher(original_layer)
    try:
        for batch in dataloader:
            if cache["index"] >= nsamples:
                break
            if isinstance(batch, dict):
                model_inputs = {
                    key: value.to(embedding_device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                try:
                    model(**model_inputs)
                except _CalibrationCapture:
                    pass
            else:
                token_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
                try:
                    model(token_ids.to(embedding_device))
                except _CalibrationCapture:
                    pass
    finally:
        layers[0] = original_layer
        model.config.use_cache = old_use_cache

    if cache["index"] != nsamples:
        raise RuntimeError(f"captured {cache['index']} calibration sequences, expected {nsamples}")
    outputs = torch.zeros_like(inputs)
    return inputs, outputs, cache["forward_kwargs"] or {}


def _head_layout(model, decoder_layer, query: nn.Linear, key: nn.Linear) -> Tuple[int, int, int]:
    attention = getattr(decoder_layer, "self_attn", None)
    num_query_heads = getattr(attention, "num_heads", None)
    num_key_value_heads = getattr(attention, "num_key_value_heads", None)
    if num_query_heads is None:
        num_query_heads = getattr(model.config, "num_attention_heads", None)
    if num_key_value_heads is None:
        num_key_value_heads = getattr(model.config, "num_key_value_heads", num_query_heads)
    if not num_query_heads or not num_key_value_heads:
        raise ValueError("could not infer query/key head counts from the model")
    if query.weight.shape[0] % num_query_heads or key.weight.shape[0] % num_key_value_heads:
        raise ValueError(
            "projection output widths are incompatible with the configured head counts"
        )
    query_head_dim = query.weight.shape[0] // num_query_heads
    key_head_dim = key.weight.shape[0] // num_key_value_heads
    if query_head_dim != key_head_dim:
        raise ValueError(
            f"query head_dim {query_head_dim} does not match key head_dim {key_head_dim}"
        )
    return num_query_heads, num_key_value_heads, query_head_dim


def _rope_for_layer(decoder_layer, hidden_states: torch.Tensor, forward_kwargs):
    position_embeddings = forward_kwargs.get("position_embeddings")
    if position_embeddings is not None:
        if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
            raise ValueError("position_embeddings must be a (cos, sin) pair")
        return position_embeddings
    attention = getattr(decoder_layer, "self_attn", None)
    rotary = getattr(attention, "rotary_emb", None)
    position_ids = forward_kwargs.get("position_ids")
    if rotary is None or position_ids is None:
        raise ValueError(
            "QK-Wanda-MR could not obtain the model's RoPE tensors; use Transformers 4.45.2 "
            "or provide decoder position_embeddings"
        )
    return rotary(hidden_states, position_ids)
