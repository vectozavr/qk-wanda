"""Exact single-weight deletion scores for Q/K reconstruction.

Adapted from the QK-Wanda research implementation. No Transformers dependency
is needed for the tensor scorer. Scores include the common 1/head_dim factor
for dot products scaled by 1/sqrt(head_dim); it does not change mask rankings.
"""

from typing import Optional, Sequence, Tuple

import torch
from torch import nn

QK_WANDA_VARIANTS = ("unmasked", "causal", "rope")


def _ensure_batched(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D] or [T, D], got {tuple(tensor.shape)}")
    return tensor


def _reshape_heads(
    tensor: torch.Tensor,
    num_heads: int,
    head_dim: int,
    name: str,
) -> torch.Tensor:
    expected = num_heads * head_dim
    if tensor.shape[-1] != expected:
        raise ValueError(
            f"{name} has output width {tensor.shape[-1]}, expected "
            f"{num_heads} heads * {head_dim} dimensions = {expected}"
        )
    return tensor.reshape(*tensor.shape[:-1], num_heads, head_dim)


def _normalize_rope_tensor(
    tensor: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    head_dim: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Hugging Face Llama normally supplies [B, T, d_h].  Accept the common
    # broadcast forms as well so the tensor utility is easy to use in tests.
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, d_h], got {tuple(tensor.shape)}")
    if tensor.shape[1:] != (sequence_length, head_dim):
        raise ValueError(
            f"{name} has shape {tuple(tensor.shape)}; expected [B, {sequence_length}, {head_dim}]"
        )
    if tensor.shape[0] == 1 and batch_size != 1:
        tensor = tensor.expand(batch_size, -1, -1)
    elif tensor.shape[0] != batch_size:
        raise ValueError(f"{name} batch size {tensor.shape[0]} does not match inputs {batch_size}")
    return tensor.detach().to(device=device, dtype=dtype)


def apply_llama_rope(
    states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply the split-half RoPE layout used by Hugging Face Llama.

    ``states`` must be ``[B, T, H, d_h]``.  Hugging Face's ``rotate_half`` pairs raw coordinate
    ``r`` with ``r + d_h/2``.  QK-Wanda-R must follow this real model layout to
    score deletion of raw projection rows correctly.
    """

    if states.ndim != 4:
        raise ValueError(f"states must have shape [B, T, H, d_h], got {tuple(states.shape)}")
    batch_size, sequence_length, _, head_dim = states.shape
    if head_dim % 2:
        raise ValueError(f"Llama RoPE requires an even head dimension, got {head_dim}")

    cos = _normalize_rope_tensor(
        cos, batch_size, sequence_length, head_dim, "cos", states.device, states.dtype
    )
    sin = _normalize_rope_tensor(
        sin, batch_size, sequence_length, head_dim, "sin", states.device, states.dtype
    )
    half = head_dim // 2
    rotated_half = torch.cat((-states[..., half:], states[..., :half]), dim=-1)
    return states * cos.unsqueeze(2) + rotated_half * sin.unsqueeze(2)


class QKWandaAccumulator:
    """Accumulate exact single-deletion QK-Wanda interaction factors.

    Inputs and raw projection outputs use row-token shapes:

    * ``inputs``: ``[B, T, d_model]``
    * ``query_outputs``: ``[B, T, H_Q * d_h]``
    * ``key_outputs``: ``[B, T, H_K * d_h]``

    Batch elements are always treated as independent calibration sequences.
    They are never flattened together before the sequencewise energy products
    are formed.  Projection outputs may include bias; only weight entries are
    ultimately scored and pruned.
    """

    def __init__(
        self,
        num_query_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        variant: str = "causal",
        accumulation_dtype: torch.dtype = torch.float32,
    ):
        if variant not in QK_WANDA_VARIANTS:
            raise ValueError(f"variant must be one of {QK_WANDA_VARIANTS}, got {variant!r}")
        if num_query_heads <= 0 or num_key_value_heads <= 0 or head_dim <= 0:
            raise ValueError("head counts and head_dim must be positive")
        if num_query_heads % num_key_value_heads:
            raise ValueError(
                "QK-Wanda currently requires the contiguous regular GQA layout: "
                "num_query_heads must be divisible by num_key_value_heads"
            )
        if variant == "rope" and head_dim % 2:
            raise ValueError("QK-Wanda-R requires an even head_dim")

        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.query_heads_per_key = num_query_heads // num_key_value_heads
        self.variant = variant
        self.accumulation_dtype = accumulation_dtype

        self.input_dim: Optional[int] = None
        self.device: Optional[torch.device] = None
        self.num_sequences = 0
        self._total_weight: Optional[torch.Tensor] = None

        self._input_energies = []
        self._query_energies = []
        self._key_energies = []
        self._sequence_weights = []
        self._gamma_query: Optional[torch.Tensor] = None
        self._gamma_key: Optional[torch.Tensor] = None
        self._cached_factors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # State used by q_proj/k_proj forward hooks.
        self._pending_input: Optional[torch.Tensor] = None
        self._pending_query: Optional[torch.Tensor] = None
        self._pending_key: Optional[torch.Tensor] = None
        self._pending_rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @property
    def query_width(self) -> int:
        return self.num_query_heads * self.head_dim

    @property
    def key_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def _initialize(self, input_dim: int, device: torch.device) -> None:
        if self.input_dim is not None:
            if self.input_dim != input_dim:
                raise ValueError(f"input width changed from {self.input_dim} to {input_dim}")
            if self.device != device:
                raise ValueError(f"accumulator device changed from {self.device} to {device}")
            return

        self.input_dim = input_dim
        self.device = device
        self._total_weight = torch.zeros((), device=device, dtype=self.accumulation_dtype)
        if self.variant != "unmasked":
            self._gamma_query = torch.zeros(
                # The opposite-key interaction factor is identical for every
                # query head sharing a KV head.  Accumulate it once per key
                # head, then repeat it only when materializing the score.
                self.num_key_value_heads,
                self.head_dim,
                input_dim,
                device=device,
                dtype=self.accumulation_dtype,
            )
            self._gamma_key = torch.zeros(
                self.num_key_value_heads,
                self.head_dim,
                input_dim,
                device=device,
                dtype=self.accumulation_dtype,
            )

    def _weights_for_batch(
        self,
        batch_size: int,
        sequence_weights: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if sequence_weights is None:
            weights = torch.ones(batch_size, device=device, dtype=self.accumulation_dtype)
        else:
            weights = torch.as_tensor(
                sequence_weights, device=device, dtype=self.accumulation_dtype
            ).reshape(-1)
            if weights.numel() != batch_size:
                raise ValueError(
                    f"sequence_weights has {weights.numel()} values for batch size {batch_size}"
                )
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise ValueError("sequence_weights must be finite and nonnegative")
        if not bool((weights > 0).any()):
            raise ValueError("each added batch must have at least one positive sequence weight")
        return weights

    @staticmethod
    def _weighted_token_product(
        factors: torch.Tensor,
        input_squares: torch.Tensor,
        sequence_weights: torch.Tensor,
    ) -> torch.Tensor:
        # factors: [B, T, H, d_h], input_squares: [B, T, d_model]
        batch_size, sequence_length, num_heads, head_dim = factors.shape
        weighted_inputs = input_squares * sequence_weights[:, None, None]
        left = factors.reshape(batch_size * sequence_length, num_heads * head_dim).transpose(0, 1)
        right = weighted_inputs.reshape(batch_size * sequence_length, input_squares.shape[-1])
        return (left @ right).reshape(num_heads, head_dim, input_squares.shape[-1])

    def add_batch(
        self,
        inputs: torch.Tensor,
        query_outputs: torch.Tensor,
        key_outputs: torch.Tensor,
        sequence_weights: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> None:
        """Add one or more independent calibration sequences."""

        inputs = _ensure_batched(inputs.detach(), "inputs")
        query_outputs = _ensure_batched(query_outputs.detach(), "query_outputs")
        key_outputs = _ensure_batched(key_outputs.detach(), "key_outputs")

        if inputs.shape[:2] != query_outputs.shape[:2] or inputs.shape[:2] != key_outputs.shape[:2]:
            raise ValueError(
                "inputs, query_outputs, and key_outputs must have identical batch and token dimensions"
            )
        if inputs.device != query_outputs.device or inputs.device != key_outputs.device:
            raise ValueError("inputs and projection outputs must be on the same device")

        batch_size, sequence_length, input_dim = inputs.shape
        self._initialize(input_dim, inputs.device)
        weights = self._weights_for_batch(batch_size, sequence_weights, inputs.device)

        x = inputs.to(dtype=self.accumulation_dtype)
        q = _reshape_heads(
            query_outputs.to(dtype=self.accumulation_dtype),
            self.num_query_heads,
            self.head_dim,
            "query_outputs",
        )
        k = _reshape_heads(
            key_outputs.to(dtype=self.accumulation_dtype),
            self.num_key_value_heads,
            self.head_dim,
            "key_outputs",
        )

        if self.variant == "unmasked":
            self._input_energies.append(x.square().sum(dim=1))
            self._query_energies.append(q.square().sum(dim=1))
            self._key_energies.append(k.square().sum(dim=1))
            self._sequence_weights.append(weights)
        elif self.variant == "causal":
            self._add_causal(x, q, k, weights)
        else:
            if cos is None or sin is None:
                raise ValueError("QK-Wanda-R requires the model's RoPE cos and sin tensors")
            self._add_rope(x, q, k, weights, cos, sin)

        self.num_sequences += batch_size
        self._total_weight.add_(weights.sum())
        self._cached_factors = None

    def _add_causal(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        input_squares = x.square()

        key_prefix = k.square().cumsum(dim=1)
        self._gamma_query.add_(self._weighted_token_product(key_prefix, input_squares, weights))

        grouped_query_squares = (
            q.square()
            .reshape(
                q.shape[0],
                q.shape[1],
                self.num_key_value_heads,
                self.query_heads_per_key,
                self.head_dim,
            )
            .sum(dim=3)
        )
        query_suffix = grouped_query_squares.flip(1).cumsum(dim=1).flip(1)
        self._gamma_key.add_(self._weighted_token_product(query_suffix, input_squares, weights))

    @staticmethod
    def _rope_raw_coordinate_factors(
        covariance_00: torch.Tensor,
        covariance_01: torch.Tensor,
        covariance_11: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Return diag(R^T C R) in Hugging Face's split-half row order."""

        head_dim = cos.shape[-1]
        half = head_dim // 2
        # HF computes y0 = c0*x0 - s0*x1 and y1 = s1*x0 + c1*x1.
        c0 = cos[..., :half].unsqueeze(2)
        c1 = cos[..., half:].unsqueeze(2)
        s0 = sin[..., :half].unsqueeze(2)
        s1 = sin[..., half:].unsqueeze(2)

        raw_first = (
            c0.square() * covariance_00 + 2 * c0 * s1 * covariance_01 + s1.square() * covariance_11
        )
        raw_second = (
            s0.square() * covariance_00 - 2 * s0 * c1 * covariance_01 + c1.square() * covariance_11
        )
        return torch.cat((raw_first, raw_second), dim=-1)

    def _add_rope(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> None:
        batch_size, sequence_length = x.shape[:2]
        cos = _normalize_rope_tensor(
            cos,
            batch_size,
            sequence_length,
            self.head_dim,
            "cos",
            x.device,
            self.accumulation_dtype,
        )
        sin = _normalize_rope_tensor(
            sin,
            batch_size,
            sequence_length,
            self.head_dim,
            "sin",
            x.device,
            self.accumulation_dtype,
        )
        q_rotated = apply_llama_rope(q, cos, sin)
        k_rotated = apply_llama_rope(k, cos, sin)
        half = self.head_dim // 2
        input_squares = x.square()

        k0, k1 = k_rotated[..., :half], k_rotated[..., half:]
        key_covariance_00 = k0.square().cumsum(dim=1)
        key_covariance_01 = (k0 * k1).cumsum(dim=1)
        key_covariance_11 = k1.square().cumsum(dim=1)
        key_factors = self._rope_raw_coordinate_factors(
            key_covariance_00, key_covariance_01, key_covariance_11, cos, sin
        )
        self._gamma_query.add_(self._weighted_token_product(key_factors, input_squares, weights))

        q0 = q_rotated[..., :half].reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.query_heads_per_key,
            half,
        )
        q1 = q_rotated[..., half:].reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.query_heads_per_key,
            half,
        )
        query_covariance_00 = q0.square().sum(dim=3).flip(1).cumsum(dim=1).flip(1)
        query_covariance_01 = (q0 * q1).sum(dim=3).flip(1).cumsum(dim=1).flip(1)
        query_covariance_11 = q1.square().sum(dim=3).flip(1).cumsum(dim=1).flip(1)
        key_factors = self._rope_raw_coordinate_factors(
            query_covariance_00, query_covariance_01, query_covariance_11, cos, sin
        )
        self._gamma_key.add_(self._weighted_token_product(key_factors, input_squares, weights))

    def interaction_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Gamma_Q and Gamma_K, flattened like the projection weights."""

        if (
            self.num_sequences == 0
            or self._total_weight is None
            or not bool(self._total_weight > 0)
        ):
            raise RuntimeError("no calibration sequences have been accumulated")
        if self._cached_factors is not None:
            return self._cached_factors

        if self.variant == "unmasked":
            x_energy = torch.cat(self._input_energies, dim=0)
            q_energy = torch.cat(self._query_energies, dim=0)
            k_energy = torch.cat(self._key_energies, dim=0)
            weights = torch.cat(self._sequence_weights, dim=0)

            query_context = k_energy.reshape(x_energy.shape[0], self.key_width)
            gamma_query = query_context.transpose(0, 1) @ (x_energy * weights[:, None])

            grouped_query = q_energy.reshape(
                x_energy.shape[0],
                self.num_key_value_heads,
                self.query_heads_per_key,
                self.head_dim,
            ).sum(dim=2)
            gamma_key = grouped_query.reshape(x_energy.shape[0], self.key_width).transpose(0, 1) @ (
                x_energy * weights[:, None]
            )
        else:
            gamma_query = self._gamma_query
            gamma_key = self._gamma_key.reshape(self.key_width, self.input_dim)

        gamma_query = gamma_query.reshape(
            self.num_key_value_heads, self.head_dim, self.input_dim
        ).repeat_interleave(self.query_heads_per_key, dim=0)
        gamma_query = gamma_query.reshape(self.query_width, self.input_dim) / self._total_weight
        gamma_key = gamma_key / self._total_weight
        self._cached_factors = (gamma_query, gamma_key)
        return self._cached_factors

    def scores(
        self,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return squared QK-Wanda saliencies in projection-weight shape."""

        expected_query = (self.query_width, self.input_dim)
        expected_key = (self.key_width, self.input_dim)
        if tuple(query_weight.shape) != expected_query:
            raise ValueError(
                f"query_weight has shape {tuple(query_weight.shape)}, expected {expected_query}"
            )
        if tuple(key_weight.shape) != expected_key:
            raise ValueError(
                f"key_weight has shape {tuple(key_weight.shape)}, expected {expected_key}"
            )
        if query_weight.device != self.device or key_weight.device != self.device:
            raise ValueError("projection weights and accumulated statistics must share a device")

        gamma_query, gamma_key = self.interaction_factors()
        query_score = query_weight.detach().to(self.accumulation_dtype).square() * gamma_query
        key_score = key_weight.detach().to(self.accumulation_dtype).square() * gamma_key
        return query_score / self.head_dim, key_score / self.head_dim

    def set_next_rope(self, cos: torch.Tensor, sin: torch.Tensor) -> None:
        """Attach RoPE tensors to the next paired q_proj/k_proj hook capture."""

        if self.variant != "rope":
            return
        if self._pending_rope is not None:
            raise RuntimeError("the previous layer forward did not consume its RoPE tensors")
        self._pending_rope = (cos.detach(), sin.detach())

    @staticmethod
    def _hook_output(output: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(output):
            raise TypeError(f"{name} hook expected a tensor output, got {type(output).__name__}")
        return output.detach()

    def _capture_projection(
        self,
        kind: str,
        inputs: Sequence[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        if not inputs or not torch.is_tensor(inputs[0]):
            raise TypeError(f"{kind}_proj hook did not receive a tensor input")
        if kind == "query":
            if self._pending_query is not None:
                raise RuntimeError("q_proj ran twice before the paired k_proj hook")
            self._pending_query = self._hook_output(output, "q_proj")
        else:
            if self._pending_key is not None:
                raise RuntimeError("k_proj ran twice before the paired q_proj hook")
            self._pending_key = self._hook_output(output, "k_proj")

        if self._pending_input is None:
            self._pending_input = inputs[0].detach()
        elif self._pending_input.shape != inputs[0].shape:
            raise RuntimeError("q_proj and k_proj received differently shaped inputs")

        if self._pending_query is not None and self._pending_key is not None:
            cos = sin = None
            if self.variant == "rope":
                if self._pending_rope is None:
                    raise RuntimeError(
                        "set_next_rope must be called before every RoPE layer forward"
                    )
                cos, sin = self._pending_rope
            self.add_batch(
                self._pending_input,
                self._pending_query,
                self._pending_key,
                cos=cos,
                sin=sin,
            )
            self._pending_input = None
            self._pending_query = None
            self._pending_key = None
            self._pending_rope = None

    def capture_query(self, _module, inputs, output) -> None:
        self._capture_projection("query", inputs, output)

    def capture_key(self, _module, inputs, output) -> None:
        self._capture_projection("key", inputs, output)

    def assert_hooks_drained(self) -> None:
        if any(
            value is not None
            for value in (
                self._pending_input,
                self._pending_query,
                self._pending_key,
                self._pending_rope,
            )
        ):
            raise RuntimeError(
                "unpaired q_proj/k_proj hooks remain; the model may bypass projection modules "
                "(for example, pretraining_tp > 1 is unsupported)"
            )


class WandaInputAccumulator:
    """Standard Wanda input-energy accumulator for non-QK projections."""

    def __init__(self, layer: nn.Linear, sequence_length: Optional[int] = None):
        self.layer = layer
        self.sequence_length = sequence_length
        self.input_energy = torch.zeros(
            layer.weight.shape[1], device=layer.weight.device, dtype=torch.float32
        )
        self.num_sequences = 0

    def add_batch(self, inputs: torch.Tensor) -> None:
        # Some OPT versions flatten batch and token axes before MLP projections.
        if inputs.ndim == 2 and self.sequence_length is not None:
            if inputs.shape[0] % self.sequence_length:
                raise ValueError("Flattened Wanda input has an incomplete sequence")
            inputs = inputs.reshape(-1, self.sequence_length, inputs.shape[-1])
        inputs = _ensure_batched(inputs.detach(), "Wanda inputs").float()
        if inputs.shape[-1] != self.input_energy.numel():
            raise ValueError("Wanda hook input width does not match its linear layer")
        self.input_energy.add_(inputs.square().sum(dim=(0, 1)))
        self.num_sequences += inputs.shape[0]

    def capture(self, _module, inputs, _output) -> None:
        if not inputs or not torch.is_tensor(inputs[0]):
            raise TypeError("Wanda hook did not receive a tensor input")
        self.add_batch(inputs[0])

    def score(self) -> torch.Tensor:
        if self.num_sequences == 0:
            raise RuntimeError("the Wanda hook did not observe any calibration inputs")
        mean_energy = self.input_energy / self.num_sequences
        return self.layer.weight.detach().abs().float() * mean_energy.sqrt().reshape(1, -1)
