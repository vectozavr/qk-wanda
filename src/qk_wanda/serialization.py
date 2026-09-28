"""Portable bit-packed masks; loading never executes pickled Python objects."""

import json
from pathlib import Path

import numpy as np
import torch


class MaskArchive:
    """Pass an instance as prune_model(mask_callback=archive), then call save."""

    def __init__(self):
        self._packed = {}
        self._shapes = {}

    def __call__(self, name, mask):
        self._packed[name] = np.packbits(mask.detach().cpu().numpy().reshape(-1), bitorder="little")
        self._shapes[name] = list(mask.shape)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps({"schema_version": 1, "shapes": self._shapes})
        with path.open("wb") as file:
            np.savez_compressed(file, __metadata__=metadata, **self._packed)


@torch.no_grad()
def apply_masks(model, path):
    """Apply masks to the same original checkpoint used to create the archive."""
    parameters = dict(model.named_parameters())
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"]))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported mask archive version")
        shapes = metadata.get("shapes", {})
        if not shapes or set(archive.files) != set(shapes) | {"__metadata__"}:
            raise ValueError("Mask archive contents do not match its metadata")
        # Validate every entry before making any change.
        for name, shape in shapes.items():
            if name not in parameters or list(parameters[name].shape) != shape:
                raise ValueError(f"Mask/model mismatch: {name}")
            data = archive[name]
            expected = (parameters[name].numel() + 7) // 8
            if data.dtype != np.uint8 or data.ndim != 1 or data.size != expected:
                raise ValueError(f"Invalid packed mask: {name}")
        for name, shape in shapes.items():
            parameter = parameters[name]
            data = np.unpackbits(archive[name], bitorder="little", count=parameter.numel()).reshape(
                shape
            )
            parameter.masked_fill_(
                torch.as_tensor(data, device=parameter.device, dtype=torch.bool), 0
            )
    return model
