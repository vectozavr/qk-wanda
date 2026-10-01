# Python API

First save calibration tokens by adding `--save-calibration` to the README pruning command. The example below uses the resulting `runs/qwen-0.5b-cpu/calibration.npy`.

```python
import torch
from transformers import AutoModelForCausalLM
from qk_wanda import MaskArchive, apply_masks, prune_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32).eval()

# Integer tensor [N, T]: fixed-length independent sequences, without padding.
import numpy as np

tokens = torch.from_numpy(np.load("runs/qwen-0.5b-cpu/calibration.npy"))
archive = MaskArchive()
report = prune_model(model, tokens, sparsity=0.5, mask_callback=archive)
archive.save("masks.npz")
model.save_pretrained("pruned-model", safe_serialization=True)

# On a fresh copy of the same original model: apply_masks(fresh_model, "masks.npz")
```

`prune_model` changes the supplied model in place. It selects both QK masks before applying either, then passes the pruned block's output to the next block. The [algorithm notes](algorithm.md) describe the score and normalization conventions.

The default is `variant="unmasked"`, the paper's QK-Wanda method, with a shared QK budget. To run QK-Wanda-M, pass `variant="causal"`; `variant="rope"` runs QK-Wanda-MR. The report records both the option in `variant` and its paper name in `scoring_label`. These options change scoring, not the model's attention computation.

## Memory and reproducibility

- **Memory:** model weights must remain resident on CPU/GPU; complete blocks may be distributed across GPUs. Disk/CPU weight offloading through Accelerate is unsupported. Calibration hidden states are stored on CPU by default, in two buffers of approximately `N × T × hidden_size × bytes_per_element` each.
- **GPU sorting memory:** `--cpu-mask-sort` moves stable sorting to CPU. It needs additional RAM and transfer time.
- **Calibration:** `--calibration-text file.txt` samples one document per nonempty line. Alternatively, provide an integer `[N, T]` `.npy` with `--calibration-tokens`. No padding or cross-sequence attention is introduced.
- **Numerics:** scoring accumulates in FP32. For models producing nonfinite FP16 activations, use `--dtype bfloat16` or `float32`. Device, dtype, library version, and calibration changes may change masks.
- **Reproducibility:** pin `--revision`, save calibration IDs, and keep `report.json` with the masks. Smaller local reconstruction loss does not guarantee better downstream quality.
