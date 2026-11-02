import os
from typing import List

import numpy as np
import torch
from safetensors.torch import load_file
import onnx
from onnx import numpy_helper


def merge_loras(loras: List[str], scales: List[str]) -> dict:
    refit_dict = {}
    for lora, scale in zip(loras, scales):
        lora_dict = load_file(lora)
        for k, v in lora_dict.items():
            # Accumulate in fp32 (also handles fp8/bf16 files); refit casts to the engine's dtype.
            v = v.float()
            if k in refit_dict:
                refit_dict[k] += scale * v
            else:
                refit_dict[k] = scale * v
    return refit_dict


def apply_loras(base_path: str, loras: List[str], scales: List[str]) -> dict:
    """Full refit weights (base + LoRA deltas) for every weight the LoRAs touch."""
    refit_dict = merge_loras(loras, scales)
    # Don't pull the whole model (~5 GB for SDXL) into RAM: read only the initializers we need.
    base = onnx.load(base_path, load_external_data=False)
    onnx_opt_dir = os.path.dirname(base_path)

    for initializer in base.graph.initializer:
        if initializer.name not in refit_dict:
            continue
        wt = refit_dict[initializer.name]
        initializer_data = numpy_helper.to_array(initializer, base_dir=onnx_opt_dir).astype(np.float16)
        # fp16 halves the RAM held for the refit; the engine converts to its own weight dtype anyway.
        refit_dict[initializer.name] = (torch.from_numpy(initializer_data).float() + wt).half().contiguous()

    return refit_dict
