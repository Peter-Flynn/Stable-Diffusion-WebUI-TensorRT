#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import os

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from modules import sd_hijack

from datastructures import ProfileSettings


class UNetModel:
    num_xl_classes = 2816
    dyn_axes = {
        "sample": {0: "2B", 2: "H", 3: "W"},
        "encoder_hidden_states": {0: "2B", 1: "77N"},
        "timesteps": {0: "2B"},
        "latent": {0: "2B", 2: "H", 3: "W"},
        "y": {0: "2B"},
    }

    def __init__(self, unet, embedding_dim: int, is_xl: bool = False):
        self.unet = unet
        self.embedding_dim = embedding_dim
        self.is_xl = is_xl
        self.in_channels = unet.in_channels

    def apply_torch_model(self):
        def disable_checkpoint(module):
            for attr in ("use_checkpoint", "checkpoint"):
                if getattr(module, attr, None) is True:
                    setattr(module, attr, False)

        self.unet.apply(disable_checkpoint)
        sd_hijack.model_hijack.apply_optimizations("None")

    def is_fp16(self) -> bool:
        return all(p.dtype == torch.float16 for p in self.unet.parameters() if p.is_floating_point())

    def get_input_names(self):
        return ["sample", "timesteps", "encoder_hidden_states"] + (["y"] if self.is_xl else [])

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {name: self.dyn_axes[name] for name in self.get_input_names() + self.get_output_names()}

    def get_sample_input(self, batch_size, latent_height, latent_width, text_len, device="cuda", dtype=torch.float32):
        """Timesteps stay fp32: fp16 can't hold fractional timesteps near 1000."""
        inputs = [
            torch.randn(batch_size, self.in_channels, latent_height, latent_width, dtype=dtype, device=device),
            torch.rand(batch_size, dtype=torch.float32, device=device) * 999,
            torch.randn(batch_size, text_len, self.embedding_dim, dtype=dtype, device=device),
        ]
        if self.is_xl:
            inputs.append(torch.randn(batch_size, self.num_xl_classes, dtype=dtype, device=device))
        return tuple(inputs)

    def get_input_profile(self, profile: ProfileSettings) -> dict:
        batch = profile.get_batch_dim()
        h_min, h_opt, h_max, w_min, w_opt, w_max = profile.get_latent_dim()
        shapes = {
            "sample": [(b, self.in_channels, h, w) for b, h, w in zip(batch, (h_min, h_opt, h_max), (w_min, w_opt, w_max))],
            "timesteps": [(b,) for b in batch],
            "encoder_hidden_states": [(b, t, self.embedding_dim) for b, t in zip(batch, (profile.t_min, profile.t_opt, profile.t_max))],
        }
        if self.is_xl:
            shapes["y"] = [(b, self.num_xl_classes) for b in batch]
        return shapes

    def export_weights_map(self, onnx_path: str, weights_map_path: str):
        """Maps UNet weight names to ONNX initializers by their fp16 data (transposed ones too)."""
        onnx_dir = os.path.dirname(onnx_path)
        by_hash = {}
        for init in onnx.load(onnx_path, load_external_data=False).graph.initializer:
            data = numpy_helper.to_array(init, base_dir=onnx_dir).astype(np.float16)
            by_hash.setdefault(hash(data.tobytes()), []).append((init.name, list(data.shape)))

        state_dict = self.unet.state_dict()
        names, shapes = {}, {}
        for name, weight in state_dict.items():
            wt = weight.detach().to(torch.float16).cpu().numpy()
            for transposed, data in ((False, wt), (True, wt.T)):
                candidates = by_hash.get(hash(data.tobytes()))
                if candidates:
                    match = next((c for c in candidates if c[0] == name), candidates[0])
                    candidates.remove(match)
                    names[name], shapes[name] = match[0], (match[1], transposed)
                    break

        unmatched = [n for n in state_dict if n not in names]
        print(f"[TensorRT] UNet: {len(names)} of {len(state_dict)} weights matched ONNX initializers", flush=True)
        if len(unmatched) > 0.02 * len(state_dict):
            raise RuntimeError(
                f"{len(unmatched)} UNet weights (e.g. {unmatched[0]}) don't match the exported ONNX model, so "
                "LoRAs couldn't be applied to them. The ONNX model was probably exported while a LoRA was "
                f"active, or from a different version of this checkpoint. Delete {os.path.basename(onnx_path)} "
                "(models/Unet-onnx) and re-export the engine, then convert the LoRA again."
            )
        with open(weights_map_path, "w") as f:
            json.dump([names, shapes], f)
