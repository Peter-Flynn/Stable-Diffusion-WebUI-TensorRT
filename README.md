# TensorRT Extension for Stable Diffusion WebUI (TensorRT 10 & 11)

A modernized fork of NVIDIA's TensorRT extension for Stable Diffusion WebUI, tested with
[stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) and
Automatic1111. Based on [ventus34's Blackwell fork](https://github.com/ventus34/Stable-Diffusion-WebUI-TensorRT-Blackwell),
with fixes collected from other community forks.

Supports Stable Diffusion 1.5, 2.1, SDXL, SDXL Turbo, and LCM.

---

## What's different from upstream

- **TensorRT 10 and 11.** TensorRT 11 removed weakly typed precision flags (`BuilderFlag.FP16`). On TensorRT 11 engines
  are built *strongly typed*: precision comes from the ONNX graph, which is exported under autocast, so convolutions and
  matmuls still run in FP16. On TensorRT 10 the classic FP16 build is used.
- **No Polygraphy at runtime.** Engines are built and run with the TensorRT Python API directly. `onnx-graphsurgeon` /
  `polygraphy` are optional and only used for ONNX graph cleanup before a build. `optimum` is no longer needed.
- **Lower VRAM use.** Activation memory is sized for the shapes actually in use (not the profile maximum) and comes from
  PyTorch's allocator, so it's shared with the rest of the WebUI. ONNX export traces at a small resolution.
- **Fewer copies per step.** Inputs that already match the engine's dtype are bound directly; outputs aren't cloned.
- **LoRA fixes.** LoRA export works on reForge/Forge (LoRAs are applied through the ModelPatcher there), refit weights
  match the engine's weight dtypes (required for strongly typed engines), `<lora:name>` without a weight works, LoRAs in
  subfolders are listed, and bf16/fp8 checkpoints no longer crash the weight mapping.
- **Non-destructive installer.** `install.py` never uninstalls or replaces your TensorRT. Set
  `TRT_EXTENSION_SKIP_INSTALL=1` to make it install nothing at all (useful for Nix or other managed environments).
- **reForge support.** Hooks `ldm_patched`'s `UNetModel.forward`, uses an ONNX-exportable attention during export, and
  forces the TorchScript ONNX exporter (`dynamo=False`).
- Respects `SD Unet: None`, shows errors in the UI instead of swallowing them, and warns when ControlNet inputs are
  ignored by a TensorRT engine.

---

## Requirements

- An NVIDIA GPU supported by your TensorRT version, with a recent driver.
- PyTorch with CUDA.
- TensorRT >= 10.0 (`tensorrt`, `tensorrt-cu12` or `tensorrt-cu13`). Blackwell (RTX 50-series) needs TensorRT >= 10.8.

**Engines are tied to the TensorRT version and GPU they were built with.** After upgrading TensorRT, rebuild your engines
(TensorRT tab → Advanced Settings → Force Rebuild). The extension reports this clearly if it can't load an engine.

---

## Installation

1. In the WebUI, open **Extensions → Install from URL** and paste this repository's URL.
2. Restart the WebUI. On startup `install.py` installs TensorRT (only if it's missing), `onnx`, `onnxscript`,
   `onnx-graphsurgeon` and `polygraphy`.

To install dependencies yourself instead:

```bash
pip install tensorrt onnx onnxscript onnx-graphsurgeon polygraphy --extra-index-url https://pypi.nvidia.com
```

---

## How to use

1. Select your checkpoint in the WebUI.
2. Open the **TensorRT** tab, pick a preset (or configure Advanced Settings) and click **Export**.
   The first build takes a few minutes while TensorRT profiles kernels; later builds reuse the timing cache.
3. Add `sd_unet` to **Settings → User Interface → Quicksettings list**, then select the `[TRT] ...` engine
   (or leave it on `Automatic`). Choose `None` to use PyTorch.

### LoRA

1. Open **TensorRT → TensorRT LoRA**, select a LoRA and click **Convert to TensorRT**.
2. Use it in your prompt as usual, e.g. `<lora:my_lora:0.8>`.

---

## Tips

- **Hires. fix**: use a dynamic engine that covers both the base and the upscaled resolution.
- **Resolution**: width and height should be divisible by 64.
- **Not supported by engines**: ControlNet and other UNet patches (FreeU, attention couple, ...). Set SD Unet to `None`
  when you need them.
