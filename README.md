# TensorRT Extension for Stable Diffusion WebUI (TensorRT 10 & 11)

A modernized fork of NVIDIA's TensorRT extension for Stable Diffusion WebUI, tested with
[stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) and
Automatic1111. Based on [ventus34's Blackwell fork](https://github.com/ventus34/Stable-Diffusion-WebUI-TensorRT-Blackwell),
with fixes collected from other community forks.

Supports Stable Diffusion 1.5, 2.1, SDXL, SDXL Turbo, and LCM.

---

## What's different from upstream

- **TensorRT 10 and 11.** TensorRT 11 removed weakly typed precision flags (`BuilderFlag.FP16`): engines are built
  *strongly typed*, so precision comes from the ONNX graph. On reForge the UNet is exported in FP16 throughout (an
  autocast export leaves normalizations, softmax and residual adds in FP32, which made engines slower than PyTorch);
  A1111 exports under autocast. On TensorRT 10 the classic FP16 build is used.
- **No Polygraphy, Optimum or ONNX graph tools.** Engines are built and run with the TensorRT Python API directly;
  TensorRT does the graph cleanup itself.
- **Lower VRAM use.** Activation memory comes from PyTorch's allocator, so it's shared with the rest of the WebUI, and
  dynamic engines size it for the shapes in use rather than the profile maximum (TensorRT >= 10.10). ONNX export
  traces at a small resolution.
- **Fewer copies per step.** Inputs that already match the engine's dtype are bound directly; outputs aren't cloned.
- **LoRA fixes.** LoRA export works on reForge/Forge (LoRAs are applied through the ModelPatcher there), refit weights
  match the engine's weight dtypes (required for strongly typed engines), `<lora:name>` without a weight works, the UNet
  weight of `<lora:name:te:unet>` is honored, aliases from LoRA cards resolve, LoRAs in subfolders are listed, and bf16/fp8
  checkpoints no longer crash the weight mapping. On reForge, converting a LoRA only reads the weights it changes.
- **Non-destructive installer.** `install.py` never uninstalls or replaces your TensorRT. Set
  `TRT_EXTENSION_SKIP_INSTALL=1` to make it install nothing at all (useful for Nix or other managed environments).
- **reForge support.** Hooks `ldm_patched`'s `UNetModel.forward`, uses an ONNX-exportable attention during export, and
  forces the TorchScript ONNX exporter (`dynamo=False`).
- **Plays nicely with reForge's model manager.** The PyTorch UNet is unloaded through the manager (never moved behind
  its back, which caused "Expected all tensors to be on the same device" errors), and while an engine is active reForge
  keeps the UNet's weights in system RAM instead of loading them into VRAM next to the engine. The manager can't see
  TensorRT's memory, so the two take turns: it unloads models to make room for an engine and its activations, and an
  engine steps aside (and reloads for sampling) when the text encoder or another model wouldn't fully fit next to it.
- **Engines build in their own process.** TensorRT's builder holds on to VRAM after a build; in a separate process it
  all goes back to the driver when the build ends, so models load fully again afterwards. LoRA refits release the
  resources TensorRT caches for them.
- **Out-of-memory handling.** Builds report kernels TensorRT skipped for lack of VRAM (they make engines slower) and
  free models first. At runtime an OOM is retried once, then the generation finishes on PyTorch; inputs no engine
  profile covers also run on PyTorch instead of failing, and so does a generation whose engine output has NaN/Inf
  values. Load errors distinguish "out of memory" from "rebuild needed".
- **Profile selection that matches the sampler.** Prompt lengths are counted with the WebUI's own tokenizer (emphasis,
  `BREAK`, embeddings, prompt editing), reForge's cond/uncond batching is accounted for, and the same scoring is used before
  and during sampling, so engines aren't reloaded mid-generation.
- Respects `SD Unet: None`, shows errors in the UI instead of swallowing them, and warns when ControlNet inputs are
  ignored by a TensorRT engine. A LoRA that hasn't been converted (or doesn't match the engine), or an engine selected
  for another checkpoint, makes that generation use PyTorch instead of producing wrong images.
- SD Unet labels stay the same when a checkpoint gets more engines, so a saved selection keeps working.

---

## Requirements

- An NVIDIA GPU supported by your TensorRT version, with a recent driver.
- PyTorch with CUDA.
- TensorRT >= 10.0 (`tensorrt`, `tensorrt-cu12` or `tensorrt-cu13`). Blackwell (RTX 50-series) needs TensorRT >= 10.8.

**Engines are tied to the TensorRT version and GPU they were built with.** After upgrading TensorRT, rebuild your engines
(TensorRT tab → Advanced Settings → Force Rebuild). The extension reports this clearly if it can't load an engine.

**Upgrading from an earlier version on reForge:** export your engines again with the same settings. The first export
re-exports the ONNX model in FP16 and the engines are rebuilt from it (no Force Rebuild needed). SD 1.5/2.1 LoRAs need
converting again afterwards; SDXL ones keep working.

---

## Installation

1. In the WebUI, open **Extensions → Install from URL** and paste this repository's URL.
2. Restart the WebUI. On startup `install.py` installs TensorRT (only if it's missing), `onnx` and `onnxscript`.

To install dependencies yourself instead:

```bash
pip install tensorrt onnx onnxscript --extra-index-url https://pypi.nvidia.com
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

- **Hires. fix**: use a dynamic engine that covers both the base and the upscaled resolution. Otherwise the upscaled pass
  uses a second engine if one covers it, or PyTorch.
- **Resolution**: width and height should be divisible by 64.
- **Not supported by engines**: ControlNet and other UNet patches (FreeU, attention couple, ...). Set SD Unet to `None`
  when you need them.
