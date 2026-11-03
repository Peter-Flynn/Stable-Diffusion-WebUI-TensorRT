# TensorRT for reForge

Runs the Stable Diffusion UNet of [stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge)
on NVIDIA TensorRT engines. Supports SD 1.x, SD 2.x and SDXL-architecture checkpoints (including Pony, Illustrious,
NoobAI and other fine-tunes, v-prediction ones too) on RTX 2080 Ti and newer GPUs.

## How it works

- **Engines are compiled per architecture, not per checkpoint.** An engine is built without weights; the loaded
  checkpoint's weights are put into it when it loads. One SDXL engine runs every SDXL checkpoint, so switching
  checkpoints doesn't need a new export, and engine files are megabytes instead of gigabytes.
- **LoRAs just work.** The engine gets exactly the weights reForge would use for PyTorch, LoRAs (and any other weight
  patch reForge supports) included. When the LoRAs change, only the weights they touch are updated. There is no
  conversion step.
- **One engine, several profiles.** A profile is a size range, batch range and prompt length range. The default is
  every base resolution plus Hires. fix up to 2x of them, with prompts of up to 750 tokens; the engine switches between
  profiles instantly, with one copy of the weights in VRAM. CFG 1 (Lightning, DMD2, Turbo) runs at half the batch
  instead of padding.
- **reForge manages its memory.** The engine is a model of reForge's model manager, which loads it, makes room for
  it and unloads it when another model needs the VRAM, like the text encoder or VAE. While a pass runs on the
  engine, the PyTorch UNet stays in RAM.
- **Anything the engine can't run uses PyTorch:** sizes no profile covers, ControlNet, UNet patches (FreeU,
  attention couple, ...), or an engine that fails to load or runs out of VRAM. You get a notice; the image is the
  same as without TensorRT.

## Requirements

- reForge, with an NVIDIA RTX 2080 Ti or newer and a recent driver.
- TensorRT 10.1 or later. `install.py` installs TensorRT if it's missing (never replaces yours) and `onnx`. Set
  `TRT_EXTENSION_SKIP_INSTALL=1` to install nothing.

## Use

1. Load a checkpoint and open the **TensorRT** tab. It shows the checkpoint's architecture.
2. Pick profiles (presets cover common SDXL and SD 1.5 sizes) and click **Build engine**. The first build takes
   several minutes; later ones reuse the timing cache.
3. Set **SD Unet** (Settings → Stable Diffusion, or add `sd_unet` to the Quicksettings list) to `Automatic`. Every
   checkpoint of that architecture now uses the engine. `None` uses PyTorch.

Engines only load with the TensorRT version and GPU they were built with; after upgrading TensorRT, build again.
Engines live in `models/Unet-trt/<architecture>/`.

## Upgrading from earlier versions

Old engines, ONNX files and converted LoRAs aren't used anymore and can take many GB: the TensorRT tab offers to
delete them. Build an engine for each architecture you use; LoRAs need no conversion.
