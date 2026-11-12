[adaptivediffusion]: https://jiakangyuan.github.io/AdaptiveDiffusion-project-page
[deepcache]: https://github.com/horseee/DeepCache
[nvidia]: https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT
[reforge]: https://github.com/Panchovix/stable-diffusion-webui-reForge
[tensorrt]: https://developer.nvidia.com/tensorrt

<h1 align="center">
	<a href="https://developer.nvidia.com/tensorrt">TensorRT</a> for
	<a href="https://github.com/Panchovix/stable-diffusion-webui-reForge">reForge</a>
</h1>

A fork of [NVIDIA's TensorRT extension][nvidia], rewritten for [reForge][] – runs SD&nbsp;1.x,
SD&nbsp;2.x, and SDXL models (Pony, Illustrious, NoobAI, etc.) on [TensorRT][], without building an
engine per checkpoint, or converting each LoRA.

## Installation
> Requires an RTX&nbsp;2080&nbsp;Ti or newer.

1. In [reForge][], open **Extensions → Install from URL**, and paste this repo.'s URL.
2. Restart reForge. TensorRT 11 (or later) is installed on launch.<br />
   Set `TRT_EXTENSION_SKIP_INSTALL=1` to skip this.
3. Load a checkpoint, open the **TensorRT** tab, and click **Build engines**. The first build takes
   a few minutes.
4. Generate. You can now use any checkpoint of the same architecture with these engines.

Engines are used while **Settings → Stable Diffusion → SD Unet** is set to `Automatic` (the
default). Set it to `Speed Hacks` to enable [Speed Hacks](#speed-hacks), or `None` to return to
PyTorch.

> **Note**<br />
> Engines can only be loaded on the GPU, and TensorRT version used to build them. After updating
> TensorRT, build again – your profiles are saved in the tab.

## Differences from Upstream
<p align="center">
	<picture>
		<source media="(prefers-color-scheme: dark)" srcset="docs/engines-dark.svg">
		<img alt="NVIDIA's extension builds a UNet engine for each checkpoint, converts each LoRA, and runs the VAE on PyTorch. This fork builds UNet and VAE engines for each architecture, and loads the weights of whatever checkpoint, LoRAs, and VAE are in use." src="docs/engines-light.svg" width="880">
	</picture>
</p>

|                | NVIDIA's extension             | This fork                          |
| -------------- | ------------------------------ | ---------------------------------- |
| WebUI          | Automatic1111                  | reForge                            |
| Engines        | One per checkpoint, GBs each   | One per profile of an architecture |
| New checkpoint | Build a new engine             | Nothing to do                      |
| LoRAs          | Convert each one first         | Work as usual                      |
| Hires. fix     | Build an engine for both sizes | Covered by the default profiles    |
| VAE            | PyTorch                        | TensorRT                           |

Anything unsupported by the engines (ControlNet, UNet patches like FreeU, sizes outside every
profile) is run on PyTorch instead, with a notice.

## Functionality
- Builds engines per architecture, rather than per checkpoint. Engines hold no weights – those of
  the loaded checkpoint, and LoRAs are loaded into them.
- Offers two engine modes, chosen on first build:
  - *Per-profile engines* (default) – one engine per resolution profile, loaded as needed.
  - *Merged engine* – every profile in one engine, which never needs swapping. Requires 24&nbsp;GB
    of VRAM, and 64&nbsp;GB of RAM.

  The mode can only be changed while no engines exist.
- Builds engines for the VAE's encoder and decoder alongside the UNet. Large decodes (ex. Hires.
  fix) are split into strips to save VRAM, with identical results.
- Checks that a generation fits in VRAM before loading anything, and falls back to PyTorch if it
  doesn't. Figures are printed to the console.
- Keeps swapped-out engines in RAM (with their weights) when there's room to spare, making them
  quick to bring back.

### Speed Hacks
Implements [DeepCache][] and [AdaptiveDiffusion][] – about half of the denoising steps are computed
by a small engine of the UNet's outermost blocks, using cached deep features from a recent full
step. Generations of 12 steps or more are roughly 1.6x faster (estimated), but images differ
slightly from those made without Speed Hacks, even with the same seed. Generations with fewer steps
are unaffected.

The first 30%, and last 5% of steps are always computed in full, as they determine composition, and
fine detail. Both are adjustable in **Settings → Stable Diffusion → TensorRT**.

<br>

<p align="center">
  <img src="docs/claude.svg" width="72" alt="Claude, drawn as an orange flower with a smiling face">
  <br>
  <sub>vibecoded with love by Claude</sub>
</p>
