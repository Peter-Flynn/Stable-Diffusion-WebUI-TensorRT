# TensorRT for reForge

A fork of [NVIDIA's TensorRT extension](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT), rewritten for
[reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge). It runs SD 1.x, SD 2.x and SDXL models on
TensorRT, fine-tunes like Pony, Illustrious and NoobAI included, on an RTX 2080 Ti or newer.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/engines-dark.svg">
    <img alt="NVIDIA's extension builds an engine for each checkpoint and converts each LoRA. This fork builds engines for each architecture, one per profile, and loads the weights of whatever checkpoint and LoRAs are in use." src="docs/engines-light.svg" width="880">
  </picture>
</p>

## How it differs

|                | NVIDIA's extension                     | This fork                                             |
| -------------- | -------------------------------------- | ----------------------------------------------------- |
| WebUI          | Automatic1111                          | reForge                                               |
| Engines        | One per checkpoint, GBs each           | One per profile of an architecture (or one for all of them), no weights inside |
| New checkpoint | Build a new engine                     | Nothing to do                                         |
| LoRAs          | Convert each one first                 | Work as usual                                         |
| Hires. fix     | Build an engine that covers both sizes | Covered by the default profiles                       |

Anything the engines can't run, like ControlNet, UNet patches such as FreeU, or a size no profile covers, runs on
PyTorch instead, with a notice.

## Use

1. Install from **Extensions → Install from URL** with this repo's URL. TensorRT 11 or later is installed on the
   next launch (set `TRT_EXTENSION_SKIP_INSTALL=1` to skip that).
2. Load a checkpoint and open the **TensorRT** tab. The first time, it asks for the engine mode (below). Click
   **Build engines**; the first build takes a few minutes.
3. Generate. Every checkpoint with the same architecture now uses the engines. One is loaded at a time: each
   batch runs on the one that fits it best, and its Hires. fix stays on it if it covers the larger size.

**Engine modes.** *Per-profile engines*, the default, are an engine per profile, built one at a time; only the one
a generation's sizes need is loaded. A *merged engine* holds every profile of an architecture, so no pass ever
loads another engine, but it takes every profile's share of TensorRT's memory at once, and its build takes them all
at once: it needs at least 24 GB of VRAM and 64 GB of RAM. The mode can only change while there are no engines.

The engines are used while **SD Unet** (in Settings → Stable Diffusion) is on `Automatic`, the default. Set it to
`None` to use PyTorch.

> [!NOTE]
> An engine only loads on the GPU and TensorRT version it was built with. After updating TensorRT, build again;
> the tab keeps the profiles.

<br>

<p align="center">
  <img src="docs/claude.svg" width="72" alt="Claude, drawn as an orange flower with a smiling face">
  <br>
  <sub>vibecoded with love by Claude</sub>
</p>
