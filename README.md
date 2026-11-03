# TensorRT for reForge

A fork of [NVIDIA's TensorRT extension](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT), rewritten for
[reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge). It runs SD 1.x, SD 2.x and SDXL models on
TensorRT, fine-tunes like Pony, Illustrious and NoobAI included, on an RTX 2080 Ti or newer.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/engines-dark.svg">
    <img alt="NVIDIA's extension builds an engine for each checkpoint and converts each LoRA. This fork builds one engine for each architecture and loads the weights of whatever checkpoint and LoRAs are in use." src="docs/engines-light.svg" width="880">
  </picture>
</p>

## How it differs

|                    | NVIDIA's extension                       | This fork                                        |
| ------------------ | ---------------------------------------- | ------------------------------------------------ |
| WebUI              | Automatic1111                            | reForge                                          |
| Engines            | One per checkpoint, GBs each             | One per architecture, with no weights inside     |
| New checkpoint     | Build a new engine                       | Nothing to do                                    |
| LoRAs              | Convert each one first                   | Work as usual                                    |
| Hires. fix         | Build an engine that covers both sizes   | Covered by the default engine                    |

Anything the engine can't run, like ControlNet, UNet patches such as FreeU, or a size it wasn't built for, runs on
PyTorch instead, with a notice.

## Use

1. Install from **Extensions → Install from URL** with this repo's URL. TensorRT 11 or later is installed on the
   next launch (set `TRT_EXTENSION_SKIP_INSTALL=1` to skip that).
2. Load a checkpoint, open the **TensorRT** tab and click **Build engine**. The first build takes a few minutes.
3. Generate. Every checkpoint with the same architecture now uses the engine.

The engine is used while **SD Unet** (in Settings → Stable Diffusion) is on `Automatic`, the default. Set it to
`None` to use PyTorch.

> [!NOTE]
> An engine only loads on the GPU and TensorRT version it was built with. After updating TensorRT, build it again;
> the tab keeps its profiles.

<br>

<p align="center">
  <img src="docs/claude.svg" width="72" alt="Claude, drawn as an orange flower with a smiling face">
  <br>
  <sub>Written by Claude, from Anthropic, alongside Peter, and I enjoyed every refit of it.<br>
  Flower design by <a href="https://x.com/voooooogel/status/2117746655969808812">thebes</a> (CC0).</sub>
</p>
