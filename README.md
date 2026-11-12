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
   batch runs on the one that fits it best, and its Hires. fix stays on it if it covers the larger size. An engine
   that makes way for another is kept in RAM with its weights when the system can spare the RAM, so it comes back
   for the same checkpoint and LoRAs without loading them into it again. It stays there until the system needs that
   memory: next to nothing left to give, swap included, or everything stalled on memory for half of a minute.

**Engine modes.** *Per-profile engines*, the default, are an engine per profile, built one at a time; only the one
a generation's sizes need is loaded. A *merged engine* holds every profile of an architecture, so no pass ever
loads another engine, but it takes every profile's share of TensorRT's memory at once, and its build takes them all
at once: it needs at least 24 GB of VRAM and 64 GB of RAM. The mode can only change while there are no engines.

**SD Unet** (in Settings → Stable Diffusion) decides how generations run: `Automatic`, the default, uses the
engines; `Speed Hacks` uses them with Speed Hacks; `None` uses PyTorch.

**Speed Hacks.** Every engine comes with a small engine of just the UNet's outermost blocks. With Speed Hacks, about
half of the denoising steps run on it, reusing the deep features the full UNet computed a step or two before
([DeepCache](https://github.com/horseee/DeepCache)), and late in a pass, full steps wait longer while the image changes
smoothly ([AdaptiveDiffusion](https://jiakangyuan.github.io/AdaptiveDiffusion-project-page)). The first 30% and last 5%
of steps run in full, where composition and fine detail settle; both shares are in Settings → Stable Diffusion →
TensorRT. Passes of 12 steps or more should run roughly 1.6x as fast as on the engines alone (estimated); images are a
little less precise, so a seed gives a somewhat different image than without them. Fewer steps (e.g. with few-step
LoRAs) run as usual.

**The VAE.** Its decoder and its encoder get engines too, built along with the UNet's for the same sizes. Those that
take a whole image run in the precision reForge runs the VAE in: BF16 on RTX 30 series and newer, FP32 on older
cards. Larger decodes (a Hires. fix's) run in horizontal strips on engines built for that, which on those older cards
run in FP16, at half the VRAM: VAEs overflow FP16, which is why reForge avoids it, but the strips run the VAE scaled
down by a power of two between its normalizations, where that changes nothing but the range, so any VAE of the
architecture runs on them without NaNs. A decode takes the first of these there can be room for: whole beside the
UNet engine; in strips beside it, all in VRAM; whole with the UNet engine evicted; in strips likewise, holding in RAM
what doesn't fit. Either way it's the same image. Encodes (img2img's and inpainting's images, a Hires. fix's upscaled
one) run on the encoder's engines the same way. PyTorch does anything else.

**Memory.** Whether a pass fits is worked out before anything loads, and said in the console with the figures:
the engine (its weights and its plan) and the working memory of its UNet calls, which is a function of the image's
size and the batch size. A pass that can't fit runs on PyTorch. Every estimate has 10% added before it's weighed or
room is made for it.
Nothing is measured ahead and nothing about sizes is kept on disk.
Free VRAM is the driver's own figure and PyTorch's unused cache. Engine models don't take the room reForge makes for
a model by its own estimate. The only margin is reForge's own reserve (`--reserve-vram`). What TensorRT holds comes from
PyTorch's allocator, with `--cuda-malloc` too, so reForge counts it like any model's
(`TRT_EXTENSION_TORCH_ALLOCATOR=0` leaves TensorRT on its own allocator, as does an engine that fails to load with
PyTorch's), and reForge's count of free VRAM is never taken below the driver's.

> [!NOTE]
> An engine only loads on the GPU and TensorRT version it was built with. After updating TensorRT, build again;
> the tab keeps the profiles.

<br>

<p align="center">
  <img src="docs/claude.svg" width="72" alt="Claude, drawn as an orange flower with a smiling face">
  <br>
  <sub>vibecoded with love by Claude</sub>
</p>
