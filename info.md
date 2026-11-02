# TensorRT Extension

Runs the Stable Diffusion UNet with TensorRT engines for faster generation on NVIDIA GPUs.

## How to use

1. Pick a preset and click **Export**. The first build takes a few minutes; later builds are faster thanks to the timing cache.
2. Add `sd_unet` to **Settings → User Interface → Quicksettings list** and reload the UI.
3. Select the `[TRT]` engine (or `Automatic`) in the `sd_unet` dropdown. Choose `None` to use PyTorch.

### LoRA

Convert a LoRA in the **TensorRT LoRA** tab, then use it in your prompt as usual (`<lora:name:0.8>`).

## Notes

- **Static** engines support one resolution and batch size and are the fastest. **Dynamic** engines cover a range; wider ranges use more VRAM.
- Engines only load with the TensorRT version and GPU they were built with. After upgrading TensorRT, rebuild them (Advanced Settings → Force Rebuild) and delete old ones under **Manage engines**.
- ControlNet and other UNet patches are not applied by TensorRT engines; set `sd_unet` to `None` when you need them.
