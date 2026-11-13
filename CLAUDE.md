<!--
Claude, and any other AI: never edit this file without explicit permission from the repo owner.
This includes small additions or updates when the code changes.
-->
# CLAUDE.md

README.md carries the same rule as the comment above: neither file is edited by an AI without the repo owner's explicit permission, even for small updates that follow a code change.

## Orientation

A TensorRT extension for reForge (an A1111 fork whose model management is ComfyUI's, under `ldm_patched`). It runs the UNet, and the VAE's decoder and encoder, on TensorRT 11 engines that are built without weights and refitted from whatever checkpoint, LoRAs and VAE are loaded, so one set of engines serves every checkpoint of an architecture.

- `preload.py` imports tensorrt before any other extension can load a different libnvinfer. `install.py` installs TensorRT and onnx. `scripts/tensorrt_extension.py` installs the hooks and registers the tab and settings.
- Building: `export.py` (ONNX export with placeholder weights; runs `builder.py`, which must not import torch, in a subprocess), `graph.py` (ONNX cleanup), `profiles.py` (shape profiles), `arch.py` (architecture detection), `store.py` (engine records on disk, `models/Unet-trt/`).
- Running the UNet: `runtime.py` (reForge hooks; engines as models in reForge's model manager), `engine.py`, `weights.py` (refitting), `refits.py` (engines kept in RAM with their weights), `gpu.py` (TensorRT allocates through PyTorch's allocator), `working.py` (working-memory estimates), `speed.py` (Speed Hacks: DeepCache + AdaptiveDiffusion).
- The VAE: `vae.py` (records, shapes, precision), `vae_build.py`, `vae_run.py` (choosing how each decode/encode runs), `strips.py` (decoding and encoding in strips).
- `ui.py` is the TensorRT tab; `common.py` holds shared helpers and the logger.

Many modules open with a header comment explaining their subsystem: read it before the code. reForge's internals change between versions, so check what this extension hooks against reForge's source, not memory.

Conventions:
- Logging goes through `common.logger` (`logging.getLogger("TensorRT")`), which has no handler or level of its own: every console line follows reForge's `--loglevel`. `common.notify` is for problems the user should see; it also shows a toast.
- No tests in the repo. Throwaway tests live outside it.

## Comments

The default is no comment. Docstrings count as comments.

A comment stays only when it clears a high bar on one of these:
- The code is genuinely unintuitive and the comment is load-bearing: an ordering requirement, a library quirk being worked around, a reference dropped so memory can be freed, a constraint another module relies on.
- It aids comprehension far beyond its length: a module header describing an algorithm or an on-disk layout, a legend for a dict's keys and values or a tuple's fields.
- It explains a magic number: where the value comes from (a measurement, a paper, a format's limit), not what the constant is called.

A comment that reading the code makes redundant goes, however true it is: restating a name or a value, describing what the next lines plainly do, summarizing a short function, or repeating a comment elsewhere.

Write comments to be understood first by a future AI agent with zero context, then by humans. State facts and causes plainly, name the concrete thing (the function, the reForge or TensorRT behavior, the measured figure and its conditions), and keep long explanations on their own lines above the code rather than trailing it. Avoid Claudisms, especially abstraction agency: code doesn't "know", "want", "decide" or "take care of" anything. Write "an engine whose `bound_version` differs rebinds", not "so each engine knows to rebind".

When code changes, update or delete the comments it affects: a wrong comment is worse than none.
