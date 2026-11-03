from modules import script_callbacks, scripts

from trt_unet import runtime, ui


class TensorRT(scripts.Script):
    def title(self):
        return "TensorRT"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        runtime.new_generation()


try:
    runtime.install_hooks()
except Exception as e:
    print(f"[TensorRT] This extension needs reForge: {e}", flush=True)
script_callbacks.on_list_unets(runtime.list_unets)
script_callbacks.on_before_ui(ui.on_before_ui)
script_callbacks.on_after_component(ui.on_after_component)
script_callbacks.on_ui_tabs(ui.on_ui_tabs)
