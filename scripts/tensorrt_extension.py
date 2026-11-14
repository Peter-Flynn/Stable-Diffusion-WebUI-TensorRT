from modules import script_callbacks, scripts

from trt_unet import runtime, ui
from trt_unet.common import logger


class TensorRT(scripts.Script):
    def title(self):
        return "TensorRT"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        runtime.new_generation(p)

    def before_process_batch(self, p, *args, **kwargs):
        runtime.new_batch()


try:
    runtime.install_hooks()
except Exception as e:
    logger.error(f"This extension needs reForge: {e}")
script_callbacks.on_list_unets(runtime.list_unets)
script_callbacks.on_ui_tabs(ui.on_ui_tabs)
script_callbacks.on_ui_settings(ui.on_ui_settings)
script_callbacks.on_before_ui(ui.add_xyz_axes)
