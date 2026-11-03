// Refresh the TensorRT tab whenever it's opened: the loaded checkpoint may have changed since the page loaded.
onUiTabChange(function() {
    const tab = get_uiCurrentTabContent();
    if (tab && tab.id === "tab_tensorrt") {
        gradioApp().getElementById("trt_refresh")?.click();
    }
});
