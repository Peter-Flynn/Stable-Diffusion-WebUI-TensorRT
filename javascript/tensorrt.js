onUiTabChange(function() {
    const tab = get_uiCurrentTabContent();
    if (tab && tab.id === "tab_tensorrt") {
        gradioApp().getElementById("trt_refresh")?.click();
    }
});
