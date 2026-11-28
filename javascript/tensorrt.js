(function () {
    const BANNER_ID = "trt_quant_banner";
    const TOOLTIP_ID = "trt_quant_tooltip";
    const THROTTLE_MS = 1500;

    let lastFetchAt = 0;
    let inFlight = false;
    let pendingTimer = null;
    let pendingRefresh = false;
    let pendingForce = false;
    let lastStateKey = "";

    function getBannerRoot() {
        if (typeof gradioApp !== "function") {
            return document;
        }
        return gradioApp() || document;
    }

    function ensureBanner() {
        const root = getBannerRoot();
        let banner = root.getElementById ? root.getElementById(BANNER_ID) : root.querySelector("#" + BANNER_ID);
        if (banner) {
            return banner;
        }
        const tabs = root.getElementById ? root.getElementById("tabs") : root.querySelector("#tabs");
        if (!tabs || !tabs.parentNode) {
            return null;
        }
        banner = document.createElement("div");
        banner.id = BANNER_ID;
        banner.setAttribute("role", "status");
        banner.setAttribute("aria-live", "polite");

        const pill = document.createElement("div");
        pill.className = "trt-quant-pill";
        pill.setAttribute("tabindex", "0");
        pill.setAttribute("aria-describedby", TOOLTIP_ID);

        const icon = document.createElement("span");
        icon.className = "trt-quant-icon";
        icon.setAttribute("aria-hidden", "true");
        icon.textContent = "\u26A0";

        const title = document.createElement("span");
        title.className = "trt-quant-title";

        const hint = document.createElement("span");
        hint.className = "trt-quant-hint";
        hint.setAttribute("aria-hidden", "true");
        hint.textContent = "\u24D8";

        const tooltip = document.createElement("div");
        tooltip.id = TOOLTIP_ID;
        tooltip.className = "trt-quant-tooltip";
        tooltip.setAttribute("role", "tooltip");

        pill.appendChild(icon);
        pill.appendChild(title);
        pill.appendChild(hint);
        pill.appendChild(tooltip);
        banner.appendChild(pill);
        tabs.parentNode.insertBefore(banner, tabs);
        return banner;
    }

    function applyBannerState(data) {
        const visible = Boolean(data && data.visible);
        const title = visible && typeof data.title === "string" ? data.title : "";
        const tooltip = visible && typeof data.tooltip === "string" ? data.tooltip : "";
        const key = JSON.stringify([visible, title, tooltip]);
        if (key === lastStateKey) {
            return;
        }
        const banner = ensureBanner();
        if (!banner) {
            return;
        }
        lastStateKey = key;
        const titleEl = banner.querySelector(".trt-quant-title");
        const tipEl = banner.querySelector(".trt-quant-tooltip");
        if (titleEl) {
            titleEl.textContent = title;
        }
        if (tipEl) {
            tipEl.textContent = tooltip;
        }
        banner.classList.toggle("trt-quant-visible", visible);
    }

    function doFetch() {
        inFlight = true;
        lastFetchAt = Date.now();
        fetch("./trt/quant-warning", { cache: "no-store" })
            .then(function (resp) {
                if (!resp.ok) {
                    return null;
                }
                return resp.json();
            })
            .then(function (data) {
                if (data && typeof data === "object") {
                    applyBannerState(data);
                }
            })
            .catch(function () {
                // Ignore transient network errors during WebUI reload/shutdown.
            })
            .finally(function () {
                inFlight = false;
                if (pendingRefresh) {
                    const forceNext = pendingForce;
                    pendingRefresh = false;
                    pendingForce = false;
                    scheduleRefresh(forceNext);
                }
            });
    }

    function scheduleRefresh(force) {
        if (inFlight) {
            pendingRefresh = true;
            if (force) {
                pendingForce = true;
            }
            return;
        }
        const elapsed = Date.now() - lastFetchAt;
        if (force || elapsed >= THROTTLE_MS) {
            if (pendingTimer !== null) {
                clearTimeout(pendingTimer);
                pendingTimer = null;
            }
            doFetch();
            return;
        }
        if (pendingTimer === null) {
            pendingTimer = setTimeout(function () {
                pendingTimer = null;
                doFetch();
            }, THROTTLE_MS - elapsed);
        }
    }

    window.trtRefreshQuantBanner = function () {
        scheduleRefresh(true);
    };

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () {
            ensureBanner();
            scheduleRefresh(true);
        });
    }

    if (typeof onUiTabChange === "function") {
        onUiTabChange(function () {
            const tab = typeof get_uiCurrentTabContent === "function" ? get_uiCurrentTabContent() : null;
            if (tab && tab.id === "tab_tensorrt") {
                gradioApp().getElementById("trt_refresh")?.click();
            }
            scheduleRefresh(true);
        });
    }

    if (typeof onOptionsChanged === "function") {
        onOptionsChanged(function () {
            scheduleRefresh(true);
        });
    }

    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(function () {
            scheduleRefresh(false);
        });
    }

    document.addEventListener("visibilitychange", function () {
        if (!document.hidden) {
            scheduleRefresh(true);
        }
    });
})();
