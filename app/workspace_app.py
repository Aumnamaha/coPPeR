#!/usr/bin/env python3
r"""Native Windows WebView2 workspace host for coPPeR.

Launches a native Win32 window using pywebview (Edge WebView2 runtime) with:
- WorkspaceBridge JS API: capture_selection(), apply_to_active_window(), load_file()
- Global Ctrl+Alt+Space hotkey via ctypes RegisterHotKey
- Falls back to HTTP mode if pywebview is unavailable

Usage::
    python app/workspace_app.py              # native WebView2 window
    COFFEE_MODE=http python app/workspace_app.py  # browser/HTTP fallback
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Check pywebview availability
# ---------------------------------------------------------------------------

try:
    import webview  # type: ignore[import-untyped]

    _HAS_WEBVIEW = True
except ImportError:
    _HAS_WEBVIEW = False


# ---------------------------------------------------------------------------
# Bridge class — exposed to JS via window.WorkspaceBridge
# ---------------------------------------------------------------------------


class WorkspaceBridge:
    """Python-side bridge methods callable from the WebView2 JavaScript layer."""

    def __init__(self) -> None:
        self._on_capture_callback: Any = None
        self._last_selection: str = ""

    # ---- Register a Python callback for when JS calls capture_selection() ----
    def set_capture_callback(self, callback: Any) -> None:
        """Set the function to call when the user triggers Ctrl+Alt+Space."""
        self._on_capture_callback = callback

    # ---- Methods exposed to JavaScript ----

    def capture_selection(self) -> dict[str, Any]:
        """Capture highlighted selection from active window (called by JS)."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from editor_bridge import capture_highlighted_selection

            result = capture_highlighted_selection()
            self._last_selection = result.get("wrapped_document", "")
            return {
                "selection": self._last_selection,
                "charCount": len(self._last_selection),
                "filename": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def apply_to_active_window(self, text: str) -> dict[str, Any]:
        """Apply edited text to the active editor window."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from editor_bridge import apply_selection_replacement

            ok = apply_selection_replacement(text)
            return {"success": bool(ok)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def load_file(self, file_path: str = "") -> dict[str, Any]:
        """Load a document file into the workspace."""
        try:
            if not file_path or not Path(file_path).exists():
                # Open native file picker via webview (handled by pywebview API)
                return {"filename": None, "content": None}

            content = Path(file_path).read_text(encoding="utf-8")
            return {
                "filename": str(Path(file_path).name),
                "content": content,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def save_file(self, path: str = "", content: str = "") -> dict[str, Any]:
        """Save document content to a file."""
        try:
            out_path = Path(path) if path else None
            if not out_path or not content:
                return {"success": False}
            out_path.write_text(content, encoding="utf-8")
            return {"success": True, "path": str(out_path)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def list_artifacts(self) -> dict[str, Any]:
        """List all saved workspace artifacts."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from artifact_store import ArtifactStore

            base_dir = os.environ.get(
                "COPPER_WORKSPACE_DIR",
                str(Path(__file__).resolve().parent.parent / "workspace_notes"),
            )
            store = ArtifactStore(base_dir)
            return {"success": True, "artifacts": store.list_notes()}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def save_artifact(
        self,
        title: str = "",
        content: str = "",
        artifact_type: str = "note",
        slug_override: str | None = None,
    ) -> dict[str, Any]:
        """Save a workspace artifact to local disk."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from artifact_store import ArtifactStore

            base_dir = os.environ.get(
                "COPPER_WORKSPACE_DIR",
                str(Path(__file__).resolve().parent.parent / "workspace_notes"),
            )
            store = ArtifactStore(base_dir)
            result = store.save_note(
                title,
                content=content,
                artifact_type=artifact_type,
                slug_override=slug_override,
            )
            return {"success": True, **result}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def delete_artifact(self, slug: str) -> dict[str, Any]:
        """Delete a workspace artifact."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from artifact_store import ArtifactStore

            base_dir = os.environ.get(
                "COPPER_WORKSPACE_DIR",
                str(Path(__file__).resolve().parent.parent / "workspace_notes"),
            )
            store = ArtifactStore(base_dir)
            deleted = store.delete_note(slug)
            return {"success": bool(deleted), "deleted": slug}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def export_docx(
        self, title: str = "", content: str = "", output_path: str = ""
    ) -> dict[str, Any]:
        """Export content to a .docx file."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from artifact_store import ArtifactStore

            result = ArtifactStore.export_to_docx(title, content, output_path)
            if "error" in result:
                return {"error": result["error"]}
            return {"success": True, **result}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Global Hotkey (Ctrl+Alt+Space) via Win32 RegisterHotKey
# ---------------------------------------------------------------------------

_HOTKEY_ID = 9001  # unique ID for our hotkey


def _register_hotkey() -> None:
    """Register Ctrl+Alt+Space as a global hotkey on Windows."""
    if sys.platform != "win32":
        return

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        MOD_CTRL = 0x0002
        HOTKEYF_ALT = 0x0004
        VK_SPACE = 0x20

        # Register Ctrl+Alt+Space (MOD_CONTROL | MOD_ALT, VK_SPACE)
        result = user32.RegisterHotKey(
            None, _HOTKEY_ID, MOD_CTRL | HOTKEYF_ALT, VK_SPACE
        )
        if result:
            print(f"Global hotkey registered: Ctrl+Alt+Space (ID={_HOTKEY_ID})")

            # Start a message loop to handle the hotkey in a background thread
            def _hotkey_loop() -> None:
                msg = wintypes.MSG()
                while user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
                    if msg.message == 0x0312 and msg.wParam == _HOTKEY_ID:  # WM_HOTKEY
                        print("Hotkey triggered — capturing selection…")
                        try:
                            bridge = getattr(_workspace_app, "_bridge", None)
                            if bridge and hasattr(bridge, "capture_selection"):
                                result = bridge.capture_selection()
                                _evaluate_js(f"""
                                        window.WorkspaceBridge._handleCapture({json.dumps(result)});
                                    """)
                        except Exception:  # noqa: S110, BLE001
                            pass
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageA(ctypes.byref(msg))

            t = threading.Thread(target=_hotkey_loop, daemon=True)
            t.start()
        else:
            print(
                "Warning: Could not register global hotkey (another app may be using it)."
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Hotkey registration failed: {exc}")


# ---------------------------------------------------------------------------
# JS bridge initialization script
# ---------------------------------------------------------------------------

_BRIDGE_JS = """
<script>
window.WorkspaceBridge = window.WorkspaceBridge || {};
var _bridgeMethods = ['captureSelection','applyToActiveWindow','loadFile','saveFile'];
_bridgeMethods.forEach(function(method) {
    if (typeof window.bridge[method] === 'function') {
        window.WorkspaceBridge[method] = function() {
            var args = Array.prototype.slice.call(arguments);
            return window.bridge[method].apply(window.bridge, args);
        };
    }
});

// Handle post-capture UI update from hotkey thread
window.WorkspaceBridge._handleCapture = function(data) {
    if (data && data.selection) {
        var label = data.selection.substring(0,40) + (data.selection.length>40?'…':'');
        document.dispatchEvent(new CustomEvent('copper:capture', {detail:{label: 'Active Selection · '+data.charCount+' chars', data:data}}));
    }
};

// Listen for file picker result
document.addEventListener('copper:file-loaded', function(e) {
    var chip = document.getElementById('context-chip-area');
    if (chip && e.detail.label) {
        chip.innerHTML = '<div class=\"context-chip\" title=\"'+e.detail.label+'\">' +
            '<span>'+e.detail.label+'</span>' +
            '<span class=\"chip-remove\" onclick=\"this.parentElement.classList.add(\\'hidden\\');this.parentElement.innerHTML=\\'\\';\">&times;</span></div>';
        chip.classList.remove('hidden');
    }
});

console.log('[coPPeR] WorkspaceBridge initialized.');
</script>
"""


# ---------------------------------------------------------------------------
# Global reference for hotkey thread access
# ---------------------------------------------------------------------------

_workspace_app: Any = None  # type: ignore[assignment]


def _evaluate_js(js_code: str) -> None:
    """Evaluate JavaScript in the main webview window."""
    if _workspace_app and hasattr(_workspace_app, "evaluate_js"):
        try:
            _workspace_app.evaluate_js(js_code)
        except Exception:  # noqa: S110, BLE001
            pass


# ---------------------------------------------------------------------------
# Embedded backend auto-start (for standalone coPPeR.exe)
# ---------------------------------------------------------------------------


def _ensure_backend_running(
    host: str = "127.0.0.1", port: int = 8088, timeout: float = 5.0
) -> bool:
    """Check if the inference server is already running.

    If not, start it in a background daemon thread and wait until /health responds.
    Returns True whether the backend was ready (already or newly started).
    """
    import time as _time
    from urllib.request import urlopen

    url = f"http://{host}:{port}/health"
    # Check if already running
    try:
        with urlopen(url, timeout=2) as resp:
            if resp.status == 200:
                print(f"[coPPeR] Backend already running at {url}")
                return True
    except Exception:  # noqa: S110, BLE001
        pass

    # Start uvicorn in a background daemon thread
    import subprocess as _sub

    root_dir = str(Path(__file__).resolve().parent.parent)
    print(f"[coPPeR] Starting embedded backend on {host}:{port}...")
    proc = _sub.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.inference_server:app",
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=root_dir,
    )

    # Wait for /health to respond
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    print(f"[coPPeR] Backend ready at {url}")
                    return True
        except Exception:  # noqa: S110, BLE001
            pass
        _time.sleep(0.25)

    proc.terminate()
    print(
        f"[coPPeR] WARN: Backend did not become ready within {timeout}s — "
        "workspace will fall back to HTTP mode if server is started separately."
    )
    return False


# ---------------------------------------------------------------------------
# Launch functions
# ---------------------------------------------------------------------------


def launch_native() -> None:
    """Launch the native WebView2 window."""
    global _workspace_app

    bridge = WorkspaceBridge()
    _workspace_app = type("_App", (), {"_bridge": bridge, "evaluate_js": _evaluate_js})  # type: ignore[assignment]

    ui_dir = str(Path(__file__).resolve().parent / "workspace_ui")
    index_path = os.path.join(ui_dir, "index.html")

    if not os.path.exists(index_path):
        print(f"ERROR: UI files not found at {index_path}")
        sys.exit(1)

    # Register global hotkey before creating window
    _register_hotkey()

    window = webview.create_window(
        title="coPPeR Workspace",
        url=f"file://{index_path}",
        js_api=bridge,
        width=1200,
        height=750,
        text_select=True,
        background_color="#F4F7FB",
    )

    # Inject bridge JS after the WebView2 document is loaded.
    # pywebview 6.x exposes window.events.loaded (not dom_ready).
    def _inject_bridge(window_obj):
        try:
            window_obj.evaluate_js(_BRIDGE_JS)
        except Exception:  # noqa: S110, BLE001
            pass

    # pywebview 6.x Event objects use += for subscription (not .append).
    _loaded_event = getattr(window.events, "loaded", None)
    if _loaded_event is not None:
        _loaded_event += _inject_bridge

    webview.start()


def launch_http_fallback(host: str = "127.0.0.1", port: int = 8088) -> None:
    """Fallback: open workspace in default browser via HTTP."""
    import sys as _sys

    url = f"http://{host}:{port}/workspace"
    print(f"[coPPeR] Opening workspace in browser: {url}")

    try:
        if _sys.platform == "win32":
            os.startfile(url)  # type: ignore[attr-defined]
        else:
            import webbrowser

            webbrowser.open(url)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not open browser: {exc}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the workspace — native WebView2 or HTTP fallback."""
    # Parse CLI flags
    no_embedded = "--no-embedded-server" in sys.argv
    mode = os.environ.get("COPPER_UI_MODE", "native").lower()

    # Auto-start embedded backend unless explicitly disabled
    if not no_embedded:
        _ensure_backend_running()

    if _HAS_WEBVIEW and mode == "native":
        print("[coPPeR] Launching native WebView2 window…")
        launch_native()
    else:
        if not _HAS_WEBVIEW:
            print("[coPPeR] pywebview not installed — using HTTP fallback.")
        print(
            f"[coPPeR] Mode={mode} — opening workspace in browser at http://127.0.0.1:8088/workspace"
        )
        launch_http_fallback()


if __name__ == "__main__":
    main()
