#!/usr/bin/env python3
r"""Windows screen-aware cursor overlay for coPPeR.

A lightweight tkinter floating command bar triggered via global hotkey
Ctrl+Alt+Space (RegisterHotKey).  Shows detected editor badge, captures
cursor-highlighted text, calls the local inference server, previews the
edit, and applies it in-place.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import tkinter as tk
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Editor bridge imports
# ---------------------------------------------------------------------------

_sys_path = str(Path(__file__).resolve().parent.parent / "src")
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
_sys_app = str(Path(__file__).resolve().parent)
if _sys_app not in sys.path:
    sys.path.insert(0, _sys_app)

from editor_bridge import (
    apply_selection_replacement,
    capture_highlighted_selection,
    detect_installed_editors,
    get_active_window_context,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_URL = "http://127.0.0.1:8088"
HOTKEY_ID = 1
MOD_CONTROL = 0x0002
MOD_ALT = 0x0001
VK_SPACE = 0x20

# Window geometry (relative to screen bottom-right)
WIN_WIDTH = 480
WIN_HEIGHT = 360
MARGIN = 10


# ---------------------------------------------------------------------------
# RegisterHotKey helper
# ---------------------------------------------------------------------------


def _register_hotkey() -> bool:
    """Register Ctrl+Alt+Space as global hotkey.

    Returns True if registration succeeded.
    """
    try:
        user32 = ctypes.windll.user32
        result = user32.RegisterHotKey(None, HOTKEY_ID, MOD_CONTROL | MOD_ALT, VK_SPACE)
        return bool(result)
    except (OSError, AttributeError):
        return False


def _unregister_hotkey() -> None:
    """Unregister the global hotkey."""
    try:
        ctypes.windll.user32.UnregisterHotKey(None, HOTKEY_ID)
    except (OSError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Overlay window
# ---------------------------------------------------------------------------


class CopperOverlay:
    """Floating tkinter overlay for screen-aware document editing."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("coPPeR -- Screen Edit")
        self.root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
        self.root.overrideredirect(True)  # frameless
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1a1a2e")

        # Position at bottom-right of primary screen
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = sw - WIN_WIDTH - MARGIN
        y = sh - WIN_HEIGHT - MARGIN
        self.root.geometry(f"+{x}+{y}")

        # State variables
        self._selected_text: str = ""
        self._result_text_widget: tk.Text | None = None
        self._editor_info: dict[str, Any] = {}

        self._build_ui()
        self._position_window()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the overlay UI layout."""
        style = {
            "bg": "#1a1a2e",
            "fg": "#e0e0e0",
            "font": ("Segoe UI", 9),
            "accent": "#4ecca3",
            "dark_bg": "#16213e",
        }

        # --- Editor badge (top bar) ---
        badge_frame = tk.Frame(self.root, bg=style["dark_bg"], height=36)
        badge_frame.pack(fill="x", padx=0, pady=0)
        badge_frame.pack_propagate(False)

        self._badge_label = tk.Label(
            badge_frame,
            text="No editor detected",
            bg=style["dark_bg"],
            fg=style["accent"],
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        self._badge_label.pack(side="left", padx=MARGIN, pady=3)

        # --- Selection preview (scrollable text area) ---
        sel_frame = tk.LabelFrame(
            self.root,
            text="Cursor Highlighted Text:",
            bg=style["bg"],
            fg=style["fg"],
            font=("Segoe UI", 9),
            bd=0,
        )
        sel_frame.pack(fill="x", padx=MARGIN, pady=(6, 3))

        self._sel_text = tk.Text(
            sel_frame,
            height=4,
            bg="#0f3460",
            fg="#c0c0c0",
            font=("Consolas", 9),
            wrap="word",
            state="disabled",
        )
        self._sel_text.pack(fill="x", padx=MARGIN, pady=2)

        # --- Instruction input ---
        instr_frame = tk.Frame(self.root, bg=style["bg"])
        instr_frame.pack(fill="x", padx=MARGIN, pady=(3, 3))

        tk.Label(
            instr_frame,
            text='Instruction: "Describe the fix right here..."',
            bg=style["bg"],
            fg="#808080",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        self._instr_entry = tk.Entry(
            instr_frame,
            bg="#0f3460",
            fg="#e0e0e0",
            font=("Segoe UI", 10),
            bd=1,
            relief="solid",
        )
        self._instr_entry.pack(fill="x", pady=(2, 4))

        # --- Buttons row ---
        btn_frame = tk.Frame(self.root, bg=style["bg"])
        btn_frame.pack(fill="x", padx=MARGIN, pady=(3, 6))

        capture_btn = tk.Button(
            btn_frame,
            text="Capture Selection",
            bg="#0f3460",
            fg=style["fg"],
            font=("Segoe UI", 9),
            bd=1,
            relief="solid",
            command=self._on_capture,
        )
        capture_btn.pack(side="left", padx=(0, 4))

        generate_btn = tk.Button(
            btn_frame,
            text="Generate Fix",
            bg=style["accent"],
            fg="#1a1a2e",
            font=("Segoe UI", 9, "bold"),
            bd=1,
            relief="solid",
            command=self._on_generate,
        )
        generate_btn.pack(side="left")

        # --- Result preview ---
        result_frame = tk.LabelFrame(
            self.root,
            text="Model Output:",
            bg=style["bg"],
            fg=style["fg"],
            font=("Segoe UI", 9),
            bd=0,
        )
        result_frame.pack(fill="both", expand=True, padx=MARGIN, pady=(3, 6))

        self._result_text = tk.Text(
            result_frame,
            height=4,
            bg="#0f3460",
            fg="#c0c0c0",
            font=("Consolas", 9),
            wrap="word",
            state="disabled",
        )
        self._result_text.pack(fill="both", expand=True, padx=MARGIN, pady=2)

        apply_btn = tk.Button(
            result_frame,
            text="Apply In-Place",
            bg="#e94560",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            bd=1,
            relief="solid",
            command=self._on_apply,
        )
        apply_btn.pack(side="right", padx=MARGIN, pady=(2, 4))

    # ------------------------------------------------------------------
    # Window positioning (draggable)
    # ------------------------------------------------------------------

    def _position_window(self) -> None:
        """Center window initially."""
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - WIN_WIDTH) // 2
        y = (sh - WIN_HEIGHT) // 2
        self.root.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # Hotkey handler
    # ------------------------------------------------------------------

    def _on_hotkey(self) -> None:
        """Callback for Ctrl+Alt+Space global hotkey."""
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception:  # noqa: BLE001, S110
            pass

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_capture(self) -> None:
        """Capture the cursor-highlighted selection."""
        capture_result = capture_highlighted_selection()
        selected = capture_result.get("selected_text", "")
        self._selected_text = selected
        self._sel_text.config(state="normal")
        self._sel_text.delete(1.0, tk.END)
        preview = selected if selected else "(no text selected -- highlight text first)"
        self._sel_text.insert(tk.END, preview)
        self._sel_text.config(state="disabled")

    def _on_generate(self) -> None:
        """Call the inference server to generate a fix."""
        instruction = self._instr_entry.get().strip()
        if not instruction and not self._selected_text:
            return  # Nothing to do

        payload = {
            "highlighted_text": self._selected_text,
            "surrounding_context": "",
            "instruction": instruction or "Fix the highlighted text",
            "active_app": self._editor_info.get("editor_type", "generic_editor"),
            "mode": "cursor_span_edit",
        }

        try:
            import requests

            resp = requests.post(
                f"{SERVER_URL}/v1/edit-span",
                json=payload,
                timeout=30,
            )
            if resp.status_code != 200:
                detail = ""
                try:
                    err_data = resp.json()
                    detail = err_data.get("detail", str(resp.text))
                except Exception:  # noqa: BLE001
                    detail = resp.text or f"HTTP {resp.status_code}"
                self._result_text.config(state="normal")
                self._result_text.delete(1.0, tk.END)
                self._result_text.insert(
                    tk.END,
                    f"[SERVER ERROR] HTTP {resp.status_code}: {detail}",
                )
                self._result_text.config(state="disabled")
                return

            data = resp.json()
            self._result_text.config(state="normal")
            self._result_text.delete(1.0, tk.END)
            result_str = data.get("result", "")
            valid = data.get("valid", False)
            reason = data.get("reason", "")

            if not result_str:
                # Server returned empty result — show validation detail
                display = (
                    f"[EMPTY RESULT] {reason}"
                    if reason
                    else "[EMPTY RESULT] No output from model"
                )
            else:
                status_mark = "[OK]" if valid else "[WARN]"
                display = f"{status_mark} {result_str}"
                if not valid and reason:
                    display += f"\n  Reason: {reason}"

            self._result_text.insert(tk.END, display)
            self._result_text.config(state="disabled")

        except Exception as exc:  # noqa: BLE001
            self._result_text.config(state="normal")
            self._result_text.delete(1.0, tk.END)
            # Classify the error for actionable messaging
            err_type = type(exc).__name__
            if "Connection" in err_type or "refused" in str(exc).lower():
                msg = (
                    f"[FAIL] Connection refused on {SERVER_URL} — "
                    f"is inference_server.py running?\nError: {exc}"
                )
            elif "Timeout" in err_type or "timed out" in str(exc).lower():
                msg = f"[FAIL] Server timed out after 30s — model may be loading\nError: {exc}"
            else:
                msg = f"[FAIL] {err_type}: {exc}"
            self._result_text.insert(tk.END, msg)
            self._result_text.config(state="disabled")

    def _on_apply(self) -> None:
        """Apply the generated fix in-place."""
        try:
            result = self._result_text.get("1.0", tk.END).strip()
            # Strip status prefix
            if result.startswith(("[OK] ", "[FAIL] ")):
                result = result[6:]
        except Exception:  # noqa: BLE001
            result = ""

        if not result or result.startswith(
            ("Server error:", "[FAIL]", "[SERVER ERROR]", "[EMPTY RESULT]")
        ):
            return  # Nothing valid to apply

        success = apply_selection_replacement(result, paste_back=True)
        if success:
            self._result_text.config(state="normal")
            self._result_text.delete(1.0, tk.END)
            self._result_text.insert(tk.END, "Applied in-place!")
            self._result_text.config(state="disabled")

    # ------------------------------------------------------------------
    # Editor info update
    # ------------------------------------------------------------------

    def update_editor_info(self, editor_info: dict[str, Any]) -> None:
        """Update the badge with detected editor information."""
        self._editor_info = editor_info
        etype = editor_info.get("editor_type", "none")
        tools = editor_info.get("available_tools", [])
        tools_str = ", ".join(tools) if tools else "none"
        title = editor_info.get("title", "")[:40] or "(no window)"

        self._badge_label.config(text=f"{title}  --  {etype}  --  Tools: {tools_str}")


# ---------------------------------------------------------------------------
# Install prompt modal
# ---------------------------------------------------------------------------


def show_install_prompt(prompt_text: str) -> None:
    """Show a modal dialog prompting the user to install an editor."""

    def _destroy(_: Any = None) -> None:
        top.destroy()

    top = tk.Toplevel()
    top.title("Install Document Editor")
    top.geometry("500x280")
    top.transient()
    top.grab_set()

    style_bg = "#1a1a2e"
    accent = "#4ecca3"

    top.configure(bg=style_bg)

    tk.Label(
        top,
        text="Document Editor Not Detected",
        bg=style_bg,
        fg=accent,
        font=("Segoe UI", 14, "bold"),
    ).pack(pady=(20, 10))

    scroll = tk.Scrollbar(top)
    scroll.pack(side="right", fill="y")

    info_text = tk.Text(
        top,
        height=8,
        bg="#0f3460",
        fg="#e0e0e0",
        font=("Consolas", 9),
        wrap="word",
        yscrollcommand=scroll.set,
    )
    info_text.pack(fill="both", expand=True, padx=20, pady=(0, 10))
    scroll.config(command=info_text.yview)

    info_text.insert(tk.END, prompt_text)
    info_text.config(state="disabled")

    def copy_command() -> None:
        top.clipboard_clear()
        for line in prompt_text.split("\n"):
            if "winget install" in line:
                cmd = line.strip()
                top.clipboard_append(cmd)
                break

    tk.Button(
        top,
        text="Copy Install Command",
        bg=accent,
        fg="#1a1a2e",
        font=("Segoe UI", 9, "bold"),
        command=copy_command,
    ).pack(pady=(0, 5))

    tk.Button(
        top,
        text="OK",
        bg="#e94560",
        fg="white",
        font=("Segoe UI", 9),
        command=_destroy,
    ).pack(pady=5)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the coPPeR desktop overlay."""
    # Detect editors on startup
    editor_result = detect_installed_editors()

    if not editor_result["has_editor"]:
        prompt = editor_result.get("install_prompt")
        if prompt:
            show_install_prompt(prompt)

    overlay = CopperOverlay()

    # Register global hotkey Ctrl+Alt+Space
    registered = _register_hotkey()
    if registered:
        print("Hotkey registered: Ctrl+Alt+Space (show overlay)")
    else:
        print("Warning: Could not register global hotkey. Use the window manually.")

    # Bind WM_DESTROY to cleanup
    def on_closing() -> None:
        _unregister_hotkey()
        overlay.root.destroy()

    overlay.root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start a thread to periodically update editor context
    def _update_loop() -> None:
        while True:
            try:
                ctx = get_active_window_context()
                overlay.update_editor_info(ctx)
            except Exception:  # noqa: BLE001, S110
                pass
            overlay.root.after(
                2000, lambda: threading.Thread(target=_update_loop).start()
            )
            break  # Only start one background loop

    threading.Thread(target=_update_loop, daemon=True).start()

    overlay.root.mainloop()


if __name__ == "__main__":
    main()
