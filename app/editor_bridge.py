#!/usr/bin/env python3
r"""Windows screen-aware editor bridge for coPPeR.

Detects installed document editors, captures cursor-highlighted selections via clipboard,
and applies surgical span replacements in-place.
"""

from __future__ import annotations

import ctypes
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 64-bit Win32 API signatures (fixes pointer truncation on x64 Windows)
# ---------------------------------------------------------------------------

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# --- user32 functions ---
user32.OpenClipboard.argtypes = [ctypes.wintypes.HWND]
user32.OpenClipboard.restype = ctypes.wintypes.BOOL

user32.CloseClipboard.argtypes = []
user32.CloseClipboard.restype = ctypes.wintypes.BOOL

user32.EmptyClipboard.argtypes = []
user32.EmptyClipboard.restype = ctypes.wintypes.BOOL

user32.GetClipboardData.argtypes = [ctypes.wintypes.UINT]
user32.GetClipboardData.restype = ctypes.wintypes.HANDLE

user32.SetClipboardData.argtypes = [ctypes.wintypes.UINT, ctypes.wintypes.HANDLE]
user32.SetClipboardData.restype = ctypes.wintypes.HANDLE

user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = ctypes.wintypes.HWND

user32.GetWindowTextLengthW.argtypes = [ctypes.wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int

user32.GetWindowTextW.argtypes = [
    ctypes.wintypes.HWND,
    ctypes.wintypes.LPWSTR,
    ctypes.c_int,
]
user32.GetWindowTextW.restype = ctypes.c_int

# SendInput: SendInput(nInputs, pInputs, cbSize) -> c_uint32
user32.SendInput.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
user32.SendInput.restype = ctypes.c_uint32

# --- kernel32 functions ---
kernel32.GlobalAlloc.argtypes = [ctypes.wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = ctypes.wintypes.HGLOBAL

kernel32.GlobalLock.argtypes = [ctypes.wintypes.HGLOBAL]
kernel32.GlobalLock.restype = ctypes.c_void_p

kernel32.GlobalUnlock.argtypes = [ctypes.wintypes.HGLOBAL]
kernel32.GlobalUnlock.restype = ctypes.wintypes.BOOL

kernel32.GlobalFree.argtypes = [ctypes.wintypes.HGLOBAL]
kernel32.GlobalFree.restype = ctypes.wintypes.HGLOBAL


# Import screen-aware tool specs from data_schemas
_sys_path = str(Path(__file__).resolve().parent.parent / "src")
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from data_schemas import EDITOR_TOOLS_SPEC

# ---------------------------------------------------------------------------
# Editor detection
# ---------------------------------------------------------------------------


def detect_installed_editors() -> dict[str, Any]:
    """Detect installed document editors on Windows.

    Checks standard paths for LibreOffice Writer (soffice.exe),
    Microsoft Word (WINWORD.EXE via registry or path), and WPS/OnlyOffice.

    Returns:
        {"editors": list[dict], "has_editor": bool, "install_prompt": str | None}
    """
    editors = []
    soffice = r"C:\Program Files\LibreOffice\program\soffice.exe"
    if Path(soffice).exists():
        editors.append(
            {
                "name": "LibreOffice Writer",
                "executable": soffice,
                "platform_key": "libreoffice_writer",
            }
        )

    # Check Microsoft Word via registry (common paths)
    word_paths = [
        r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
    ]
    for wp in word_paths:
        if Path(wp).exists():
            editors.append(
                {
                    "name": "Microsoft Word",
                    "executable": wp,
                    "platform_key": "microsoft_word",
                }
            )
            break

    # Fallback: check via winget / Get-Command (best-effort)
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-Command soffice -ErrorAction SilentlyContinue).Source",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.stdout.strip():
            editors.append(
                {
                    "name": "LibreOffice Writer (PATH)",
                    "executable": Path(result.stdout.strip()).resolve(),
                    "platform_key": "libreoffice_writer",
                }
            )
    except Exception:  # noqa: BLE001, S110
        pass

    has_editor = len(editors) > 0

    install_prompt = None
    if not has_editor:
        lines = [
            "No document editor detected.",
            "",
            "Install LibreOffice (recommended, free & open-source):",
            "   winget install TheDocumentFoundation.LibreOffice",
            "",
            "Or Microsoft Word (included with Microsoft 365).",
            "After installation, restart the coPPeR overlay app.",
        ]
        install_prompt = "\n".join(lines)

    return {
        "editors": editors,
        "has_editor": has_editor,
        "install_prompt": install_prompt,
    }


# ---------------------------------------------------------------------------
# Active window context
# ---------------------------------------------------------------------------


def get_active_window_context() -> dict[str, Any]:
    """Read the active foreground window title and classify the editor.

    Uses Windows user32 API (GetForegroundWindow + GetWindowTextW).

    Returns:
        {"title": str, "editor_type": str, "available_tools": list[str]}
    """
    title = ""
    try:
        hwnd = user32.GetForegroundWindow()
        if hwnd:
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value or ""
    except (OSError, AttributeError):
        pass

    # Classify editor from window title
    title_lower = title.lower()
    if "libreoffice" in title_lower or "soffice" in title_lower:
        editor_type = "libreoffice_writer"
    elif "winword" in title_lower or "microsoft word" in title_lower:
        editor_type = "microsoft_word"
    elif any(kw in title_lower for kw in ["notepad", "vs code", "sublime"]):
        editor_type = "markdown_editor"
    else:
        editor_type = "generic_editor"

    tools = EDITOR_TOOLS_SPEC.get("supported_actions", [])
    return {
        "title": title,
        "editor_type": editor_type,
        "available_tools": tools,
    }


# ---------------------------------------------------------------------------
# Clipboard helpers with proper 64-bit pointer handling
# ---------------------------------------------------------------------------

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
GMEM_ZEROINIT = 0x0040
GMEM_FLAGS = GMEM_MOVEABLE | GMEM_ZEROINIT


def _read_clipboard_unicode() -> str | None:
    """Read CF_UNICODETEXT from clipboard. Returns text or None."""
    handle = user32.GetClipboardData(CF_UNICODETEXT)
    if not handle:
        return None

    ptr = kernel32.GlobalLock(handle)
    try:
        if not ptr:
            return None
        # c_void_p on 64-bit Windows is 8 bytes; wstring_at handles null-terminated wide string
        return ctypes.wstring_at(ptr)
    finally:
        kernel32.GlobalUnlock(handle)


def _write_clipboard_unicode(text: str) -> bool:
    """Write text to clipboard as CF_UNICODETEXT. Returns True on success."""
    encoded = text.encode("utf-16le") + b"\x00\x00"
    size = len(encoded)

    handle = kernel32.GlobalAlloc(GMEM_FLAGS, size)
    if not handle:
        return False

    ptr = kernel32.GlobalLock(handle)
    try:
        if not ptr:
            return False
        ctypes.memmove(ptr, encoded, size)
    finally:
        kernel32.GlobalUnlock(handle)

    result = user32.SetClipboardData(CF_UNICODETEXT, handle)
    # SetClipboardData takes ownership of the handle on success
    return result is not None


def _clipboard_backup() -> str | None:
    """Save current clipboard content to restore later."""
    if not user32.OpenClipboard(0):
        return None

    try:
        data = _read_clipboard_unicode()
        return data
    finally:
        user32.CloseClipboard()


def _clipboard_restore(text: str | None) -> bool:
    """Restore previously saved clipboard content."""
    if text is None:
        return True

    try:
        if not user32.OpenClipboard(0):
            return False
        try:
            user32.EmptyClipboard()
            ok = _write_clipboard_unicode(text)
            return ok
        finally:
            user32.CloseClipboard()
    except (OSError, AttributeError):
        return False


def capture_highlighted_selection(
    before_context: str = "", after_context: str = ""
) -> dict[str, Any]:
    """Capture currently highlighted text via clipboard (backup/restore).

    Strategy:
        1. Backup current clipboard content
        2. Send Ctrl+C to copy selection
        3. Read new clipboard content as the selected text
        4. Restore original clipboard

    Returns:
        {"selected_text": str, "wrapped_document": str, "clipboard_restored": bool}
    """
    if platform.system() != "Windows":
        wrapped = (
            f"<|doc_start|>{before_context}"
            f"<|sel_start|><|sel_end|>{after_context}<|doc_end|>"
        )
        return {
            "selected_text": "",
            "wrapped_document": wrapped,
            "clipboard_restored": True,
        }

    original = _clipboard_backup()

    try:
        # Send Ctrl+C to copy selection using SendInput
        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_uint16),
                ("wScan", ctypes.c_uint16),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.c_uint32), ("ki", KEYBDINPUT)]

        ctrl_down = INPUT(INPUT_KEYBOARD, KEYBDINPUT(0x11, 0, 0, 0, None))
        c_down = INPUT(INPUT_KEYBOARD, KEYBDINPUT(0x43, 0, 0, 0, None))
        c_up = INPUT(INPUT_KEYBOARD, KEYBDINPUT(0x43, 0, KEYEVENTF_KEYUP, 0, None))
        ctrl_up = INPUT(INPUT_KEYBOARD, KEYBDINPUT(0x11, 0, KEYEVENTF_KEYUP, 0, None))

        arr_type = INPUT * 4
        user32.SendInput(
            4,
            ctypes.byref(arr_type(ctrl_down, c_down, c_up, ctrl_up)),
            ctypes.sizeof(INPUT),
        )

        time.sleep(0.05)

        # Read clipboard for selected text
        if not user32.OpenClipboard(0):
            return {
                "selected_text": "",
                "wrapped_document": "",
                "clipboard_restored": False,
            }

        try:
            h = user32.GetClipboardData(CF_UNICODETEXT)
            selected = ""
            if h:
                ptr = kernel32.GlobalLock(h)
                try:
                    if ptr:
                        selected = ctypes.wstring_at(ptr).strip()
                finally:
                    kernel32.GlobalUnlock(h)
        finally:
            user32.CloseClipboard()

    finally:
        restored = _clipboard_restore(original)

    # Build wrapped document string with screen-aware tokens
    wrapped = (
        f"<|doc_start|>{before_context}"
        f"<|sel_start|>{selected}<|sel_end|>"
        f"{after_context}<|doc_end|>"
    )

    return {
        "selected_text": selected,
        "wrapped_document": wrapped,
        "clipboard_restored": restored,
    }


# ---------------------------------------------------------------------------
# Apply replacement in-place via clipboard paste
# ---------------------------------------------------------------------------


def apply_selection_replacement(
    replacement_text: str,
    paste_back: bool = True,
    dry_run: bool = False,
) -> bool:
    """Place replacement text on clipboard and optionally send Ctrl+V.

    Args:
        replacement_text: Text to replace the highlighted span with.
        paste_back: If True, actually simulate Ctrl+V to apply in-place.
        dry_run: If True, only check readiness without sending keystrokes.

    Returns:
        True if successful (or dry-run passed).
    """
    if dry_run:
        return len(replacement_text) > 0

    # Place text on clipboard
    _clipboard_backup()
    try:
        if not user32.OpenClipboard(0):
            return False

        try:
            user32.EmptyClipboard()
            ok = _write_clipboard_unicode(replacement_text)
            return ok
        finally:
            user32.CloseClipboard()
    except (OSError, AttributeError):
        return False

    if not paste_back:
        return True

    # Simulate Ctrl+V to replace selection in-place
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_uint16),
            ("wScan", ctypes.c_uint16),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_uint32), ("ki", KEYBDINPUT)]

    ctrl_down = INPUT(INPUT_KEYBOARD, KEYBDINPUT(0x11, 0, 0, 0, None))
    v_down = INPUT(INPUT_KEYBOARD, KEYBDINPUT(0x56, 0, 0, 0, None))
    v_up = INPUT(INPUT_KEYBOARD, KEYBDINPUT(0x56, 0, KEYEVENTF_KEYUP, 0, None))
    ctrl_up = INPUT(INPUT_KEYBOARD, KEYBDINPUT(0x11, 0, KEYEVENTF_KEYUP, 0, None))

    arr_type = INPUT * 4
    user32.SendInput(
        4,
        ctypes.byref(arr_type(ctrl_down, v_down, v_up, ctrl_up)),
        ctypes.sizeof(INPUT),
    )

    return True


# ---------------------------------------------------------------------------
# Main (standalone test)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    result = detect_installed_editors()
    print(json.dumps(result, indent=2))
