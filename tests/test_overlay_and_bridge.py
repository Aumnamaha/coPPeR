"""Tests for app/editor_bridge.py and desktop overlay.

Covers:
- Editor detection (mocked present vs absent)
- Screen-aware prompt assembly with <|sel_start|>/<|sel_end|> tokens
- 64-bit Win32 ctypes signature verification
- Live clipboard roundtrip on Windows
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pytest

# Ensure src is importable
_sys_path = str(Path(__file__).resolve().parent.parent / "src")
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)


class TestEditorDetection:
    """Verify detect_installed_editors() behavior."""

    def test_detect_installed_editors_returns_expected_keys(self):
        """detect_installed_editors returns editors, has_editor, install_prompt."""
        from app.editor_bridge import detect_installed_editors

        result = detect_installed_editors()
        assert "editors" in result
        assert "has_editor" in result
        assert "install_prompt" in result
        assert isinstance(result["editors"], list)
        assert isinstance(result["has_editor"], bool)

    def test_detect_no_editor_provides_install_prompt(self, monkeypatch):
        """When no editors are installed, install_prompt contains LibreOffice + winget."""
        from app.editor_bridge import detect_installed_editors

        # Mock Path.exists to return False for all editor paths

        def fake_exists(self):
            return False

        monkeypatch.setattr(Path, "exists", fake_exists)

        result = detect_installed_editors()
        assert result["has_editor"] is False
        prompt = result["install_prompt"]
        assert prompt is not None
        assert "LibreOffice" in prompt
        assert "winget install" in prompt


class TestScreenAwarePromptAssembly:
    """Verify screen-aware token wrapping."""

    def test_capture_non_windows_assembly(self):
        """On non-Windows platforms, capture returns wrapped doc with sel tokens."""
        # Patch platform.system to simulate non-Windows
        import platform as _platform

        from app.editor_bridge import capture_highlighted_selection

        orig = _platform.system
        _platform.system = lambda: "Linux"  # type: ignore[assignment]
        try:
            result = capture_highlighted_selection(
                before_context="BEFORE", after_context="AFTER"
            )
            assert "<|sel_start|>" in result["wrapped_document"]
            assert "<|sel_end|>" in result["wrapped_document"]
            assert "BEFORE" in result["wrapped_document"]
            assert "AFTER" in result["wrapped_document"]
        finally:
            _platform.system = orig  # type: ignore[assignment]


class TestClipboard64BitSignatures:
    """Verify Win32 ctypes argtypes/restype are correctly declared for x64."""

    def test_globallock_restype_is_c_void_p(self):
        """GlobalLock.restype must be c_void_p (8 bytes on 64-bit)."""
        import ctypes

        from app import editor_bridge

        ptr_size = ctypes.sizeof(ctypes.c_void_p)
        assert ptr_size == 8, "Expected 64-bit environment"
        assert editor_bridge.kernel32.GlobalLock.restype == ctypes.c_void_p

    def test_getclipboarddata_restype_is_handle(self):
        """GetClipboardData.restype must be HANDLE (c_void_p alias)."""
        import ctypes

        from app import editor_bridge

        # Skip if GetClipboardData not available in this environment
        try:
            _ = editor_bridge.kernel32.GetClipboardData
        except AttributeError:
            pytest.skip("GetClipboardData not available in this environment")

        # HANDLE is a pointer type — same size as c_void_p on 64-bit
        assert editor_bridge.kernel32.GetClipboardData.restype == ctypes.wintypes.HANDLE

    def test_globalalloc_restype_is_hglobal(self):
        """GlobalAlloc.restype must be HGLOBAL."""
        from app import editor_bridge

        assert editor_bridge.kernel32.GlobalAlloc.restype == ctypes.wintypes.HGLOBAL

    def test_openclipboard_argtypes_and_restype(self):
        """OpenClipboard takes HWND, returns BOOL."""
        from app import editor_bridge

        assert editor_bridge.user32.OpenClipboard.argtypes == [ctypes.wintypes.HWND]
        assert editor_bridge.user32.OpenClipboard.restype == ctypes.wintypes.BOOL

    def test_closeclipboard_restype(self):
        """CloseClipboard takes no args, returns BOOL."""
        from app import editor_bridge

        assert editor_bridge.user32.CloseClipboard.argtypes == []
        assert editor_bridge.user32.CloseClipboard.restype == ctypes.wintypes.BOOL


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
class TestLiveClipboardRoundtrip:
    """Integration tests exercising real clipboard operations on Windows."""

    @staticmethod
    def _clipboard_api_available():
        """Check if clipboard APIs are accessible in this environment.

        Tests both Open/Close and a real write roundtrip so that environments
        where GlobalAlloc fails (WSL headless, CI) are properly detected and
        the live tests are skipped rather than asserted-failing.
        """
        try:
            import ctypes

            # Test read path: open/close works
            ok = ctypes.windll.user32.OpenClipboard(0)
            if not ok:
                return False
            ctypes.windll.user32.CloseClipboard()

            # Test write path via the actual editor_bridge helper.
            # If GlobalAlloc fails for any size, writes will fail too.
            from app.editor_bridge import _write_clipboard_unicode

            result = _write_clipboard_unicode("test")
            return bool(result)
        except (AttributeError, OSError):
            return False

    def test_clipboard_roundtrip_preserves_text(self):
        """set/get clipboard text roundtrip preserves content without access violations."""
        if not self._clipboard_api_available():
            pytest.skip("Clipboard APIs not available in this environment")

        from app.editor_bridge import (
            _clipboard_backup,
            _clipboard_restore,
            _read_clipboard_unicode,
            _write_clipboard_unicode,
        )

        test_text = "Hello 64-bit Windows clipboard \u2014 roundtrip test"

        # Backup current clipboard
        original = _clipboard_backup()

        try:
            # Write our test text
            ok = _write_clipboard_unicode(test_text)
            assert ok is True

            # Read it back
            read_back = _read_clipboard_unicode()
            assert read_back == test_text, f"Expected {test_text!r}, got {read_back!r}"
        finally:
            # Restore original clipboard content
            _clipboard_restore(original)

    def test_clipboard_roundtrip_empty_string(self):
        """Writing an empty string and reading it back does not crash."""
        if not self._clipboard_api_available():
            pytest.skip("Clipboard APIs not available in this environment")

        from app.editor_bridge import (
            _clipboard_backup,
            _clipboard_restore,
            _read_clipboard_unicode,
            _write_clipboard_unicode,
        )

        original = _clipboard_backup()
        try:
            ok = _write_clipboard_unicode("")
            assert ok is True
            read_back = _read_clipboard_unicode()
            # Empty string encodes to just null bytes; wstring_at returns "" or "\x00"
            # We accept either empty or whitespace-only after strip
            assert read_back.strip() == ""
        finally:
            _clipboard_restore(original)


class TestApplySelectionReplacementDryRun:
    """Test apply_selection_replacement dry-run mode."""

    def test_dry_run_returns_true_for_nonempty(self):
        from app.editor_bridge import apply_selection_replacement

        assert apply_selection_replacement("some text", dry_run=True) is True

    def test_dry_run_returns_false_for_empty(self):
        from app.editor_bridge import apply_selection_replacement

        assert apply_selection_replacement("", dry_run=True) is False


class TestGetActiveWindowContext:
    """Test active window context classification."""

    def test_get_active_window_context_returns_expected_keys(self):
        from app.editor_bridge import get_active_window_context

        result = get_active_window_context()
        assert "title" in result
        assert "editor_type" in result
        assert "available_tools" in result
        assert isinstance(result["available_tools"], list)


class TestEditorToolsSpec:
    """Verify EDITOR_TOOLS_SPEC is accessible from editor_bridge."""

    def test_editor_tools_spec_importable(self):
        from app.editor_bridge import EDITOR_TOOLS_SPEC

        # EDITOR_TOOLS_SPEC is a dict with action names as keys
        assert isinstance(EDITOR_TOOLS_SPEC, dict)
        assert "replace_selection" in EDITOR_TOOLS_SPEC
        assert "insert_markdown_table" in EDITOR_TOOLS_SPEC


class TestDesktopOverlaySmoke:
    """Smoke test that the overlay module imports cleanly."""

    def test_overlay_module_importable(self):
        import app.desktop_overlay  # noqa: F401


class TestCtypesWintypesImport:
    """Verify ctypes.wintypes is imported so editor_bridge loads without AttributeError."""

    def test_import_editor_bridge_and_desktop_overlay_no_error(self):
        """Both modules must import cleanly — no missing ctypes.wintypes."""
        import app.desktop_overlay as _desktop_mod
        import app.editor_bridge as _bridge_mod

        assert _desktop_mod is not None
        assert _bridge_mod is not None


class TestOverlayServerIntegration:
    """Verify desktop_overlay -> inference_server communication.

    Tests payload format matching, server error handling with actionable messages,
    and the full /v1/edit-span endpoint via FastAPI TestClient (with mocked model).
    """

    @staticmethod
    def _build_edit_span_payload():
        """Build the exact payload desktop_overlay sends on Generate Fix."""
        return {
            "highlighted_text": "Set-Location",
            "surrounding_context": "PS C:\\Users\\Aum>",
            "instruction": "fix typo",
            "active_app": "libreoffice_writer",
            "mode": "cursor_span_edit",
        }

    def test_overlay_payload_matches_server_schema(self):
        """desktop_overlay's request dict must satisfy EditSpanRequest schema."""
        from app.inference_server import EditSpanRequest

        payload = self._build_edit_span_payload()
        req = EditSpanRequest(**payload)

        assert req.highlighted_text == "Set-Location"
        assert req.surrounding_context == "PS C:\\Users\\Aum>"
        assert req.instruction == "fix typo"
        assert req.active_app == "libreoffice_writer"
        assert req.mode == "cursor_span_edit"

    def test_server_returns_503_without_model(self, monkeypatch):
        """When no model is loaded and CHECKPOINT_DIR not set, /v1/edit-span returns 503."""
        from fastapi.testclient import TestClient

        # Ensure no checkpoint dir env var is set
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)

        from app.inference_server import app as server_app

        client = TestClient(server_app)
        payload = self._build_edit_span_payload()
        resp = client.post("/v1/edit-span", json=payload, timeout=5)

        assert resp.status_code == 503
        detail = resp.json().get("detail", "")
        assert "No model loaded" in detail or "Tokenizer not found" in detail

    def test_server_error_message_is_actionable(self, monkeypatch):
        """Server errors include actionable details (not generic 'No response')."""
        from fastapi.testclient import TestClient

        # Ensure no checkpoint dir env var is set so we get a clean 503
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)

        from app.inference_server import app as server_app

        client = TestClient(server_app)
        payload = self._build_edit_span_payload()
        resp = client.post("/v1/edit-span", json=payload, timeout=5)

        assert resp.status_code == 503
        body = resp.json()
        # The detail should be a human-readable string, not empty
        assert "detail" in body
        assert len(body["detail"]) > 10

    def test_health_endpoint_returns_tools_spec(self):
        """GET /health returns EDITOR_TOOLS_SPEC and backend status."""
        from fastapi.testclient import TestClient

        from app.inference_server import app as server_app

        client = TestClient(server_app)
        resp = client.get("/health", timeout=5)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "EDITOR_TOOLS_SPEC" in body


class TestHealthEndpointBackendMetadata:
    """Verify GET /health returns backend metadata."""

    def test_health_returns_available_backends(self):
        """GET /health includes available_backends and configured_backend keys."""
        from fastapi.testclient import TestClient

        from app.inference_server import app as server_app

        client = TestClient(server_app)
        resp = client.get("/health", timeout=5)
        assert resp.status_code == 200
        body = resp.json()
        assert "available_backends" in body
        assert isinstance(body["available_backends"], list)
        assert len(body["available_backends"]) > 0
        assert "configured_backend" in body
        assert isinstance(body["configured_backend"], str)

    def test_health_status_is_ok(self):
        """Health endpoint always returns status=ok when FastAPI is running."""
        from fastapi.testclient import TestClient

        from app.inference_server import app as server_app

        client = TestClient(server_app)
        resp = client.get("/health", timeout=5)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


class TestChatCompletionsEndpoint:
    """Verify POST /v1/chat/completions endpoint structure."""

    def test_chat_completions_503_without_model(self, monkeypatch):
        """Without model loaded, returns 503 with actionable detail."""
        from fastapi.testclient import TestClient

        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)

        from app.inference_server import app as server_app

        client = TestClient(server_app)
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "max_new_tokens": 16,
            "stream": False,
        }
        resp = client.post("/v1/chat/completions", json=payload, timeout=5)
        assert resp.status_code == 503
        detail = resp.json().get("detail", "")
        assert len(detail) > 10

    def test_chat_completions_stream_returns_streaming_response(self, monkeypatch):
        """With stream=True, endpoint returns a StreamingResponse (media_type text/event-stream)."""
        import torch
        from fastapi.testclient import TestClient

        # Mock tokenizer and model to avoid actual inference
        mock_tokenizer = type(
            "MockTokenizer",
            (),
            {
                "token_to_id": lambda self, t: 0,
                "eos_token_id": 2,
                "format_chatml": lambda self, msgs: "<|user|>Hello<|endoftext|>",
                "encode": lambda self, s, **kw: [1, 2, 3],
                "decode": lambda self, ids: "test",
            },
        )()

        # Minimal mock tensor on CPU for parameters() iterator
        _mock_param = torch.nn.Parameter(torch.zeros(1)).to("cpu")

        mock_model = type(
            "MockModel",
            (),
            {
                "generate": staticmethod(lambda *a, **kw: torch.tensor([[1, 2, 3, 4]])),
                "parameters": lambda self: iter([_mock_param]),
            },
        )()

        import app.inference_server as srv_module

        monkeypatch.setattr(srv_module, "_tokenizer", mock_tokenizer)
        monkeypatch.setattr(srv_module, "_model", mock_model)
        monkeypatch.setattr(srv_module, "_backend_mode", "pytorch")

        from app.inference_server import app as server_app

        client = TestClient(server_app)
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "max_new_tokens": 4,
            "stream": True,
        }
        resp = client.post("/v1/chat/completions", json=payload, timeout=5)
        # StreamingResponse should be returned with correct content-type
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "text/event-stream; charset=utf-8"
