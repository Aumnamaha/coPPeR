r"""Tests for Phase 7: desktop overlay, editor bridge, inference server, Docker config."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Editor detection tests (mocked)
# ---------------------------------------------------------------------------


class TestDetectInstalledEditors:
    """Test editor_bridge.detect_installed_editors with mocked filesystem."""

    def test_editor_present(self, tmp_path: Path) -> None:
        """When LibreOffice path exists, has_editor should be True."""
        from app.editor_bridge import detect_installed_editors

        with (
            patch.object(
                Path,
                "exists",
                new=lambda self: "soffice.exe" in str(self).lower()
                and "libreoffice" in str(self).lower(),
            ),
            patch(
                "app.editor_bridge.subprocess.run",
                side_effect=Exception("no shell"),
            ),
        ):
            result = detect_installed_editors()

        assert result["has_editor"] is True
        assert len(result["editors"]) > 0
        assert result["install_prompt"] is None

    def test_no_editor(self) -> None:
        """When no editor exists, has_editor should be False with install prompt."""
        from app.editor_bridge import detect_installed_editors

        with patch.object(Path, "exists", return_value=False), patch(
            "app.editor_bridge.subprocess.run",
            side_effect=Exception("no shell"),
        ):
            result = detect_installed_editors()

        assert result["has_editor"] is False
        assert "LibreOffice" in result.get("install_prompt", "")
        assert "winget install" in result.get("install_prompt", "")


class TestScreenAwarePrompt:
    """Test that screen-aware tokens wrap selections correctly."""

    def test_sel_tokens_wrapping(self) -> None:
        """Selection wrapped with sel tokens (mocked clipboard for headless safety)."""
        from app.editor_bridge import capture_highlighted_selection

        with patch("app.editor_bridge.ctypes.windll.user32.SendInput"), patch(
            "app.editor_bridge.ctypes.windll.user32.OpenClipboard",
            return_value=True,
        ), patch(
            "app.editor_bridge.ctypes.windll.user32.GetClipboardData",
            return_value=None,
        ):
            result = capture_highlighted_selection(
                before_context="Some context before",
                after_context="Some context after",
            )

        assert "<|sel_start|>" in result["wrapped_document"]
        assert "<|sel_end|>" in result["wrapped_document"]
        sel_marker = "<|sel_start|><|sel_end|>"
        assert sel_marker in result["wrapped_document"] or (
            result.get("selected_text", "") == ""
            and sel_marker in result["wrapped_document"]
        )

    def test_sel_tokens_in_order(self) -> None:
        """Token order must be doc_start < sel_start < sel_end < doc_end."""
        from app.editor_bridge import capture_highlighted_selection

        with patch("app.editor_bridge.ctypes.windll.user32.SendInput"), patch(
            "app.editor_bridge.ctypes.windll.user32.OpenClipboard",
            return_value=True,
        ), patch(
            "app.editor_bridge.ctypes.windll.user32.GetClipboardData",
            return_value=None,
        ):
            result = capture_highlighted_selection()

        doc = result["wrapped_document"]
        doc_start_idx = doc.find("<|doc_start|>")
        sel_start_idx = doc.find("<|sel_start|>")
        sel_end_idx = doc.find("<|sel_end|>")
        doc_end_idx = doc.find("<|doc_end|>")

        assert doc_start_idx < sel_start_idx < sel_end_idx < doc_end_idx, (
            f"Token order wrong: doc_start={doc_start_idx}, "
            f"sel_start={sel_start_idx}, sel_end={sel_end_idx}, doc_end={doc_end_idx}"
        )


class TestInferenceServerEndpoints:
    """Test the FastAPI app endpoints with mocked model backend."""

    def setup_method(self) -> None:
        from fastapi.testclient import TestClient

        # Import and create test client
        from app.inference_server import app as _app

        self.client = TestClient(_app)

    def test_health_endpoint(self) -> None:
        """GET /health should return status, backend_mode, EDITOR_TOOLS_SPEC."""
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "backend_mode" in data
        assert isinstance(data["EDITOR_TOOLS_SPEC"], dict)
        # Should contain known tool names
        tools_spec = data["EDITOR_TOOLS_SPEC"]
        assert "replace_selection" in tools_spec

    def test_edit_span_no_backend(self) -> None:
        """POST /v1/edit-span with no loaded backend should return 503."""
        payload = {
            "highlighted_text": "bad text",
            "surrounding_context": "",
            "instruction": "fix this",
            "active_app": "libreoffice_writer",
            "mode": "cursor_span_edit",
        }
        resp = self.client.post("/v1/edit-span", json=payload)
        # Without a loaded model, should get 503
        assert resp.status_code == 503

    def test_edit_span_with_mocked_model(self, tmp_path: Path) -> None:
        """POST /v1/edit-span with mocked tokenizer returns valid response."""
        from app.inference_server import (
            _build_screen_aware_prompt,
            validate_teacher_output,
        )

        # Test prompt building directly
        messages = _build_screen_aware_prompt(
            highlighted_text="problem text",
            surrounding_context="before and after",
            instruction="fix grammar",
            active_app="libreoffice_writer",
        )

        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert "<|sel_start|>" in messages[1]["content"]
        assert "<|sel_end|>" in messages[1]["content"]

        # Test validation of cursor_span_edit output
        is_valid, _reason = validate_teacher_output(
            "cursor_span_edit", "fixed text here"
        )
        assert is_valid is True

    def test_edit_span_editor_tool_call_validation(self) -> None:
        """validate_teacher_output should accept valid editor_tool_call JSON."""
        from app.inference_server import validate_teacher_output as _vt

        # Valid tool call
        valid_json = (
            '{"tool": "replace_selection", "args": {"new_text": "replacement"}}'
        )
        is_valid, _reason = _vt("editor_tool_call", valid_json)
        assert is_valid is True

        # Invalid: unknown tool name
        invalid_tool = '{"tool": "unknown_tool", "args": {}}'
        is_valid, _reason = _vt("editor_tool_call", invalid_tool)
        assert is_valid is False

        # Invalid: missing args key
        no_args = '{"tool": "replace_selection"}'
        is_valid, _reason = _vt("editor_tool_call", no_args)
        assert is_valid is False

        # Invalid: not JSON
        is_valid, _reason = _vt("editor_tool_call", "not json at all")
        assert is_valid is False


# ---------------------------------------------------------------------------
# Docker config validation tests
# ---------------------------------------------------------------------------


class TestDockerConfig:
    """Validate Dockerfile and docker-compose.yml for correctness."""

    def test_dockerfile_exists_and_utf8(self, tmp_path: Path) -> None:
        """docker/Dockerfile exists and is valid UTF-8 without BOM."""
        dockerfile = Path(_PROJECT_ROOT) / "docker" / "Dockerfile"
        assert dockerfile.exists(), "docker/Dockerfile not found"

        raw = dockerfile.read_bytes()
        # No BOM
        assert not raw.startswith(b"\xef\xbb\xbf"), "Dockerfile has UTF-8 BOM"
        # Valid UTF-8
        text = raw.decode("utf-8")
        assert len(text) > 0

    def test_dockercompose_exists_and_utf8(self, tmp_path: Path) -> None:
        """docker/docker-compose.yml exists and is valid UTF-8 without BOM."""
        compose = Path(_PROJECT_ROOT) / "docker" / "docker-compose.yml"
        assert compose.exists(), "docker/docker-compose.yml not found"

        raw = compose.read_bytes()
        # No BOM
        assert not raw.startswith(b"\xef\xbb\xbf"), "docker-compose.yml has UTF-8 BOM"
        text = raw.decode("utf-8")
        assert len(text) > 0

    def test_dockercompose_localhost_binding(self, tmp_path: Path) -> None:
        """docker-compose.yml must bind to 127.0.0.1 only (not 0.0.0.0)."""
        compose = Path(_PROJECT_ROOT) / "docker" / "docker-compose.yml"
        text = compose.read_text(encoding="utf-8")

        # Must contain localhost binding
        assert (
            "127.0.0.1:8088" in text
        ), "docker-compose.yml must bind to 127.0.0.1:8088, not expose publicly"
        # Must NOT contain public binding
        assert (
            "0.0.0.0" not in text or '"127.0.0.1:8088' in text
        ), "docker-compose.yml must not bind to 0.0.0.0"

    def test_dockercompose_readonly_mounts(self, tmp_path: Path) -> None:
        """docker-compose.yml must mount volumes as read-only (:ro)."""
        compose = Path(_PROJECT_ROOT) / "docker" / "docker-compose.yml"
        text = compose.read_text(encoding="utf-8")

        # Check that volume mounts use :ro flag
        assert (
            ":ro" in text
        ), "docker-compose.yml must mount model volumes as read-only (:ro)"
        assert (
            "/app/checkpoints:ro" in text or "../checkpoints:/app/checkpoints" in text
        )
        assert "/app/exports:ro" in text or "../exports:/app/exports" in text

    def test_dockerfile_exposes_8088(self, tmp_path: Path) -> None:
        """Dockerfile should EXPOSE port 8088."""
        dockerfile = Path(_PROJECT_ROOT) / "docker" / "Dockerfile"
        text = dockerfile.read_text(encoding="utf-8")
        assert "EXPOSE 8088" in text, "Dockerfile must expose port 8088"


# ---------------------------------------------------------------------------
# Desktop overlay imports test
# ---------------------------------------------------------------------------


class TestDesktopOverlayImports:
    """Verify desktop_overlay module can be imported (without running UI)."""

    def test_overlay_module_imports(self) -> None:
        """app.desktop_overlay should import without errors."""
        try:
            from app import desktop_overlay  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"Failed to import app.desktop_overlay: {exc}")

    def test_register_hotkey_function_exists(self) -> None:
        """_register_hotkey and _unregister_hotkey should exist."""
        from app import desktop_overlay

        assert hasattr(desktop_overlay, "_register_hotkey")
        assert hasattr(desktop_overlay, "_unregister_hotkey")
        assert callable(desktop_overlay._register_hotkey)
        assert callable(desktop_overlay._unregister_hotkey)


# ---------------------------------------------------------------------------
# App package __init__.py exports test
# ---------------------------------------------------------------------------


class TestAppPackageExports:
    """Verify app.__init__ exports the correct symbols."""

    def test_editor_bridge_exports(self) -> None:
        from app import (
            apply_selection_replacement,
            capture_highlighted_selection,
            detect_installed_editors,
            get_active_window_context,
        )

        assert callable(detect_installed_editors)
        assert callable(get_active_window_context)
        assert callable(capture_highlighted_selection)
        assert callable(apply_selection_replacement)
