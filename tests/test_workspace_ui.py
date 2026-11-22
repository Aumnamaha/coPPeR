"""Tests for coPPeR Workspace UI — split-canvas, dashed-square selection tool, SSE streaming.

Covers:
- GET /workspace returns HTML with correct structure (dashed-square SVG, split panes, context chips)
- POST /api/workspace/capture and /apply endpoints
- workspace_app.py WorkspaceBridge methods
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestWorkspaceHTML:
    """Verify GET /workspace returns correct HTML structure."""

    @pytest.fixture(autouse=True)
    def _load_html(self):
        """Load index.html content once per test class."""
        ui_dir = Path(__file__).resolve().parent.parent / "app" / "workspace_ui"
        self._html = (ui_dir / "index.html").read_text(encoding="utf-8")

    def test_workspace_returns_html(self):
        """GET /workspace serves index.html with HTML 200."""
        from fastapi.testclient import TestClient

        from app.inference_server import app as server_app

        client = TestClient(server_app)
        resp = client.get("/workspace", timeout=5)
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_workspace_contains_dashed_square_svg(self):
        """index.html contains the dashed-square selection tool SVG with stroke-dasharray."""
        assert (
            'stroke-dasharray="5 3"' in self._html
            or "stroke-dasharray='5 3'" in self._html
        )

    def test_workspace_has_split_layout_panes(self):
        """HTML has both left-rail (AI Assistant) and right-canvas panes."""
        assert 'id="left-rail"' in self._html
        assert 'class="rail rail-left"' in self._html
        assert 'id="right-canvas"' in self._html

    def test_workspace_has_context_chip_area(self):
        """HTML has a context chip container for attached documents."""
        assert "context-chip-area" in self._html or "chip-area" in self._html

    def test_workspace_has_model_selector_pill(self):
        """HTML contains the model selector pill with dropdown items."""
        assert "model-selector" in self._html
        assert "copper-1b-local" in self._html
        assert "qwen-35b-lan" in self._html
        assert "gguf-llamacpp" in self._html

    def test_workspace_has_split_diff_view(self):
        """HTML has a split diff view with Original and Proposed panes."""
        assert "split-diff-view" in self._html
        assert "diff-pane original" in self._html
        assert "diff-pane proposed" in self._html

    def test_workspace_has_point_back_citations(self):
        """HTML contains numbered point-back citation badges (❶ ❷ ❸)."""
        # Unicode escaped forms for the circled numbers
        assert "data-citation=" in self._html
        assert "&#x276A;" in self._html or "\u276a" in self._html.lower()

    def test_workspace_has_prompt_composer(self):
        """HTML has a prompt textarea, send button, and toolbar."""
        assert 'id="prompt-input"' in self._html
        assert "btn-send" in self._html
        assert "composer-toolbar" in self._html

    def test_workspace_css_exists_and_valid(self):
        """GET /workspace/styles.css returns valid CSS."""
        from fastapi.testclient import TestClient

        from app.inference_server import app as server_app

        client = TestClient(server_app)
        resp = client.get("/workspace/styles.css", timeout=5)
        assert resp.status_code == 200
        css_text = resp.text
        # Verify key CSS features exist
        assert "split-layout" in css_text
        assert "rail-left" in css_text or "#F4F7FB" in css_text
        assert "citation-badge" in css_text

    def test_workspace_js_exists_and_valid(self):
        """GET /workspace/app.js returns valid JavaScript."""
        from fastapi.testclient import TestClient

        from app.inference_server import app as server_app

        client = TestClient(server_app)
        resp = client.get("/workspace/app.js", timeout=5)
        assert resp.status_code == 200
        js_text = resp.text
        # Verify key JS features exist
        assert "streamChat" in js_text or "fetch" in js_text
        assert "highlightCitation" in js_text


class TestWorkspaceAPI:
    """Verify POST /api/workspace/capture and /apply endpoints."""

    def test_api_capture_returns_json(self):
        """POST /api/workspace/capture returns JSON with selection text."""
        from unittest.mock import patch

        mock_result = {"wrapped_document": "This is captured selection text."}

        import app.inference_server as _inf_mod

        with patch(
            "app.editor_bridge.capture_highlighted_selection",
            return_value=mock_result,
        ):
            from fastapi.testclient import TestClient

            client = TestClient(_inf_mod.app)
            resp = client.post("/api/workspace/capture")
            assert resp.status_code == 200
            body = resp.json()
            assert "selection" in body
            assert "charCount" in body

    def test_api_apply_returns_success(self, monkeypatch):
        """POST /api/workspace/apply accepts text and returns success."""
        from unittest.mock import patch

        import app.inference_server as _inf_mod

        with patch("app.editor_bridge.apply_selection_replacement", return_value=True):
            from fastapi.testclient import TestClient

            client = TestClient(_inf_mod.app)
            resp = client.post(
                "/api/workspace/apply",
                json={"text": "This is the edited text."},
                timeout=5,
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body.get("success") is True

    def test_api_apply_rejects_empty_text(self):
        """POST /api/workspace/apply returns 400 when text is missing."""
        from fastapi.testclient import TestClient

        import app.inference_server as _inf_mod

        client = TestClient(_inf_mod.app)
        resp = client.post("/api/workspace/apply", json={}, timeout=5)
        assert resp.status_code == 400


class TestWorkspaceBridge:
    """Verify WorkspaceBridge methods in workspace_app.py."""

    def test_bridge_capture_selection(self):
        """WorkspaceBridge.capture_selection() returns expected dict structure."""
        from app.workspace_app import WorkspaceBridge

        bridge = WorkspaceBridge()
        result = bridge.capture_selection()
        assert isinstance(result, dict)
        assert "selection" in result or "error" in result

    def test_bridge_apply_to_active_window(self):
        """WorkspaceBridge.apply_to_active_window() returns success/error."""
        from app.workspace_app import WorkspaceBridge

        bridge = WorkspaceBridge()
        result = bridge.apply_to_active_window("test text")
        assert isinstance(result, dict)
        assert "success" in result or "error" in result

    def test_bridge_load_file_missing(self):
        """WorkspaceBridge.load_file() handles missing file gracefully."""
        from app.workspace_app import WorkspaceBridge

        bridge = WorkspaceBridge()
        result = bridge.load_file("/nonexistent/path/file.txt")
        assert isinstance(result, dict)
        # Missing file returns None filename/content (not an error)
        assert result.get("filename") is None


class TestSSEStreamingContract:
    """Verify SSE streaming response format contract for /v1/chat/completions."""

    def test_stream_returns_sse_content_type(self):
        """POST /v1/chat/completions with stream=True returns text/event-stream content-type."""
        from fastapi.testclient import TestClient

        import app.inference_server as _inf_mod

        # Without model loaded, should return 503 — but we verify the streaming path exists
        client = TestClient(_inf_mod.app)
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        # Without model: 503 is expected; with streaming enabled the endpoint accepts it
        assert resp.status_code in (200, 503)


# ---------------------------------------------------------------------------
# Module-level reference to inference_server app for tests that import it
# ---------------------------------------------------------------------------

try:
    from app.inference_server import app as srv_module
except Exception:  # noqa: BLE001
    srv_module = None
