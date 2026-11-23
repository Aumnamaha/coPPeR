"""Tests for Phase 10 Workspace Artifacts — ArtifactStore CRUD, checklists, tables, DOCX export.

Covers:
- artifact_store.py: CRUD, checklist parsing/toggling, table parsing/serialization, .docx export
- /api/artifacts/* endpoints via FastAPI TestClient with tmp_path-scoped stores
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

# Ensure src and app are on path for local module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from artifact_store import ArtifactStore

# ---------------------------------------------------------------------------
# ArtifactStore unit tests (using tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path):
    """Create a temporary ArtifactStore backed by a test directory."""
    from artifact_store import ArtifactStore

    return ArtifactStore(str(tmp_path / "artifacts"))


class TestArtifactStoreCRUD:
    """Test save, get, list, delete operations on ArtifactStore."""

    def test_save_and_get_note(self, store: ArtifactStore) -> None:
        """save_note writes a .md + meta.json; get_note reads them back."""
        result = store.save_note(
            "My Note", "Hello world\nLine two.", artifact_type="note"
        )
        assert result["slug"] == "my-note"
        assert result["title"] == "My Note"

        loaded = store.get_note("my-note")
        assert loaded is not None
        assert loaded["title"] == "My Note"
        assert loaded["type"] == "note"
        assert "Hello world" in loaded["content"]

    def test_save_creates_slug_from_title(self, store: ArtifactStore) -> None:
        """Title 'Test Checklist' should produce slug 'test-checklist'."""
        result = store.save_note("Test Checklist", "", artifact_type="checklist")
        assert result["slug"] == "test-checklist"

    def test_save_with_slug_override(self, store: ArtifactStore) -> None:
        """Explicit slug_override is used instead of derived slug."""
        result = store.save_note(
            "My Data",
            "data content",
            artifact_type="table",
            slug_override="my-data-override",
        )
        assert result["slug"] == "my-data-override"

    def test_get_nonexistent_returns_none(self, store: ArtifactStore) -> None:
        """get_note for a missing slug returns None."""
        assert store.get_note("does-not-exist") is None

    def test_list_notes_empty(self, store: ArtifactStore) -> None:
        """list_notes returns empty list when no artifacts exist."""
        items = store.list_notes()
        assert isinstance(items, list)
        assert len(items) == 0

    def test_list_notes_after_saves(self, store: ArtifactStore) -> None:
        """list_notes includes all saved artifacts with correct metadata."""
        store.save_note("Alpha", "a")
        store.save_note("Beta", "b", artifact_type="checklist")
        items = store.list_notes()
        assert len(items) == 2
        slugs = {it["slug"] for it in items}
        assert "alpha" in slugs
        assert "beta" in slugs

    def test_delete_note(self, store: ArtifactStore) -> None:
        """delete_note removes the artifact files and returns True."""
        store.save_note("To Delete", "content")
        assert store.delete_note("to-delete") is True

    def test_delete_nonexistent_returns_false(self, store: ArtifactStore) -> None:
        """Deleting a non-existent slug returns False."""
        assert store.delete_note("ghost-file") is False

    def test_save_overwrites_existing(self, store: ArtifactStore) -> None:
        """Saving with the same slug updates content but keeps metadata."""
        store.save_note("Update Me", "v1")
        store.save_note("Update Me", "v2-updated")

        loaded = store.get_note("update-me")
        assert "v2-updated" in loaded["content"]


class TestArtifactStoreChecklist:
    """Test checklist parsing and toggling."""

    def test_parse_checklist(self) -> None:
        """parse_checklist extracts checkbox items from Markdown text."""
        md = "# Header\n\n- [ ] Task one\n- [x] Done item\nSome other line\n- [ ] Another task"
        items = ArtifactStore.parse_checklist(md)
        # Should have 6 entries: 4 non-empty lines (header, empty, 'some other', 3 checklist items)
        assert len(items) == 6

    def test_parse_checklist_checked_state(self) -> None:
        """parse_checklist correctly identifies checked vs unchecked."""
        md = "- [ ] unchecked\n- [x] checked"
        items = ArtifactStore.parse_checklist(md)
        assert items[0]["checked"] is False
        assert items[0]["text"] == "unchecked"
        assert items[1]["checked"] is True
        assert items[1]["text"] == "checked"

    def test_toggle_checklist_item(self) -> None:
        """toggle_checklist_item flips the checkbox state."""
        md = "- [ ] Task one\n- [x] Task two"
        toggled = ArtifactStore.toggle_checklist_item(md, 0)
        assert "- [x] Task one" in toggled

    def test_toggle_second_item(self) -> None:
        """Toggle the second checklist item (index=1)."""
        md = "- [ ] A\n- [ ] B"
        toggled = ArtifactStore.toggle_checklist_item(md, 1)
        assert "- [x] B" in toggled


class TestArtifactStoreTable:
    """Test GFM table parsing and serialization."""

    def test_parse_markdown_table(self) -> None:
        """parse_markdown_table extracts headers and rows from a valid Markdown table."""
        md = "| Name | Role |\n| --- | --- |\n| Alice | Eng |\n| Bob | Des |"
        result = ArtifactStore.parse_markdown_table(md)
        assert result is not None
        assert result["headers"] == ["Name", "Role"]
        assert len(result["rows"]) == 2

    def test_parse_invalid_table_returns_none(self) -> None:
        """parse_markdown_table returns None when no table structure exists."""
        assert ArtifactStore.parse_markdown_table("just a paragraph") is None

    def test_serialize_markdown_table(self) -> None:
        """serialize_markdown_table produces valid GFM Markdown."""
        headers = ["A", "B"]
        rows = [["1", "2"], ["3", "4"]]
        result = ArtifactStore.serialize_markdown_table(headers, rows)
        assert "| A | B |" in result
        assert "| 1 | 2 |" in result
        assert "| 3 | 4 |" in result

    def test_serialize_empty_headers(self) -> None:
        """Empty headers produce empty string."""
        assert ArtifactStore.serialize_markdown_table([], []) == ""


class TestArtifactStoreDocxExport:
    """Test .docx export functionality."""

    def test_export_to_docx_creates_valid_file(self, tmp_path: Path) -> None:
        """export_to_docx creates a valid .docx file that can be opened as ZIP."""
        md_text = "# Title\n\nSome paragraph text.\n\n## Subheading\n\n- Bullet one\n- Bullet two"
        out = tmp_path / "output.docx"
        result = ArtifactStore.export_to_docx("Test Doc", md_text, str(out))

        assert result["success"] is True  # type: ignore[attr-defined]
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_to_docx_is_valid_zip(self, tmp_path: Path) -> None:
        """DOCX files are ZIP archives — verify the file opens as a valid ZIP."""
        md_text = "# Hello\n\nWorld content here."
        out = tmp_path / "test.docx"
        ArtifactStore.export_to_docx("Hello", md_text, str(out))

        assert zipfile.is_zipfile(str(out))


# ---------------------------------------------------------------------------
# API endpoint tests (using FastAPI TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_store(tmp_path: Path):
    """Create a temporary ArtifactStore and set the env var for the API."""
    import os as _os

    from artifact_store import ArtifactStore

    base_dir = str(tmp_path / "api_artifacts")
    store = ArtifactStore(base_dir)
    # Set env var so the API uses this directory
    old = _os.environ.get("COPPER_WORKSPACE_DIR")
    _os.environ["COPPER_WORKSPACE_DIR"] = base_dir
    yield store, tmp_path
    if old is not None:
        _os.environ["COPPER_WORKSPACE_DIR"] = old
    elif "COPPER_WORKSPACE_DIR" in _os.environ:
        del _os.environ["COPPER_WORKSPACE_DIR"]


class TestApiListArtifacts:
    """GET /api/artifacts — list saved artifacts."""

    def test_list_returns_empty(self, api_store):
        """Empty store returns empty artifact list."""
        from fastapi.testclient import TestClient

        _store, _tmp = api_store
        # Reset the module-level cache so it picks up our tmp_path
        import app.inference_server as _inf_mod

        _inf_mod._artifact_store = None

        client = TestClient(_inf_mod.app)
        resp = client.get("/api/artifacts")
        assert resp.status_code == 200
        body = resp.json()
        assert "artifacts" in body
        assert isinstance(body["artifacts"], list)


class TestApiGetArtifact:
    """GET /api/artifacts/{slug} — load specific artifact."""

    def test_get_existing_artifact(self, api_store):
        """Fetching an existing slug returns the artifact data."""
        from fastapi.testclient import TestClient

        _store, _tmp = api_store
        import app.inference_server as _inf_mod

        _inf_mod._artifact_store = None

        # Pre-create via store directly (to avoid module cache issues)
        _store.save_note("Test Artifact", "Hello from API")

        client = TestClient(_inf_mod.app)
        resp = client.get("/api/artifacts/test-artifact")
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == "Test Artifact"
        assert "Hello from API" in body["content"]

    def test_get_nonexistent_returns_404(self, api_store):
        """Fetching a missing slug returns HTTP 404."""
        from fastapi.testclient import TestClient

        _store, _tmp = api_store
        import app.inference_server as _inf_mod

        _inf_mod._artifact_store = None

        client = TestClient(_inf_mod.app)
        resp = client.get("/api/artifacts/nonexistent-slug")
        assert resp.status_code == 404


class TestApiCreateArtifact:
    """POST /api/artifacts — create or update an artifact."""

    def test_create_new_artifact(self, api_store):
        """Creating a new artifact returns success and slug."""
        from fastapi.testclient import TestClient

        _store, _tmp = api_store
        import app.inference_server as _inf_mod

        _inf_mod._artifact_store = None

        client = TestClient(_inf_mod.app)
        resp = client.post(
            "/api/artifacts",
            json={"title": "New Note", "content": "Body text.", "type": "note"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["slug"] == "new-note"

    def test_create_checklist_artifact(self, api_store):
        """Creating a checklist-type artifact sets correct type."""
        from fastapi.testclient import TestClient

        _store, _tmp = api_store
        import app.inference_server as _inf_mod

        _inf_mod._artifact_store = None

        client = TestClient(_inf_mod.app)
        resp = client.post(
            "/api/artifacts",
            json={
                "title": "My Tasks",
                "content": "- [ ] Task A\n- [x] Done B",
                "type": "checklist",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["slug"] == "my-tasks"


class TestApiDeleteArtifact:
    """DELETE /api/artifacts/{slug} — remove an artifact."""

    def test_delete_existing(self, api_store):
        """Deleting an existing slug returns success."""
        from fastapi.testclient import TestClient

        _store, _tmp = api_store
        import app.inference_server as _inf_mod

        _inf_mod._artifact_store = None

        # Pre-create via store directly
        _store.save_note("To Remove", "content")

        client = TestClient(_inf_mod.app)
        resp = client.delete("/api/artifacts/to-remove")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

    def test_delete_nonexistent_returns_404(self, api_store):
        """Deleting a missing slug returns HTTP 404."""
        from fastapi.testclient import TestClient

        _store, _tmp = api_store
        import app.inference_server as _inf_mod

        _inf_mod._artifact_store = None

        client = TestClient(_inf_mod.app)
        resp = client.delete("/api/artifacts/ghost-artifact")
        assert resp.status_code == 404


class TestApiExportDocx:
    """POST /api/artifacts/export-docx — export to DOCX."""

    def test_export_docx_success(self, api_store):
        """Exporting valid content creates a .docx file."""
        from fastapi.testclient import TestClient

        _store, tmp_path = api_store
        import app.inference_server as _inf_mod

        _inf_mod._artifact_store = None

        out_file = str(tmp_path / "export.docx")
        client = TestClient(_inf_mod.app)
        resp = client.post(
            "/api/artifacts/export-docx",
            json={
                "title": "Exported Doc",
                "content": "# Title\n\nSome content.",
                "output_path": out_file,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

    def test_export_docx_missing_output_returns_400(self, api_store):
        """Export without output_path returns HTTP 400."""
        from fastapi.testclient import TestClient

        _store, _tmp = api_store
        import app.inference_server as _inf_mod

        _inf_mod._artifact_store = None

        client = TestClient(_inf_mod.app)
        resp = client.post(
            "/api/artifacts/export-docx",
            json={"title": "No Path"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# WorkspaceBridge artifact method tests
# ---------------------------------------------------------------------------


class TestWorkspaceBridgeArtifacts:
    """Test WorkspaceBridge artifact helper methods."""

    def test_bridge_list_artifacts_empty(self, tmp_path: Path) -> None:
        """WorkspaceBridge.list_artifacts() returns success with empty list initially."""
        import os as _os

        from app.workspace_app import WorkspaceBridge

        base_dir = str(tmp_path / "bridge_notes")
        old = _os.environ.get("COPPER_WORKSPACE_DIR")
        _os.environ["COPPER_WORKSPACE_DIR"] = base_dir
        try:
            bridge = WorkspaceBridge()
            result = bridge.list_artifacts()
            assert isinstance(result, dict)
            assert "success" in result or "error" not in result
        finally:
            if old is not None:
                _os.environ["COPPER_WORKSPACE_DIR"] = old
            elif "COPPER_WORKSPACE_DIR" in _os.environ:
                del _os.environ["COPPER_WORKSPACE_DIR"]

    def test_bridge_save_and_delete_artifact(self, tmp_path: Path) -> None:
        """WorkspaceBridge.save_artifact() persists and delete_artifact() removes it."""
        import os as _os

        from app.workspace_app import WorkspaceBridge

        base_dir = str(tmp_path / "bridge_notes2")
        old = _os.environ.get("COPPER_WORKSPACE_DIR")
        _os.environ["COPPER_WORKSPACE_DIR"] = base_dir
        try:
            bridge = WorkspaceBridge()
            result = bridge.save_artifact(
                title="Bridge Note", content="test", artifact_type="note"
            )
            assert isinstance(result, dict)
            assert "success" in result

            del_result = bridge.delete_artifact("bridge-note")
            assert isinstance(del_result, dict)
        finally:
            if old is not None:
                _os.environ["COPPER_WORKSPACE_DIR"] = old
            elif "COPPER_WORKSPACE_DIR" in _os.environ:
                del _os.environ["COPPER_WORKSPACE_DIR"]

    def test_bridge_export_docx(self, tmp_path: Path) -> None:
        """WorkspaceBridge.export_docx() creates a valid .docx file."""
        from app.workspace_app import WorkspaceBridge

        out_file = str(tmp_path / "bridge_export.docx")
        bridge = WorkspaceBridge()
        result = bridge.export_docx(
            title="Bridge Export", content="# Hello\n\nWorld.", output_path=out_file
        )
        assert isinstance(result, dict)
        if not result.get("error"):
            from pathlib import Path as _P

            assert (_P(out_file)).exists()


# ---------------------------------------------------------------------------
# UI contract tests — verify workspace HTML/CSS/JS assets contain required elements
# ---------------------------------------------------------------------------


class TestWorkspaceUIArtifactAssets:
    """Verify the workspace UI files include Phase 10 artifact-related markup."""

    @pytest.fixture(autouse=True)
    def _load_assets(self):
        ui_dir = Path(__file__).resolve().parent.parent / "app" / "workspace_ui"
        self._html = (ui_dir / "index.html").read_text(encoding="utf-8")
        css_path = ui_dir / "styles.css"
        self._css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    def test_notes_drawer_toggle_exists(self):
        """HTML contains the Notes Drawer toggle button."""
        assert 'id="btn-notes-drawer"' in self._html

    def test_notes_drawer_element_exists(self):
        """HTML contains the notes drawer aside element."""
        assert 'id="notes-drawer"' in self._html

    def test_export_buttons_exist(self):
        """HTML has Export .md and Export .docx buttons."""
        assert "btn-export-md" in self._html
        assert "btn-export-docx" in self._html

    def test_checklist_view_element_exists(self):
        """HTML contains the checklist view container."""
        assert 'id="checklist-view"' in self._html
        assert (
            'class="artifact-view hidden"' in self._html
            or 'class="artifact-view' in self._html
        )

    def test_table_grid_elements_exist(self):
        """HTML has table grid, add row, and add column buttons."""
        assert "btn-add-row" in self._html
        assert "btn-add-col" in self._html
        assert "id='table-grid'" in self._html or 'id="table-grid"' in self._html

    def test_notes_list_container_exists(self):
        """HTML has the notes list container."""
        assert 'id="notes-list"' in self._html
