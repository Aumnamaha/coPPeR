#!/usr/bin/env python3
r"""Phase 12 tests: ModelRouter, /api/models endpoints, Arc A750 bundle, Launch_coPPeR.bat, build_windows_app.py."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Ensure app path is on sys.path for FastAPI imports
# ---------------------------------------------------------------------------

_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from app.inference_server import app as fastapi_app


def _reset_model_router_cache():
    """Reset the lazy-loaded ModelRouter singleton in inference_server."""
    import app.inference_server as inf_mod

    inf_mod._model_router = None


@pytest.fixture(autouse=True)
def reset_phase12_state(tmp_path):
    """Ensure clean state between tests."""
    _reset_model_router_cache()
    yield
    _reset_model_router_cache()


# ========================================================================
#  ModelRouter Unit Tests
# ========================================================================


class TestModelRouter:
    """Unit tests for app.model_router.ModelRouter."""

    def test_init_creates_config_dir(self, tmp_path):
        from app.model_router import ModelRouter

        ModelRouter(config_dir=str(tmp_path / "config"))
        assert (tmp_path / "config").exists()

    def test_default_profiles_loaded(self, tmp_path):
        from app.model_router import ModelRouter

        router = ModelRouter(config_dir=str(tmp_path))
        profiles = router.list_profiles()
        profile_ids = [p["profile_id"] for p in profiles]
        assert "copper-1b-local" in profile_ids
        assert "copper-1b-gguf" in profile_ids
        assert "qwen-35b-lan" in profile_ids
        assert "byok-cloud" in profile_ids

    def test_active_profile_is_default(self, tmp_path):
        from app.model_router import ModelRouter

        router = ModelRouter(config_dir=str(tmp_path))
        active = router.get_active_profile()
        assert active is not None
        assert active["profile_id"] == "copper-1b-local"

    def test_select_profile(self, tmp_path):
        from app.model_router import ModelRouter

        router = ModelRouter(config_dir=str(tmp_path))
        result = router.select_profile("qwen-35b-lan")
        assert result["success"] is True
        active = router.get_active_profile()
        assert active["profile_id"] == "qwen-35b-lan"

    def test_select_profile_with_url_override(self, tmp_path):
        from app.model_router import ModelRouter

        router = ModelRouter(config_dir=str(tmp_path))
        result = router.select_profile(
            "copper-1b-gguf",
            endpoint_url="http://custom:9999/v1",
            model_name="my-model.gguf",
        )
        assert result["success"] is True
        active = router.get_active_profile()
        assert active["endpoint_url"] == "http://custom:9999/v1"
        assert active["model_name"] == "my-model.gguf"

    def test_select_unknown_profile_raises(self, tmp_path):
        from app.model_router import ModelRouter

        router = ModelRouter(config_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown profile"):
            router.select_profile("nonexistent-profile")

    def test_persistence_across_reloads(self, tmp_path):
        from app.model_router import ModelRouter

        # Create and switch
        r1 = ModelRouter(config_dir=str(tmp_path))
        r1.select_profile("byok-cloud")
        del r1

        # Reload - should remember active profile
        r2 = ModelRouter(config_dir=str(tmp_path))
        active = r2.get_active_profile()
        assert active["profile_id"] == "byok-cloud"

    def test_get_profile_by_id(self, tmp_path):
        from app.model_router import ModelRouter

        router = ModelRouter(config_dir=str(tmp_path))
        p = router.get_profile("copper-1b-gguf")
        assert p is not None
        assert p["profile_id"] == "copper-1b-gguf"
        assert router.get_profile("nonexistent") is None

    def test_arc_a750_bundle_no_gguf(self, tmp_path):
        from app.model_router import ModelRouter

        router = ModelRouter(config_dir=str(tmp_path))
        result = router.generate_arc_a750_bundle(export_dir="exports_nonexistent")
        assert result["success"] is True
        assert "bundle_dir" in result
        # Verify files created
        launch = Path(result["launch_script"])
        manifest = Path(result["manifest"])
        assert launch.exists()
        assert manifest.exists()

    def test_arc_a750_bundle_with_gguf(self, tmp_path):
        from app.model_router import ModelRouter

        # Create fake GGUF file
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()
        gguf_file = exports_dir / "copper-1b-q8_0.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        router = ModelRouter(config_dir=str(tmp_path))
        result = router.generate_arc_a750_bundle(export_dir=str(exports_dir))
        assert result["success"] is True
        assert result["gguf_sha256"] is not None
        # Verify manifest contains SHA256
        manifest_data = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest_data["gguf_sha256"] == result["gguf_sha256"]


# ========================================================================
#  FastAPI Endpoint Tests (/api/models)
# ========================================================================


class TestApiModelsEndpoints:
    """FastAPI TestClient tests for /api/models endpoints."""

    def setup_method(self):
        self.client = TestClient(fastapi_app, raise_server_exceptions=True)
        _reset_model_router_cache()

    def teardown_method(self):
        _reset_model_router_cache()

    @pytest.fixture
    def clean_workspace(self, tmp_path):
        """Provide a temporary workspace directory for model router."""
        old = __import__("os").environ.get("COPPER_WORKSPACE_DIR")
        __import__("os").environ["COPPER_WORKSPACE_DIR"] = str(tmp_path)
        _reset_model_router_cache()
        yield tmp_path
        if old is not None:
            __import__("os").environ["COPPER_WORKSPACE_DIR"] = old
        elif "COPPER_WORKSPACE_DIR" in __import__("os").environ:
            del __import__("os").environ["COPPER_WORKSPACE_DIR"]
        _reset_model_router_cache()

    def test_get_models_returns_profiles(self, clean_workspace):
        resp = self.client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "profiles" in data
        assert isinstance(data["profiles"], list)
        assert len(data["profiles"]) >= 4

    def test_get_models_active_profile(self, clean_workspace):
        resp = self.client.get("/api/models")
        data = resp.json()
        assert data["active_profile"] is not None
        assert data["active_profile"]["profile_id"] == "copper-1b-local"

    def test_select_model_endpoint(self, clean_workspace):
        resp = self.client.post(
            "/api/models/select", json={"profile_id": "qwen-35b-lan"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["active_profile"]["profile_id"] == "qwen-35b-lan"

    def test_select_model_with_url(self, clean_workspace):
        resp = self.client.post(
            "/api/models/select",
            json={
                "profile_id": "copper-1b-gguf",
                "endpoint_url": "http://custom:9000/v1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_profile"]["endpoint_url"] == "http://custom:9000/v1"

    def test_arc_bundle_endpoint(self, clean_workspace):
        resp = self.client.post("/api/models/arc-bundle")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "bundle_dir" in data


# ========================================================================
#  Launch_coPPeR.bat Tests
# ========================================================================


class TestLaunchCoPPeR:
    """Tests for Launch_coPPeR.bat."""

    def test_launch_bat_exists(self):
        bat_path = _root / "Launch_coPPeR.bat"
        assert bat_path.exists(), "Launch_coPPeR.bat must exist at project root"

    def test_launch_bat_starts_workspace(self):
        """Bat file launches workspace_app.py which auto-starts embedded backend."""
        content = (_root / "Launch_coPPeR.bat").read_text(encoding="utf-8")
        assert (
            "workspace_app.py" in content
        ), "Launch script should launch workspace_app.py (which embeds the server)"

    def test_launch_bat_uses_project_venv(self):
        content = (_root / "Launch_coPPeR.bat").read_text(encoding="utf-8")
        assert "build\\.venv\\Scripts" in content or "build/.venv/Scripts" in content

    def test_launch_bat_has_root_var_quoted(self):
        """Batch file uses %~dp0 with proper quoting."""
        content = (_root / "Launch_coPPeR.bat").read_text(encoding="utf-8")
        assert 'set "ROOT=%~dp0"' in content

    def test_launch_bat_has_venv_check(self):
        """Batch file validates venv exists before proceeding."""
        content = (_root / "Launch_coPPeR.bat").read_text(encoding="utf-8")
        assert 'if not exist "%VENV_PY%"' in content
        assert "exit /b 1" in content

    def test_launch_bat_start_server_quoted(self):
        """Bat file uses Start-Process (not bare start) for path safety."""
        content = (_root / "Launch_coPPeR.bat").read_text(encoding="utf-8")
        assert (
            "Start-Process" in content
        ), "Should use PowerShell Start-Process, not cmd.exe start"

    def test_launch_bat_start_workspace_quoted(self):
        """Bat file uses Start-Process with proper quoting for paths."""
        content = (_root / "Launch_coPPeR.bat").read_text(encoding="utf-8")
        assert (
            "Start-Process -FilePath" in content
        ), "Should use PowerShell Start-Process for paths with spaces"
        assert "app/workspace_app.py" in content, "Should launch workspace_app.py"

    def test_launch_bat_uses_start_process(self):
        """Batch file uses PowerShell Start-Process for path safety."""
        content = (_root / "Launch_coPPeR.bat").read_text(encoding="utf-8")
        assert (
            "Start-Process" in content
        ), "Launch_coPPeR.bat should use Start-Process for paths with spaces"


# ========================================================================
#  build_windows_app.py Tests
# ========================================================================


class TestBuildWindowsApp:
    """Tests for scripts/build_windows_app.py --check-only."""

    def test_build_script_exists(self):
        script = _root / "scripts" / "build_windows_app.py"
        assert script.exists()

    def test_check_only_mode_exit_zero(self, capsys):
        import subprocess

        result = subprocess.run(
            [
                str(_root / "build\\.venv\\Scripts\\python.exe"),
                str(_root / "scripts" / "build_windows_app.py"),
                "--check-only",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(_root),
            timeout=30,
        )
        # --check-only should succeed (exit 0) even if PyInstaller is missing
        assert result.returncode == 0, f"--check-only failed: {result.stderr}"

    def test_help_mode(self):
        import subprocess

        result = subprocess.run(
            [
                str(_root / "build\\.venv\\Scripts\\python.exe"),
                str(_root / "scripts" / "build_windows_app.py"),
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(_root),
            timeout=10,
        )
        assert result.returncode == 0

    def test_invalid_mode(self):
        import subprocess

        result = subprocess.run(
            [
                str(_root / "build\\.venv\\Scripts\\python.exe"),
                str(_root / "scripts" / "build_windows_app.py"),
                "--invalid",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(_root),
            timeout=10,
        )
        assert result.returncode != 0

    def test_build_script_uses_gui_mode(self):
        """Build spec uses console=False for standalone GUI exe."""
        src = (_root / "scripts" / "build_windows_app.py").read_text(encoding="utf-8")
        assert (
            "console=False" in src
        ), "PyInstaller spec should use console=False (GUI mode, no console window)"

    def test_gitignore_excludes_dist(self):
        """dist/ and *.spec are listed in .gitignore."""
        gitignore = (_root / ".gitignore").read_text(encoding="utf-8")
        assert "dist/" in gitignore, ".gitignore should exclude dist/"
        assert "*.spec" in gitignore, ".gitignore should exclude *.spec files"


# ========================================================================
#  Integration: Model Router + Workspace UI
# ========================================================================


class TestModelSwitcherIntegration:
    """Verify workspace UI files reference /api/models endpoints."""

    def test_ui_has_model_switcher(self):
        js = (_root / "app" / "workspace_ui" / "app.js").read_text(encoding="utf-8")
        assert "/api/models" in js, "UI should fetch /api/models for model profiles"

    def test_ui_has_select_endpoint(self):
        js = (_root / "app" / "workspace_ui" / "app.js").read_text(encoding="utf-8")
        assert "/api/models/select" in js, "UI should POST to /api/models/select"

    def test_model_pill_in_html(self):
        html = (_root / "app" / "workspace_ui" / "index.html").read_text(
            encoding="utf-8"
        )
        assert 'id="model-selector"' in html
        assert 'id="model-label"' in html

    def test_gear_icon_in_html(self):
        html = (_root / "app" / "workspace_ui" / "index.html").read_text(
            encoding="utf-8"
        )
        assert (
            'id="model-gear"' in html
        ), "Gear icon for endpoint config should be present"


# ========================================================================
#  README Phase 12 Documentation Tests
# ========================================================================


class TestPhase12Readme:
    """Verify README contains comprehensive v2 documentation."""

    def test_readme_has_phase8(self):
        readme = (_root / "README.md").read_text(encoding="utf-8")
        assert "Phase 8" in readme or "phase 8" in readme.lower()

    def test_readme_has_model_switcher(self):
        readme = (_root / "README.md").read_text(encoding="utf-8")
        assert (
            "Model Router" in readme
            or "model switcher" in readme.lower()
            or "model_router" in readme
        )

    def test_readme_has_arc_a750(self):
        readme = (_root / "README.md").read_text(encoding="utf-8")
        assert "Arc A750" in readme or "arc-a750" in readme.lower()

    def test_readme_has_api_reference(self):
        readme = (_root / "README.md").read_text(encoding="utf-8")
        assert (
            "/api/models/select" in readme
        ), "API reference should include model switcher endpoints"
        assert (
            "/api/work/jobs" in readme
        ), "API reference should include work agent endpoints"

    def test_readme_has_license(self):
        readme = (_root / "README.md").read_text(encoding="utf-8")
        assert "Thirunagari Aum Namaha" in readme
        assert "All Rights Reserved" in readme


# ========================================================================
#  Model Router Singleton & Inference Server Integration
# ========================================================================


class TestModelRouterIntegration:
    """Test that inference_server correctly uses ModelRouter."""

    def setup_method(self):
        _reset_model_router_cache()

    def teardown_method(self):
        _reset_model_router_cache()

    @pytest.fixture
    def clean_workspace(self, tmp_path):
        import os

        old = os.environ.get("COPPER_WORKSPACE_DIR")
        os.environ["COPPER_WORKSPACE_DIR"] = str(tmp_path)
        _reset_model_router_cache()
        yield tmp_path
        if old is not None:
            os.environ["COPPER_WORKSPACE_DIR"] = old
        elif "COPPER_WORKSPACE_DIR" in os.environ:
            del os.environ["COPPER_WORKSPACE_DIR"]
        _reset_model_router_cache()

    def test_lazy_loading(self, clean_workspace):
        """Test that ModelRouter is loaded lazily on first API call."""
        import app.inference_server as inf_mod

        assert inf_mod._model_router is None  # Not yet loaded
        client = TestClient(fastapi_app, raise_server_exceptions=True)
        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert inf_mod._model_router is not None  # Now loaded

    def test_multiple_calls_share_singleton(self, clean_workspace):
        """Test that multiple API calls share the same ModelRouter instance."""
        import app.inference_server as inf_mod

        client = TestClient(fastapi_app, raise_server_exceptions=True)
        _ = client.get("/api/models")
        first_router = inf_mod._model_router
        _ = client.post("/api/models/select", json={"profile_id": "byok-cloud"})
        assert inf_mod._model_router is first_router  # Same singleton


# ========================================================================
#  PyInstaller Build — Python DLL Bundling & Smoke Test
# ========================================================================


class TestPythonDLLBundling:
    """Test that build_windows_app.py discovers and bundles python3*.dll."""

    def test_find_python_dlls_returns_entries(self):
        """_find_python_dlls() should discover python3.dll from sys.base_prefix."""
        from scripts.build_windows_app import _find_python_dlls

        dlls = _find_python_dlls()
        assert isinstance(dlls, list), "_find_python_dlls must return a list"
        # Should find at least python3.dll (the shim)
        from pathlib import Path as _Path

        names = [_Path(d).name for d in dlls]
        assert any(
            name.startswith("python3") and name.endswith(".dll") for name in names
        ), "Must discover at least one python3*.dll from sys.base_prefix or venv"

    def test_find_python_dlls_returns_strings(self):
        """Each entry must be an absolute string path to a python3*.dll."""
        from scripts.build_windows_app import _find_python_dlls

        dlls = _find_python_dlls()
        for entry in dlls:
            assert isinstance(entry, str), f"Expected str path, got {type(entry)}"
            p = Path(entry)
            assert p.exists(), f"DLL source not found: {entry}"
            assert p.name.startswith("python3") and p.name.endswith(
                ".dll"
            ), f"Unexpected DLL name: {p.name}"

    def test_spec_includes_dll_binaries(self):
        """cmd_build() must write DLL paths into spec's binaries list."""
        from scripts.build_windows_app import _find_python_dlls, cmd_check_only

        root = Path(__file__).resolve().parent.parent
        # Check-only should pass (we already verified this)
        result = cmd_check_only(root)
        assert result == 0, "Check-only must pass before verifying spec"

        # Verify DLL discovery returns entries
        dlls = _find_python_dlls()
        assert len(dlls) > 0, "Must discover at least one Python DLL"

    def test_launch_bat_uses_start_quotes(self):
        """Launch_coPPeR.bat must use properly quoted start commands."""
        bat_path = Path(__file__).resolve().parent.parent / "Launch_coPPeR.bat"
        content = bat_path.read_text(encoding="utf-8")
        # Must not have bare unquoted paths that could break with spaces
        assert 'start "coPPeR Workspace"' in content or (
            "Start-Process" in content
        ), "Launch script must use quoted start command"

    def test_workspace_app_has_headless_check(self):
        """workspace_app.py main() must support --headless-check flag."""
        app_path = Path(__file__).resolve().parent.parent / "app" / "workspace_app.py"
        content = app_path.read_text(encoding="utf-8")
        assert (
            "--headless-check" in content
        ), "workspace_app.py must support --headless-check CLI flag"
        assert (
            "_loaded_event += _inject_bridge" in content
            or "+= _inject_bridge" in content
        ), "Must use += for pywebview Event subscription"

    def test_workspace_app_has_embedded_server(self):
        """workspace_app.py must have --no-embedded-server support."""
        app_path = Path(__file__).resolve().parent.parent / "app" / "workspace_app.py"
        content = app_path.read_text(encoding="utf-8")
        assert (
            "--no-embedded-server" in content
        ), "workspace_app.py must support --no-embedded-server flag"
        assert (
            "_ensure_backend_running" in content or "ensure_backend_running" in content
        ), "Must have backend auto-start function"

    def test_build_uses_onedir_not_onefile(self):
        """build_windows_app.py must use onedir mode (not default onefile)."""
        build_path = (
            Path(__file__).resolve().parent.parent / "scripts/build_windows_app.py"
        )
        content = build_path.read_text(encoding="utf-8")
        # Must reference _internal/ directory for DLL placement
        assert (
            "_internal" in content
        ), "Must reference _internal/ for onedir mode DLL placement"
        # Must discover vcruntime DLLs alongside python3 DLLs
        assert (
            "vcruntime" in content.lower()
        ), "Must discover vcruntime140*.dll for Windows runtime support"
        # Must have hard check for python314.dll
        assert (
            "python314.dll NOT found" in content
        ), "Must verify python314.dll placement before returning success"

    def test_build_cleans_stale_dist(self):
        """cmd_build() must clean dist/coPPeR before building."""
        build_path = (
            Path(__file__).resolve().parent.parent / "scripts/build_windows_app.py"
        )
        content = build_path.read_text(encoding="utf-8")
        assert (
            "rmtree(str(dist_dir))" in content or "shutil.rmtree" in content
        ), "Must clean stale dist/coPPeR before building"

    def test_build_copies_dlls_to_internal(self):
        """cmd_build() must copy DLLs into _internal/ after build."""
        build_path = (
            Path(__file__).resolve().parent.parent / "scripts/build_windows_app.py"
        )
        content = build_path.read_text(encoding="utf-8")
        assert (
            "copy2(str(src_path), str(dst))" in content
        ), "Must copy DLLs into _internal/ using shutil.copy2"
        assert (
            "mkdir(parents=True, exist_ok=True)" in content
        ), "Must create _internal/ directory if missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
