#!/usr/bin/env python3
r"""Pluggable model profile router for coPPeR v2.

Manages inference profiles persisted to workspace_notes/model_profiles.json:
  - copper-1b-local   (PyTorch CUDA/CPU on checkpoints/copper_1b_final)
  - copper-1b-gguf    (GGUF via llama-server / LM Studio at http://127.0.0.1:8080/v1)
  - qwen-35b-lan      (LAN Qwen endpoint at http://192.168.56.1:1234/v1)
  - byok-cloud        (Configurable OpenAI-compatible cloud endpoint, BYOK key)

Exposes list_profiles(), get_active_profile(), select_profile(), and generate_arc_a750_bundle().
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_PROFILES = [
    {
        "profile_id": "copper-1b-local",
        "label": "coPPeR-1B · Local",
        "backend": "pytorch",
        "endpoint_url": "",
        "model_name": "copper-1b-final",
        "description": "PyTorch CUDA/CPU inference from local checkpoint.",
    },
    {
        "profile_id": "copper-1b-gguf",
        "label": "coPPeR-1B · GGUF / Arc A750",
        "backend": "llamacpp",
        "endpoint_url": "http://127.0.0.1:8080/v1",
        "model_name": "copper-1b-q8_0.gguf",
        "description": "GGUF via llama-server / LM Studio (Intel Arc A750).",
    },
    {
        "profile_id": "qwen-35b-lan",
        "label": "Qwen-35B · LAN",
        "backend": "lmstudio",
        "endpoint_url": "http://192.168.56.1:1234/v1",
        "model_name": "qwen-35b",
        "description": "LAN Qwen endpoint via LM Studio.",
    },
    {
        "profile_id": "byok-cloud",
        "label": "BYOK Cloud · OpenAI-Compatible",
        "backend": "openai-compatible",
        "endpoint_url": "",
        "model_name": "",
        "description": "Custom OpenAI-compatible cloud endpoint (set BYOK key via env).",
    },
]

DEFAULT_ACTIVE = "copper-1b-local"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ModelProfile:
    """Represents one inference profile configuration."""

    profile_id: str
    label: str
    backend: str
    endpoint_url: str = ""
    model_name: str = ""
    description: str = ""
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelRouterState:
    """Persisted state for the model router."""

    profiles: list[dict[str, Any]] = field(default_factory=list)
    active_profile_id: str = DEFAULT_ACTIVE
    updated_at: str = ""


# ---------------------------------------------------------------------------
# ModelRouter — Core engine
# ---------------------------------------------------------------------------


class ModelRouter:
    """Manages pluggable inference profiles with local persistence."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        if config_dir is None:
            from .artifact_store import DEFAULT_BASE_DIR

            config_dir = os.environ.get(
                "COPPER_WORKSPACE_DIR",
                str(Path(__file__).resolve().parent.parent / DEFAULT_BASE_DIR),
            )
        self.config_dir = Path(config_dir).resolve()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config_path = self.config_dir / "model_profiles.json"
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> ModelRouterState:
        """Load persisted state from disk, or initialize defaults."""
        if self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text())
                return ModelRouterState(
                    profiles=data.get("profiles", []),
                    active_profile_id=data.get("active_profile_id", DEFAULT_ACTIVE),
                    updated_at=datetime.now(UTC).isoformat(),
                )
            except (json.JSONDecodeError, KeyError):
                pass

        # Initialize with defaults
        state = ModelRouterState(
            profiles=[dict(p) for p in DEFAULT_PROFILES],
            active_profile_id=DEFAULT_ACTIVE,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._save_state(state)
        return state

    def _save_state(self, state: ModelRouterState | None = None) -> None:
        """Persist current state to disk."""
        if state is None:
            state = self._state
        data = {
            "profiles": state.profiles,
            "active_profile_id": state.active_profile_id,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Profile CRUD
    # ------------------------------------------------------------------

    def list_profiles(self) -> list[dict[str, Any]]:
        """Return all configured profiles with active flag."""
        return [
            {**p, "active": p.get("profile_id") == self._state.active_profile_id}
            for p in self._state.profiles
        ]

    def get_active_profile(self) -> dict[str, Any] | None:
        """Return the currently active profile."""
        for p in self._state.profiles:
            if p.get("profile_id") == self._state.active_profile_id:
                return {**p, "active": True}
        return None

    def select_profile(
        self,
        profile_id: str,
        endpoint_url: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Switch active profile and optionally override URL/model."""
        found = False
        for p in self._state.profiles:
            if p["profile_id"] == profile_id:
                p.setdefault("active", False)
                if endpoint_url is not None:
                    p["endpoint_url"] = endpoint_url
                if model_name is not None:
                    p["model_name"] = model_name
                self._state.active_profile_id = profile_id
                found = True
                break

        if not found:
            raise ValueError(f"Unknown profile: {profile_id}")

        # Deactivate all others
        for p in self._state.profiles:
            p["active"] = p.get("profile_id") == profile_id

        self._save_state()
        return {"success": True, "profile_id": profile_id}

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        """Return a single profile by ID."""
        for p in self._state.profiles:
            if p.get("profile_id") == profile_id:
                return {
                    **p,
                    "active": p.get("profile_id") == self._state.active_profile_id,
                }
        return None

    # ------------------------------------------------------------------
    # Arc A750 Bundle Generation
    # ------------------------------------------------------------------

    def generate_arc_a750_bundle(
        self, export_dir: str | Path = "exports"
    ) -> dict[str, Any]:
        """Generate Intel Arc A750 deployment bundle manifest.

        Creates arc_a750_launch.ps1 and manifest.json in workspace_notes/.
        Verifies SHA256 of exports/copper-1b-q8_0.gguf when present.
        """

        export_path = Path(export_dir) / "copper-1b-q8_0.gguf"
        gguf_hash: str | None = None

        if export_path.exists():
            sha256 = hashlib.sha256()
            with open(export_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            gguf_hash = sha256.hexdigest()

        bundle_dir = self.config_dir
        launch_script = (
            "#!/usr/bin/env pwsh\n"
            "# coPPeR Intel Arc A750 Launcher (auto-generated)\n"
            f"# GGUF: {export_path}\n"
            f'# SHA256: {gguf_hash or "N/A"}\n'
            "\n"
            'Write-Host "[coPPeR] Starting llama-server for Arc A750..." -ForegroundColor Cyan\n'
            '$env:COPPER_BACKEND = "llamacpp"\n'
            f'$env:LLAMA_CPP_URL = "http://127.0.0.1:8080/v1"\n'
            "\n"
            "# Start llama-server in background\n"
            f"start-process -WindowStyle Hidden python -ArgumentList "
            '"-m", "llama_cpp.server", "--model", "{export_path}", '
            '"--host", "127.0.0.1", "--port", "8080"\n'
            "\n"
            "# Wait for server to be ready\n"
            "$max_wait = 30\n"
            "$i = 0\n"
            "while ($i -lt $max_wait) {\n"
            '    try { Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -ErrorAction Stop; break }\n'
            "    catch { Start-Sleep -Milliseconds 500; $i++ }\n"
            "}\n"
            "\n"
            "# Launch coPPeR workspace\n"
            f'start-process python -ArgumentList "{Path(__file__).parent.parent / "workspace_app.py"}"\n'
            'Write-Host "[coPPeR] Workspace launched." -ForegroundColor Green\n'
        )

        manifest = {
            "bundle_name": "copper-1b-arc-a750",
            "gguf_path": str(export_path),
            "gguf_sha256": gguf_hash,
            "endpoint_url": "http://127.0.0.1:8080/v1",
            "launch_script": "arc_a750_launch.ps1",
            "generated_at": datetime.now(UTC).isoformat(),
        }

        launch_path = bundle_dir / "arc_a750_launch.ps1"
        manifest_path = bundle_dir / "manifest.json"

        with open(launch_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(launch_script)
        with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2)

        return {
            "success": True,
            "bundle_dir": str(bundle_dir),
            "launch_script": str(launch_path),
            "manifest": manifest_path,
            "gguf_sha256": gguf_hash,
        }


# ---------------------------------------------------------------------------
# Singleton accessor (for inference_server lazy loading)
# ---------------------------------------------------------------------------

_router: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    """Lazy-load the ModelRouter singleton."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
