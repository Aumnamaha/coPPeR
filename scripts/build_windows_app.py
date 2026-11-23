#!/usr/bin/env python3
r"""Packaging script for coPPeR v2 Windows desktop application.

Supports two modes:
  --check-only   Verify all assets, entrypoints, and specs are present (exit 0 on success).
  --build        Run PyInstaller to produce dist/coPPeR/coPPeR.exe.

Usage:
    python scripts/build_windows_app.py --check-only
    python scripts/build_windows_app.py --build
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def check_assets(root: Path) -> list[str]:
    """Verify all required files exist."""
    needed = [
        "app/inference_server.py",
        "app/workspace_app.py",
        "app/model_router.py",
        "app/artifact_store.py",
        "app/agent_runner.py",
        "app/editor_bridge.py",
        "src/model.py",
        "src/data_schemas.py",
        "src/tokenizer_wrapper.py",
        "LICENSE",
        "README.md",
    ]
    ui_dir = root / "app" / "workspace_ui"
    if not (ui_dir / "index.html").exists():
        needed.append("app/workspace_ui/index.html")
    results: list[str] = []
    for rel in needed:
        p = root / rel
        if not p.exists():
            results.append(rel)
    gguf = root / "exports" / "copper-1b-q8_0.gguf"
    if not gguf.exists():
        print(
            "[WARN] exports/copper-1b-q8_0.gguf not found (Arc A750 bundle will skip SHA256)."
        )
    return results


def check_pyinstaller_available() -> bool:
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller  # noqa: F401

        return True
    except ImportError:
        return False


def cmd_check_only(root: Path) -> int:
    """Run all verification checks without building."""
    print("[coPPeR] Running --check-only verification...")
    missing = check_assets(root)
    if missing:
        msg = "[FAIL] Missing files: " + ", ".join(missing)
        print(msg)
        return 1
    print("[OK] All required source files present.")
    ui_dir = root / "app" / "workspace_ui"
    for asset in ["index.html", "styles.css", "app.js"]:
        if not (ui_dir / asset).exists():
            print("[FAIL] Missing UI asset: app/workspace_ui/" + asset)
            return 1
    print("[OK] Workspace UI assets present.")
    has_pyi = check_pyinstaller_available()
    if has_pyi:
        import PyInstaller

        print(
            "[OK] PyInstaller "
            + str(PyInstaller.__version__)
            + " available for --build mode."
        )
    else:
        print(
            "[WARN] PyInstaller not installed. Run pip install pyinstaller in build/.venv to enable --build."
        )
    inf_server = root / "app" / "inference_server.py"
    content = inf_server.read_text(encoding="utf-8")
    required_endpoints = ["/api/models", "/api/models/select", "/api/models/arc-bundle"]
    for ep in required_endpoints:
        target1 = '@app.get("' + ep
        target2 = '@app.post("' + ep
        if target1 not in content and target2 not in content:
            print("[FAIL] Missing endpoint " + ep + " in inference_server.py")
            return 1
    bat_path = root / "Launch_coPPeR.bat"
    if not bat_path.exists():
        print("[FAIL] Launch_coPPeR.bat not found")
        return 1
    print("[OK] Launch_coPPeR.bat present.")
    print()
    print(
        "[SUCCESS] All checks passed. Ready for --build mode if PyInstaller is installed."
    )
    return 0


def cmd_build(root: Path) -> int:
    """Run PyInstaller to produce coPPeR.exe."""
    import importlib

    try:
        importlib.import_module("PyInstaller")
    except ImportError:
        print("[FAIL] PyInstaller not installed.")
        return 1
    result = cmd_check_only(root)
    if result != 0:
        print("[FAIL] Check-only failed. Fix issues before building.")
        return result

    dist_dir = root / "dist" / "coPPeR"
    work_dir = root / "build" / "pyinstaller"
    spec_path = work_dir / "coPPeR.spec"
    app_py = str(root / "app" / "workspace_app.py")
    ui_dir = str(root / "app" / "workspace_ui")
    src_dir = str(root / "src")

    # Write PyInstaller spec file — GUI mode (no console window)
    dist_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(str(spec_path), "w", encoding="utf-8", newline="\n") as sf:
        # Hidden imports: all app modules + third-party deps
        hidden = [
            "app.inference_server",
            "app.artifact_store",
            "app.agent_runner",
            "app.model_router",
            "app.editor_bridge",
            "src.model",
            "src.data_schemas",
            "src.tokenizer_wrapper",
            "fastapi",
            "uvicorn",
            "pydantic",
            "torch",
            "safetensors",
            "tokenizers",
            "webview",
        ]
        sf.write("a = Analysis(\n")
        sf.write("    [" + repr(app_py) + "],\n")
        sf.write("    pathex=[" + repr(str(root)) + "],\n")
        sf.write("    hiddenimports=[" + ", ".join(repr(h) for h in hidden) + "],\n")
        sf.write("    binaries=[],\n")
        # Bundle workspace_ui and src as datas
        sf.write("    datas=[\n")
        sf.write("        (" + repr(ui_dir) + ", " + repr("app/workspace_ui") + "),\n")
        sf.write("        (" + repr(src_dir) + ", " + repr("src") + "),\n")
        sf.write("    ],\n")
        sf.write(")\n\npyz = PYZ(a.pure)\n\nexe = EXE(\n")
        sf.write("    pyz, a.scripts, [],\n")
        # GUI app: console=False (no separate console window)
        sf.write("    name='coPPeR', console=False, strip=True, upx=True,\n")
        sf.write("    disable_windowed_traceback=False, target_arch=None,\n")
        sf.write("    codesign_identity=None, entitlements_file=None,\n")
        sf.write(")\n")

    print("[INFO] Generated PyInstaller spec: " + str(spec_path))
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec_path),
        "--clean",
        "--noconfirm",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
    ]
    print("[INFO] Running: " + " ".join(cmd))
    result = subprocess.run(cmd, check=False, cwd=str(root))
    if result.returncode != 0:
        print("[FAIL] PyInstaller exited with code " + str(result.returncode))
        return result.returncode
    exe_path = dist_dir / "coPPeR.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(
            "\n[SUCCESS] Built coPPeR.exe ("
            + str(round(size_mb, 1))
            + " MB) at "
            + str(exe_path)
        )
    else:
        print("[WARN] PyInstaller succeeded but coPPeR.exe not found in dist/")
    return 0


def main() -> int:
    root = _root()
    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print("Usage:")
        print("  python scripts/build_windows_app.py --check-only   # Verify assets")
        print(
            "  python scripts/build_windows_app.py --build         # Build coPPeR.exe"
        )
        return 0
    mode = sys.argv[1]
    if mode == "--check-only":
        return cmd_check_only(root)
    elif mode == "--build":
        return cmd_build(root)
    else:
        print("[ERROR] Unknown mode: " + mode)
        print("Use --check-only or --build")
        return 1


if __name__ == "__main__":
    sys.exit(main())
