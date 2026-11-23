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


def _find_python_dlls() -> list[tuple[str, str]]:
    """Discover python3*.dll from sys.base_prefix and sys.executable parent.

    Returns list of (source_path, dest_dir) tuples for PyInstaller --add-binary.
    """
    import sys as _sys
    from pathlib import Path as _Path

    dlls: list[tuple[str, str]] = []
    for base in [_Path(_sys.base_prefix), _Path(_sys.executable).parent]:
        if not base.exists():
            continue
        for dll_path in sorted(base.glob("python3*.dll")):
            pair = (str(dll_path), ".")  # copy to _internal/ root
            if pair not in dlls:
                dlls.append(pair)
    return dlls


def cmd_build(root: Path) -> int:
    """Run PyInstaller to produce coPPeR.exe."""
    import importlib as _importlib

    try:
        _importlib.import_module("PyInstaller")
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

    # Discover Python base DLLs (python3.dll, python314.dll etc.)
    py_dlls = _find_python_dlls()
    print(
        "[INFO] Found Python DLLs to bundle: "
        + ", ".join(d[0].split(chr(92))[-1] for d in py_dlls)
        if py_dlls
        else "[WARN] No python3*.dll found"
    )

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
        # Add Python DLLs as binaries
        if py_dlls:
            dll_entries = [repr((src, dest)) for src, dest in py_dlls]
            sf.write("    binaries=[" + ", ".join(dll_entries) + "],\n")
        else:
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
    if not exe_path.exists():
        print("[WARN] PyInstaller succeeded but coPPeR.exe not found in dist/")
        return 1

    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(
        "\n[SUCCESS] Built coPPeR.exe ("
        + str(round(size_mb, 1))
        + " MB) at "
        + str(exe_path)
    )

    # Post-build: for console=True builds, verify DLLs in _internal/.
    # For console=False (GUI) builds, DLLs are embedded in PKG and extracted
    # at runtime by the runw.exe bootloader — no _internal/ folder is created.
    internal_dir = dist_dir / "_internal"
    if not internal_dir.exists():
        print("[INFO] GUI build (console=False) — Python DLLs are embedded in PKG.")
    else:
        for src_path, _dest in py_dlls:
            dll_name = Path(src_path).name
            dst = internal_dir / dll_name
            if not dst.exists():
                import shutil as _shutil

                _shutil.copy2(str(src_path), str(dst))
                print("[INFO] Copied " + dll_name + " -> _internal/")

    # Smoke test: run coPPeR.exe --headless-check --no-embedded-server
    smoke_cmd = [
        str(exe_path),
        "--headless-check",
        "--no-embedded-server",
    ]
    print("\n[INFO] Running frozen smoke test: " + " ".join(smoke_cmd))
    smoke_result = subprocess.run(
        smoke_cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(dist_dir),
    )
    if smoke_result.returncode == 0:
        print("[SUCCESS] Frozen smoke test PASSED.")
        if smoke_result.stdout.strip():
            for line in smoke_result.stdout.strip().splitlines():
                print("  " + line)
    else:
        print(
            "[WARN] Smoke test returned code "
            + str(smoke_result.returncode)
            + " (may require GUI runtime):"
        )
        if smoke_result.stdout.strip():
            for line in smoke_result.stdout.strip().splitlines():
                print("  " + line)
        if smoke_result.stderr.strip():
            for line in smoke_result.stderr.strip().splitlines()[:5]:
                print("  ERR: " + line)
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
