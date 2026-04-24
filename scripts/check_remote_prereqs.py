"""Check local environment variables needed for deployment commands."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def check_command(name: str) -> bool:
    path = shutil.which(name)
    print(f"{name}: {'found at ' + path if path else 'missing'}")
    return path is not None


def check_modal() -> bool:
    path = shutil.which("modal")
    if path:
        print(f"modal: found at {path}")
        return True

    venv_modal = ROOT / ".venv" / "Scripts" / "modal.exe"
    if venv_modal.exists():
        print(f"modal: found at {venv_modal}")
        return True

    print("modal: missing")
    return False


def main() -> int:
    ok = True
    ok = check_command("docker") and ok
    ok = check_modal() and ok

    for key in ("HF_TOKEN", "TRACKIO_SPACE_ID", "TRACKIO_PROJECT"):
        present = bool(os.environ.get(key))
        print(f"{key}: {'set' if present else 'missing'}")
        ok = present and ok

    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        print(f"Docker daemon: {'running ' + result.stdout.strip() if result.returncode == 0 else 'not reachable'}")
        ok = result.returncode == 0 and ok
    except Exception as exc:
        print(f"Docker daemon: not reachable ({exc})")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
