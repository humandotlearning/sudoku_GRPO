"""Push the Sudoku OpenEnv package to Hugging Face Spaces."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "sudoku_env"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set. Add a Hugging Face write token before pushing.", file=sys.stderr)
        return 2

    user = whoami(token=token)["name"]
    repo_id = os.environ.get("SUDOKU_OPENENV_REPO", f"{user}/sudoku-openenv")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    cmd = [
        str(ROOT / ".venv" / "Scripts" / "openenv.exe"),
        "push",
        "--repo-id",
        repo_id,
    ]
    print(f"Pushing OpenEnv environment to {repo_id}")
    return subprocess.call(cmd, cwd=ENV_DIR)


if __name__ == "__main__":
    raise SystemExit(main())

