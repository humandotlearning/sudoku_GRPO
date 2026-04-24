import importlib
import sys
from pathlib import Path


def test_hf_space_top_level_server_import():
    env_root = Path(__file__).resolve().parents[1] / "sudoku_env"
    sys.path.insert(0, str(env_root))
    try:
        app_module = importlib.import_module("server.app")
        assert app_module.app.title == "OpenEnv Environment HTTP API"
    finally:
        sys.path.remove(str(env_root))

