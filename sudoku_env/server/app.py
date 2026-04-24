"""FastAPI app for the Sudoku OpenEnv environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError("openenv-core is required to serve this environment") from exc

try:
    from ..models import SudokuAction, SudokuObservation
    from .sudoku_environment import SudokuEnvironment
except ImportError:  # pragma: no cover - Docker/root import fallback
    from models import SudokuAction, SudokuObservation
    from server.sudoku_environment import SudokuEnvironment


app = create_app(
    SudokuEnvironment,
    SudokuAction,
    SudokuObservation,
    env_name="sudoku_env",
    max_concurrent_envs=16,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
