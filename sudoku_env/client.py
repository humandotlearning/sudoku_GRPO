"""Client for the Sudoku OpenEnv environment."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SudokuAction, SudokuObservation, SudokuState


class SudokuEnv(EnvClient[SudokuAction, SudokuObservation, SudokuState]):
    """WebSocket client for the Sudoku OpenEnv environment."""

    def _step_payload(self, action: SudokuAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[SudokuObservation]:
        obs_data = payload.get("observation", {})
        observation = SudokuObservation(
            board=obs_data.get("board", []),
            initial_board=obs_data.get("initial_board", []),
            message=obs_data.get("message", ""),
            valid_moves=obs_data.get("valid_moves", 0),
            remaining_empty=obs_data.get("remaining_empty", 0),
            status=obs_data.get("status", "ongoing"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> SudokuState:
        return SudokuState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", 40),
            seed=payload.get("seed"),
            moves=payload.get("moves", payload.get("step_count", 0)),
            valid_moves=payload.get("valid_moves", 0),
            remaining_empty=payload.get("remaining_empty", 0),
            status=payload.get("status", "ongoing"),
        )

