"""Pydantic models for the Sudoku OpenEnv environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

Board = list[list[int]]
SudokuStatus = Literal["ongoing", "success", "failed"]


class SudokuAction(Action):
    """Action that attempts to place one number on the Sudoku board."""

    row: int = Field(..., ge=0, le=8, description="Row index, from 0 to 8")
    col: int = Field(..., ge=0, le=8, description="Column index, from 0 to 8")
    number: int = Field(..., ge=1, le=9, description="Digit to place, from 1 to 9")


class SudokuObservation(Observation):
    """Observation returned by the Sudoku environment."""

    board: Board = Field(..., description="Current 9x9 board; 0 means empty")
    initial_board: Board = Field(..., description="Initial fixed puzzle board")
    message: str = Field(..., description="Human-readable state and feedback")
    valid_moves: int = Field(default=0, ge=0, description="Accepted moves so far")
    remaining_empty: int = Field(default=0, ge=0, le=81, description="Empty cells left")
    status: SudokuStatus = Field(default="ongoing", description="Episode status")


class SudokuState(State):
    """Server-side Sudoku state without the hidden solution."""

    difficulty: int = Field(default=40, ge=0, le=81)
    seed: int | None = Field(default=None)
    moves: int = Field(default=0, ge=0)
    valid_moves: int = Field(default=0, ge=0)
    remaining_empty: int = Field(default=0, ge=0, le=81)
    status: SudokuStatus = Field(default="ongoing")

