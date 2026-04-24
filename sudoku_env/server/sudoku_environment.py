"""Sudoku game logic exposed as an OpenEnv environment."""

from __future__ import annotations

import copy
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import Board, SudokuAction, SudokuObservation, SudokuState
except ImportError:  # pragma: no cover - Docker/root import fallback
    from models import Board, SudokuAction, SudokuObservation, SudokuState


def is_valid_placement(board: Board, row: int, col: int, number: int) -> bool:
    """Return whether `number` can be placed at `row`, `col` under Sudoku rules."""
    if not (0 <= row < 9 and 0 <= col < 9 and 1 <= number <= 9):
        return False

    if board[row][col] != 0:
        return False

    if number in board[row]:
        return False

    if number in (board[r][col] for r in range(9)):
        return False

    box_row = 3 * (row // 3)
    box_col = 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == number:
                return False

    return True


def _find_empty(board: Board) -> tuple[int, int] | None:
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col
    return None


def _solve_sudoku(board: Board, rng: random.Random | None = None) -> bool:
    empty = _find_empty(board)
    if empty is None:
        return True

    row, col = empty
    numbers = list(range(1, 10))
    if rng is not None:
        rng.shuffle(numbers)

    for number in numbers:
        if is_valid_placement(board, row, col, number):
            board[row][col] = number
            if _solve_sudoku(board, rng):
                return True
            board[row][col] = 0

    return False


def generate_complete_board(rng: random.Random) -> Board:
    """Generate a complete valid Sudoku board."""
    board = [[0 for _ in range(9)] for _ in range(9)]

    for box in range(3):
        numbers = list(range(1, 10))
        rng.shuffle(numbers)
        for i in range(3):
            for j in range(3):
                board[box * 3 + i][box * 3 + j] = numbers[i * 3 + j]

    solved = _solve_sudoku(board, rng)
    if not solved:  # pragma: no cover - defensive guard
        raise RuntimeError("failed to generate Sudoku board")

    return board


def count_empty(board: Board) -> int:
    return sum(1 for row in board for value in row if value == 0)


def is_complete_valid_board(board: Board) -> bool:
    required = set(range(1, 10))
    rows_ok = all(set(row) == required for row in board)
    cols_ok = all({board[row][col] for row in range(9)} == required for col in range(9))
    boxes_ok = True
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            values = {
                board[row][col]
                for row in range(box_row, box_row + 3)
                for col in range(box_col, box_col + 3)
            }
            boxes_ok = boxes_ok and values == required
    return rows_ok and cols_ok and boxes_ok


def format_board(board: Board) -> str:
    """Format a board in compact ASCII for LLM observations."""
    lines: list[str] = []
    for row_idx, row in enumerate(board):
        if row_idx in (3, 6):
            lines.append("------+-------+------")
        cells = ["." if value == 0 else str(value) for value in row]
        lines.append(
            " ".join(cells[0:3]) + " | " + " ".join(cells[3:6]) + " | " + " ".join(cells[6:9])
        )
    return "\n".join(lines)


class SudokuEnvironment(Environment[SudokuAction, SudokuObservation, SudokuState]):
    """OpenEnv environment where an agent solves Sudoku one placement at a time."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, difficulty: int = 40, max_moves: int = 100):
        self.default_difficulty = difficulty
        self.max_moves = max_moves
        self._state = SudokuState(
            episode_id=str(uuid4()),
            step_count=0,
            difficulty=difficulty,
            seed=None,
            moves=0,
            valid_moves=0,
            remaining_empty=0,
            status="ongoing",
        )
        self._board: Board = [[0 for _ in range(9)] for _ in range(9)]
        self._initial_board: Board = [[0 for _ in range(9)] for _ in range(9)]
        self._solution: Board = [[0 for _ in range(9)] for _ in range(9)]

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        difficulty: int | None = None,
        **_: object,
    ) -> SudokuObservation:
        difficulty = self.default_difficulty if difficulty is None else difficulty
        difficulty = max(0, min(81, int(difficulty)))

        rng = random.Random(seed)
        complete_board = generate_complete_board(rng)
        self._solution = copy.deepcopy(complete_board)
        self._board = copy.deepcopy(complete_board)

        cells = [(row, col) for row in range(9) for col in range(9)]
        rng.shuffle(cells)
        for row, col in cells[:difficulty]:
            self._board[row][col] = 0

        self._initial_board = copy.deepcopy(self._board)
        self._state = SudokuState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            difficulty=difficulty,
            seed=seed,
            moves=0,
            valid_moves=0,
            remaining_empty=count_empty(self._board),
            status="ongoing",
        )

        return self._observation(
            "New Sudoku puzzle. Place one number at a time using row, col, and number."
        )

    def step(
        self,
        action: SudokuAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> SudokuObservation:
        del timeout_s

        if self._state.status != "ongoing":
            return self._observation("Game is already over.", reward=self._score())

        self._state.step_count += 1
        self._state.moves += 1

        if self._state.moves > self.max_moves:
            self._state.status = "failed"
            return self._observation("Move limit exceeded.", reward=self._score())

        row, col, number = action.row, action.col, action.number
        if self._initial_board[row][col] != 0:
            self._state.status = "failed"
            return self._observation(
                f"Invalid move: row {row}, col {col} is a fixed clue.",
                reward=self._score(),
            )

        if not is_valid_placement(self._board, row, col, number):
            self._state.status = "failed"
            return self._observation(
                f"Invalid move: cannot place {number} at row {row}, col {col}.",
                reward=self._score(),
            )

        self._board[row][col] = number
        self._state.valid_moves += 1
        self._state.remaining_empty = count_empty(self._board)

        if self._state.remaining_empty == 0:
            self._state.status = (
                "success" if self._board == self._solution and is_complete_valid_board(self._board) else "failed"
            )
            if self._state.status == "success":
                return self._observation("Puzzle solved.", reward=self._score())
            return self._observation("Board is full but does not match the hidden solution.", reward=self._score())

        return self._observation(
            f"Accepted move: placed {number} at row {row}, col {col}.",
            reward=self._score(),
        )

    @property
    def state(self) -> SudokuState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Sudoku OpenEnv",
            description="Solve Sudoku by placing one valid number per action.",
            version="0.1.0",
            documentation_url="https://huggingface.co/spaces/openenv/sudoku",
        )

    def _score(self) -> float:
        if self._state.status == "success":
            return 30.0
        if self._state.status == "failed" and self._state.valid_moves == 0:
            return -2.0
        return float(self._state.valid_moves) * 0.2

    def _observation(self, feedback: str, reward: float | None = None) -> SudokuObservation:
        board = copy.deepcopy(self._board)
        initial = copy.deepcopy(self._initial_board)
        reward_value = self._score() if reward is None else reward
        message = (
            f"{feedback}\n"
            f"Status: {self._state.status}. Valid moves: {self._state.valid_moves}. "
            f"Empty cells left: {self._state.remaining_empty}.\n\n"
            f"Current board:\n{format_board(board)}\n\n"
            "Use place_number(row, col, number) with zero-based row/col indices. "
            "Only place numbers into cells that are empty in both the initial board and current board."
        )

        return SudokuObservation(
            board=board,
            initial_board=initial,
            message=message,
            valid_moves=self._state.valid_moves,
            remaining_empty=self._state.remaining_empty,
            status=self._state.status,
            done=self._state.status != "ongoing",
            reward=reward_value,
        )

