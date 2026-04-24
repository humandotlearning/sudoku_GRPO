import copy

from sudoku_env.models import SudokuAction
from sudoku_env.server.sudoku_environment import (
    SudokuEnvironment,
    count_empty,
    is_complete_valid_board,
    is_valid_placement,
)


def test_reset_creates_puzzle_without_solution_leakage():
    env = SudokuEnvironment()
    obs = env.reset(seed=123, difficulty=40)

    assert obs.status == "ongoing"
    assert count_empty(obs.board) == 40
    assert obs.remaining_empty == 40
    assert is_complete_valid_board(env._solution)
    assert "solution" not in obs.model_dump_json().lower()
    assert "solution" not in env.state.model_dump_json().lower()


def test_valid_solution_moves_solve_puzzle():
    env = SudokuEnvironment()
    obs = env.reset(seed=42, difficulty=5)

    empties = [
        (row, col)
        for row in range(9)
        for col in range(9)
        if obs.initial_board[row][col] == 0
    ]

    result = None
    for row, col in empties:
        result = env.step(SudokuAction(row=row, col=col, number=env._solution[row][col]))

    assert result is not None
    assert result.status == "success"
    assert result.done is True
    assert result.reward == 30.0


def test_cannot_modify_fixed_clue():
    env = SudokuEnvironment()
    obs = env.reset(seed=7, difficulty=40)
    row, col = next(
        (r, c)
        for r in range(9)
        for c in range(9)
        if obs.initial_board[r][c] != 0
    )

    result = env.step(SudokuAction(row=row, col=col, number=obs.initial_board[row][col]))

    assert result.status == "failed"
    assert result.done is True
    assert result.reward == -2.0


def test_invalid_duplicate_move_fails():
    env = SudokuEnvironment()
    obs = env.reset(seed=9, difficulty=40)
    row, col = next(
        (r, c)
        for r in range(9)
        for c in range(9)
        if obs.initial_board[r][c] == 0
    )
    duplicate = next(value for value in obs.board[row] if value != 0)

    assert not is_valid_placement(copy.deepcopy(obs.board), row, col, duplicate)
    result = env.step(SudokuAction(row=row, col=col, number=duplicate))

    assert result.status == "failed"
    assert result.done is True
    assert result.reward == -2.0


def test_valid_partial_move_gets_shaped_reward():
    env = SudokuEnvironment()
    obs = env.reset(seed=11, difficulty=40)
    row, col = next(
        (r, c)
        for r in range(9)
        for c in range(9)
        if obs.initial_board[r][c] == 0
    )

    result = env.step(SudokuAction(row=row, col=col, number=env._solution[row][col]))

    assert result.status == "ongoing"
    assert result.reward == 0.2
    assert result.valid_moves == 1
    assert result.remaining_empty == 39


def test_move_limit_guard_fails_episode():
    env = SudokuEnvironment(max_moves=0)
    env.reset(seed=12, difficulty=1)
    row, col = next(
        (r, c)
        for r in range(9)
        for c in range(9)
        if env._initial_board[r][c] == 0
    )

    result = env.step(SudokuAction(row=row, col=col, number=env._solution[row][col]))

    assert result.status == "failed"
    assert result.done is True

