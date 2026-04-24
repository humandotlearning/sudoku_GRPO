---
title: Sudoku OpenEnv
emoji: 🎮
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - sudoku
---

# Sudoku OpenEnv

An OpenEnv-compatible Sudoku environment for agentic RL training. The agent
receives a puzzle board and plays by submitting typed actions:

```python
SudokuAction(row=0, col=1, number=5)
```

The environment keeps the solution hidden, validates moves against Sudoku rules,
and returns shaped rewards for valid progress plus a high terminal reward when
the puzzle is solved.

## API

- `reset(seed=None, difficulty=40)` creates a new Sudoku puzzle.
- `step(SudokuAction(row, col, number))` attempts to place a number.
- `state` returns episode metadata without revealing the solution.

## Run locally

```bash
uv sync
uv run server --port 8000
```

Then connect with:

```python
from sudoku_env import SudokuAction, SudokuEnv

with SudokuEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=42, difficulty=40)
    print(result.observation.message)
    result = env.step(SudokuAction(row=0, col=0, number=1))
```

