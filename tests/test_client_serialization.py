from sudoku_env.client import SudokuEnv
from sudoku_env.models import SudokuAction


def test_client_step_payload_uses_action_fields():
    client = SudokuEnv(base_url="http://localhost:8000")
    payload = client._step_payload(SudokuAction(row=1, col=2, number=3))

    assert payload["row"] == 1
    assert payload["col"] == 2
    assert payload["number"] == 3


def test_client_parse_result_round_trips_observation():
    client = SudokuEnv(base_url="http://localhost:8000")
    payload = {
        "observation": {
            "board": [[0] * 9 for _ in range(9)],
            "initial_board": [[0] * 9 for _ in range(9)],
            "message": "ready",
            "valid_moves": 0,
            "remaining_empty": 81,
            "status": "ongoing",
        },
        "reward": 0.0,
        "done": False,
    }

    result = client._parse_result(payload)

    assert result.observation.message == "ready"
    assert result.observation.remaining_empty == 81
    assert result.done is False

