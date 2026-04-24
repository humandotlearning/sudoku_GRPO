# Sudoku OpenEnv GRPO

This workspace contains a custom Sudoku OpenEnv environment and a Modal training
entrypoint for a GRPO smoke run with Unsloth, TRL, and Trackio.

## Local setup

```powershell
uv venv --python 3.11.11
uv pip install --python .venv\Scripts\python.exe -e sudoku_env[dev] modal huggingface_hub pytest httpx
```

## Validate the environment

```powershell
uv run --python .venv\Scripts\python.exe pytest
.venv\Scripts\openenv.exe validate sudoku_env --verbose
```

Docker Desktop must be running before building the OpenEnv image:

```powershell
.venv\Scripts\openenv.exe build sudoku_env -t sudoku-openenv:latest
```

## Push to Hugging Face

Set `HF_TOKEN` with write access, then run:

```powershell
.venv\Scripts\python.exe scripts\push_openenv.py
```

The default Space repo is `{hf_user}/sudoku-openenv`.

## Modal smoke training

Create a Modal secret named `sudoku-grpo-secrets` containing:

- `HF_TOKEN`
- `TRACKIO_SPACE_ID`, for example `{hf_user}/sudoku-trackio`
- `TRACKIO_PROJECT`, for example `sudoku-grpo`

Then launch:

```powershell
modal run --detach training/modal_train_sudoku_grpo.py --max-steps 20
```

