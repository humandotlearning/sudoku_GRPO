"""Modal entrypoint for Sudoku GRPO training with Unsloth, TRL, and Trackio."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

import modal


APP_NAME = "sudoku-grpo"
VOLUME_NAME = "sudoku-grpo-runs"
SECRET_NAME = "sudoku-grpo-secrets"
RUNS_DIR = pathlib.Path("/runs")


def _training_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("git", "build-essential", "curl")
        .pip_install(
            "torch>=2.8.0",
            "triton>=3.4.0",
            "torchvision",
            "bitsandbytes",
            "accelerate",
            "datasets",
            "huggingface_hub",
            "peft",
            "pillow",
            "timm",
            "tokenizers",
            "trackio",
            "transformers>=5.5.0",
            "trl>=0.28.0",
            "openenv-core[core]>=0.2.3",
        )
        .pip_install(
            "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
            "unsloth[base] @ git+https://github.com/unslothai/unsloth",
        )
    )


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = modal.Secret.from_name(SECRET_NAME)


@app.function(
    image=_training_image(),
    gpu=["L40S", "A100-40GB"],
    timeout=4 * 60 * 60,
    volumes={RUNS_DIR: volume},
    secrets=[secret],
)
def train_sudoku_grpo(
    env_repo_id: str,
    output_repo_id: str,
    max_steps: int = 20,
    difficulty: int = 40,
    dataset_size: int = 128,
    model_name: str = "unsloth/gemma-4-E2B-it",
    max_seq_length: int = 4096,
    lora_rank: int = 32,
) -> dict[str, str | int]:
    import torch
    from datasets import Dataset
    from huggingface_hub import whoami
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastVisionModel

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing from the Modal secret sudoku-grpo-secrets.")

    trackio_space_id = os.environ.get("TRACKIO_SPACE_ID")
    if not trackio_space_id:
        user = whoami(token=hf_token)["name"]
        trackio_space_id = f"{user}/sudoku-trackio"
        os.environ["TRACKIO_SPACE_ID"] = trackio_space_id

    os.environ.setdefault("TRACKIO_PROJECT", "sudoku-grpo")

    package_url = f"git+https://huggingface.co/spaces/{env_repo_id}"
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_url])

    from sudoku_env import SudokuAction, SudokuEnv

    env_url = f"https://{env_repo_id.replace('/', '-')}.hf.space"
    run_name = f"gemma4-e2b-sudoku-smoke-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_prompt = (
        "Solve the Sudoku puzzle by using the available place_number tool. "
        "Rows and columns are zero-based. Only place a digit in a cell that is empty "
        "in both the initial board and current board. Keep using the tool until the puzzle is solved."
    )

    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": training_prompt}],
                "difficulty": difficulty,
                "seed": seed,
            }
            for seed in range(dataset_size)
        ]
    )

    class SudokuToolEnv:
        def __init__(self):
            self._sync_client = SudokuEnv(base_url=env_url).sync()
            self._sync_client.__enter__()
            self.status = "not_started"
            self.valid_moves = 0
            self.reward = -2.0
            self.done = False

        def reset(self, **kwargs) -> str:
            seed = kwargs.get("seed")
            current_difficulty = kwargs.get("difficulty", difficulty)
            result = self._sync_client.reset(seed=seed, difficulty=current_difficulty)
            obs = result.observation
            self.status = obs.status
            self.valid_moves = obs.valid_moves
            self.reward = -2.0
            self.done = result.done
            return obs.message

        def place_number(self, row: int, col: int, number: int) -> str:
            """
            Place one number on the Sudoku board.

            Args:
                row: Zero-based row index from 0 to 8.
                col: Zero-based column index from 0 to 8.
                number: Digit to place from 1 to 9.

            Returns:
                Feedback with the updated board and episode status.
            """
            if self.done:
                raise ValueError("Game is already over.")

            result = self._sync_client.step(SudokuAction(row=row, col=col, number=number))
            obs = result.observation
            self.status = obs.status
            self.valid_moves = obs.valid_moves
            self.done = result.done
            self.reward = self._score()
            return obs.message

        def _score(self) -> float:
            if self.status == "success":
                return 30.0
            if self.status == "failed" and self.valid_moves == 0:
                return -2.0
            return float(self.valid_moves) * 0.2

        def __del__(self):
            try:
                self._sync_client.__exit__(None, None, None)
            except Exception:
                pass

    def sudoku_reward(environments, **kwargs) -> list[float]:
        return [float(env._score()) for env in environments]

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Environment URL: {env_url}")
    print(f"Trackio Space: {trackio_space_id}")
    print(f"Output repo: {output_repo_id}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    FastVisionModel.for_training(model)

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=5e-5,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_completion_length=max_seq_length,
        max_steps=max_steps,
        save_steps=max(10, max_steps),
        report_to="trackio",
        run_name=run_name,
        output_dir=str(output_dir),
        push_to_hub=True,
        hub_model_id=output_repo_id,
        hub_private_repo=True,
        hub_strategy="every_save",
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        loss_type="bnpo",
        mask_truncated_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=sudoku_reward,
        args=training_args,
        train_dataset=dataset,
        environment_factory=SudokuToolEnv,
    )
    trainer.train()
    trainer.push_to_hub()
    volume.commit()

    return {
        "run_name": run_name,
        "output_repo_id": output_repo_id,
        "env_repo_id": env_repo_id,
        "trackio_space_id": trackio_space_id,
        "max_steps": max_steps,
    }


@app.local_entrypoint()
def main(
    env_repo_id: str = "",
    output_repo_id: str = "",
    max_steps: int = 20,
    difficulty: int = 40,
    dataset_size: int = 128,
) -> None:
    from huggingface_hub import whoami

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN locally so the launcher can infer default repo IDs.")

    user = whoami(token=hf_token)["name"]
    env_repo_id = env_repo_id or f"{user}/sudoku-openenv"
    output_repo_id = output_repo_id or f"{user}/gemma4-e2b-sudoku-grpo-lora"

    call = train_sudoku_grpo.spawn(
        env_repo_id=env_repo_id,
        output_repo_id=output_repo_id,
        max_steps=max_steps,
        difficulty=difficulty,
        dataset_size=dataset_size,
    )
    print(f"Spawned Modal training call: {call.object_id}")
    print(f"Environment Space: https://huggingface.co/spaces/{env_repo_id}")
    print(f"Output model repo: https://huggingface.co/{output_repo_id}")

