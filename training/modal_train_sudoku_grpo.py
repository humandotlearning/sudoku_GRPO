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
        .uv_pip_install(
            "torch==2.10.0",
            "triton>=3.4.0",
            "torchvision==0.25.0",
            "bitsandbytes",
            "accelerate",
            "datasets",
            "huggingface_hub",
            "peft",
            "pillow",
            "timm",
            "tokenizers",
            "nvidia-ml-py",
            "trackio>=0.25.0",
            "transformers>=5.5.0",
            "trl>=0.28.0",
            "openenv-core[core]>=0.2.3",
        )
        .uv_pip_install(
            "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
            "unsloth[base] @ git+https://github.com/unslothai/unsloth",
        )
        .uv_pip_install("pydantic==2.10.6")
        .uv_pip_install(
            "mergekit",
            "immutables==0.21",
            extra_options="--no-deps",
        )
        .uv_pip_install("llm-blender", "weave")
        .uv_pip_install("trl>=0.28.0", "transformers>=5.5.0", "jmespath")
        .run_commands(
            "python -c \"import os, torch; import transformers.utils.hub as hub; hub.TRANSFORMERS_CACHE = getattr(hub, 'TRANSFORMERS_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')); from trl import GRPOConfig, GRPOTrainer; from openenv.core import EnvClient; print('trainer import ok', torch.__version__)\""
        )
    )


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = modal.Secret.from_name(SECRET_NAME)


@app.function(
    image=_training_image(),
    gpu=["L4", "A10G"],
    timeout=4 * 60 * 60,
    volumes={RUNS_DIR: volume},
    secrets=[secret],
)
def train_sudoku_grpo(
    env_repo_id: str,
    output_repo_id: str,
    max_steps: int = 20,
    difficulty: int = 12,
    min_difficulty: int = 0,
    max_difficulty: int = 40,
    difficulty_step: int = 4,
    gate_window: int = 32,
    gate_success_threshold: float = 0.65,
    dataset_size: int = 128,
    model_name: str = "unsloth/gemma-4-E2B-it",
    max_seq_length: int = 4096,
    max_completion_length: int = 64,
    lora_rank: int = 32,
    trackio_space_id: str = "Humanlearning/trackio",
    trackio_project: str = "sudoku-grpo",
    num_generations: int = 8,
) -> dict[str, str | int]:
    import torch
    import transformers.utils.hub as transformers_hub
    from datasets import Dataset
    from huggingface_hub import whoami

    if not hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
        transformers_hub.TRANSFORMERS_CACHE = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "huggingface",
            "hub",
        )

    import trackio
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastVisionModel

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing from the Modal secret sudoku-grpo-secrets.")

    if not trackio_space_id:
        user = whoami(token=hf_token)["name"]
        trackio_space_id = f"{user}/sudoku-trackio"

    os.environ["TRACKIO_SPACE_ID"] = trackio_space_id
    os.environ["TRACKIO_PROJECT"] = trackio_project

    package_url = f"git+https://huggingface.co/spaces/{env_repo_id}"
    subprocess.check_call(
        ["/.uv/uv", "pip", "install", "--python", sys.executable, "--no-deps", package_url]
    )

    from sudoku_env import SudokuAction, SudokuEnv

    env_url = f"https://{env_repo_id.replace('/', '-')}.hf.space"
    run_name = f"gemma4-e2b-sudoku-gated-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_difficulty = min_difficulty or difficulty
    max_difficulty = max(initial_difficulty, max_difficulty)
    gate_window = max(1, gate_window)
    difficulty_step = max(1, difficulty_step)
    curriculum = {
        "difficulty": int(initial_difficulty),
        "stage": 0,
        "recent_rewards": [],
        "recent_successes": [],
        "last_success_rate": 0.0,
        "last_mean_reward": 0.0,
    }

    training_prompt = (
        "Use the place_number tool exactly once, then stop. Rows and columns are zero-based. "
        "Do not write explanations. Choose one empty cell only, and place a value that is forced "
        "by Sudoku row, column, and box rules. Never place a number in a fixed clue."
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
            self.trace_messages: list[dict[str, str]] = []
            self.trace_metadata: dict[str, str | int | float | bool | None] = {}

        def reset(self, **kwargs) -> str:
            seed = kwargs.get("seed")
            current_difficulty = int(curriculum["difficulty"])
            result = self._sync_client.reset(seed=seed, difficulty=current_difficulty)
            obs = result.observation
            self.status = obs.status
            self.valid_moves = obs.valid_moves
            self.reward = -2.0
            self.done = result.done
            self.trace_messages = [
                {
                    "role": "user",
                    "content": f"{training_prompt}\n\nInitial Sudoku observation:\n{obs.message}",
                }
            ]
            self.trace_metadata = {
                "seed": seed,
                "difficulty": current_difficulty,
                "curriculum_stage": int(curriculum["stage"]),
                "status": self.status,
                "valid_moves": self.valid_moves,
                "done": self.done,
            }
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
            self.trace_messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": f"place_number(row={row}, col={col}, number={number})",
                    },
                    {"role": "tool", "content": obs.message},
                ]
            )
            self.trace_metadata.update(
                {
                    "status": self.status,
                    "valid_moves": self.valid_moves,
                    "done": self.done,
                    "reward": self.reward,
                }
            )
            return obs.message

        def _score(self) -> float:
            if self.status == "success":
                return 30.0
            if self.status == "failed":
                if self.valid_moves == 0:
                    return -2.0
                return -1.0
            if self.valid_moves == 0:
                return -0.75
            if self.valid_moves == 1:
                return 1.0
            return 0.5

        def __del__(self):
            try:
                self._sync_client.__exit__(None, None, None)
            except Exception:
                pass

    def _completion_to_text(completion) -> str:
        if completion is None:
            return ""
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list):
            parts = []
            for item in completion:
                if isinstance(item, dict):
                    parts.append(str(item.get("content", item)))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(completion)

    trace_step = {"value": 0}

    def _is_curriculum_success(env) -> bool:
        status = getattr(env, "status", "")
        valid_moves = int(getattr(env, "valid_moves", 0))
        return status == "success" or (
            status == "ongoing" and valid_moves == 1 and not bool(getattr(env, "done", False))
        )

    def _trackio_alert(title: str, text: str, level_name: str = "INFO") -> None:
        try:
            alert_level = getattr(trackio.AlertLevel, level_name, trackio.AlertLevel.INFO)
            trackio.alert(title=title, text=text, level=alert_level)
        except Exception as exc:
            print(f"Trackio alert skipped: {exc!r}")

    def _update_curriculum(rewards: list[float], environments) -> dict[str, float]:
        recent_rewards = curriculum["recent_rewards"]
        recent_successes = curriculum["recent_successes"]
        for reward, env in zip(rewards, environments):
            recent_rewards.append(float(reward))
            recent_successes.append(1.0 if _is_curriculum_success(env) else 0.0)

        del recent_rewards[:-gate_window]
        del recent_successes[:-gate_window]

        success_rate = (
            float(sum(recent_successes) / len(recent_successes)) if recent_successes else 0.0
        )
        mean_reward = float(sum(recent_rewards) / len(recent_rewards)) if recent_rewards else 0.0
        curriculum["last_success_rate"] = success_rate
        curriculum["last_mean_reward"] = mean_reward

        advanced = 0.0
        if (
            len(recent_successes) >= gate_window
            and success_rate >= gate_success_threshold
            and int(curriculum["difficulty"]) < max_difficulty
        ):
            previous = int(curriculum["difficulty"])
            curriculum["difficulty"] = min(max_difficulty, previous + difficulty_step)
            curriculum["stage"] = int(curriculum["stage"]) + 1
            recent_rewards.clear()
            recent_successes.clear()
            advanced = 1.0
            _trackio_alert(
                title="Sudoku curriculum advanced",
                text=(
                    f"Stage {curriculum['stage']} moved difficulty "
                    f"{previous} -> {curriculum['difficulty']} after "
                    f"{gate_window} samples at success_rate={success_rate:.3f}, "
                    f"mean_reward={mean_reward:.3f}."
                ),
            )

        return {
            "curriculum/difficulty": float(curriculum["difficulty"]),
            "curriculum/stage": float(curriculum["stage"]),
            "curriculum/rolling_success_rate": success_rate,
            "curriculum/rolling_mean_reward": mean_reward,
            "curriculum/window_samples": float(len(recent_successes)),
            "curriculum/advanced": advanced,
        }

    def sudoku_reward(environments, **kwargs) -> list[float]:
        rewards = [float(env._score()) for env in environments]
        completions = kwargs.get("completions") or kwargs.get("completion") or []
        trace_step["value"] += 1
        curriculum_metrics = _update_curriculum(rewards, environments)

        for index, env in enumerate(environments):
            messages = list(getattr(env, "trace_messages", []))
            if index < len(completions):
                completion_text = _completion_to_text(completions[index])
                if completion_text:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Raw generated completion:\n{completion_text}",
                        }
                    )
            metadata = dict(getattr(env, "trace_metadata", {}))
            metadata.update(
                {
                    "sample_index": index,
                    "reward": rewards[index],
                    "trace_step": trace_step["value"],
                    "curriculum_difficulty": int(curriculum["difficulty"]),
                    "curriculum_stage": int(curriculum["stage"]),
                    "curriculum_success": _is_curriculum_success(env),
                }
            )
            try:
                trackio.log(
                    {
                        f"sudoku_trace/sample_{index}": trackio.Trace(
                            messages=messages,
                            metadata=metadata,
                        ),
                        f"sudoku_progress/sample_{index}/valid_moves": float(
                            getattr(env, "valid_moves", 0)
                        ),
                        f"sudoku_progress/sample_{index}/done": float(
                            bool(getattr(env, "done", False))
                        ),
                        f"sudoku_progress/sample_{index}/failed": float(
                            getattr(env, "status", "") == "failed"
                        ),
                        f"sudoku_progress/sample_{index}/curriculum_success": float(
                            _is_curriculum_success(env)
                        ),
                    },
                    step=trace_step["value"],
                )
            except Exception as exc:
                print(f"Trackio trace logging skipped: {exc!r}")

        try:
            trackio.log(curriculum_metrics, step=trace_step["value"])
        except Exception as exc:
            print(f"Trackio curriculum logging skipped: {exc!r}")

        return rewards

    class TrackioSystemMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            try:
                metrics = trackio.log_gpu()
            except Exception as exc:
                print(f"Trackio GPU metrics skipped: {exc!r}")
                return control

            if metrics:
                summary = ", ".join(f"{key}={value}" for key, value in sorted(metrics.items())[:4])
                print(f"Trackio GPU metrics logged at step {state.global_step}: {summary}")
            return control

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Environment URL: {env_url}")
    print(f"Trackio Space: {trackio_space_id}")
    print(f"Output repo: {output_repo_id}")
    print(
        "Curriculum gate: "
        f"difficulty {initial_difficulty}->{max_difficulty}, step={difficulty_step}, "
        f"window={gate_window}, threshold={gate_success_threshold}"
    )

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
        gradient_accumulation_steps=max(2, num_generations),
        num_generations=num_generations,
        max_completion_length=max_completion_length,
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
        mask_truncated_completions=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=sudoku_reward,
        args=training_args,
        train_dataset=dataset,
        environment_factory=SudokuToolEnv,
        callbacks=[TrackioSystemMetricsCallback()],
    )
    trainer.train()
    trainer.push_to_hub()
    volume.commit()

    return {
        "run_name": run_name,
        "output_repo_id": output_repo_id,
        "env_repo_id": env_repo_id,
        "trackio_space_id": trackio_space_id,
        "trackio_project": trackio_project,
        "max_steps": max_steps,
        "min_difficulty": initial_difficulty,
        "max_difficulty": max_difficulty,
        "difficulty_step": difficulty_step,
        "gate_window": gate_window,
        "gate_success_threshold": gate_success_threshold,
        "max_completion_length": max_completion_length,
        "num_generations": num_generations,
    }


@app.local_entrypoint()
def main(
    env_repo_id: str = "",
    output_repo_id: str = "",
    max_steps: int = 20,
    difficulty: int = 12,
    min_difficulty: int = 0,
    max_difficulty: int = 40,
    difficulty_step: int = 4,
    gate_window: int = 32,
    gate_success_threshold: float = 0.65,
    dataset_size: int = 128,
    max_completion_length: int = 64,
    trackio_space_id: str = "Humanlearning/trackio",
    trackio_project: str = "sudoku-grpo",
    num_generations: int = 8,
) -> None:
    from huggingface_hub import whoami

    if not env_repo_id or not output_repo_id:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "Set HF_TOKEN locally or pass both --env-repo-id and --output-repo-id."
            )
        user = whoami(token=hf_token)["name"]
        env_repo_id = env_repo_id or f"{user}/sudoku-openenv"
        output_repo_id = output_repo_id or f"{user}/gemma4-e2b-sudoku-grpo-lora"

    call = train_sudoku_grpo.spawn(
        env_repo_id=env_repo_id,
        output_repo_id=output_repo_id,
        max_steps=max_steps,
        difficulty=difficulty,
        min_difficulty=min_difficulty,
        max_difficulty=max_difficulty,
        difficulty_step=difficulty_step,
        gate_window=gate_window,
        gate_success_threshold=gate_success_threshold,
        dataset_size=dataset_size,
        max_completion_length=max_completion_length,
        trackio_space_id=trackio_space_id,
        trackio_project=trackio_project,
        num_generations=num_generations,
    )
    print(f"Spawned Modal training call: {call.object_id}")
    print(f"Environment Space: https://huggingface.co/spaces/{env_repo_id}")
    print(f"Output model repo: https://huggingface.co/{output_repo_id}")
