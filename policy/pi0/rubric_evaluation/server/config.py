from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class EvalConfig:
    # Task
    task_name: str = "blocks_ranking_rgb"
    task_config: str = "demo_randomized"

    # Policy checkpoint (served by backends)
    train_config_name: str = "pi0_base_aloha_robotwin_lora"
    model_name: str = "blocks_ranking_split"
    checkpoint_id: str = "latest"

    # Backend ports (one per GPU)
    host: str = "127.0.0.1"
    base_port: int = 8000
    gpus: str = "all"  # "all" or "0,1,2" etc.

    # Evaluation
    num_episodes: int = 500
    seed_start: int = 0
    max_steps: int = 500  # env.step_lim is used if available; this is a fallback
    pi0_step: int = 50  # action chunk length (matches deploy_policy.yml default)

    # Output
    runs_root: Path = Path("policy/pi0/rubric_evaluation/runs")
    server_host: str = "0.0.0.0"
    server_port: int = 8899

    # Rubric settings
    rubric_variant: str = "baseline"  # "baseline" or "subtask"

    # Concurrency
    rollout_workers_per_backend: int = 1  # start with 1; increase if GPU underutilized

    def backend_ports(self, detected_gpu_ids: List[int]) -> List[int]:
        return [self.base_port + i for i in detected_gpu_ids]