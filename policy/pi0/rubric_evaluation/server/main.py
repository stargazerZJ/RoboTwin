from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from policy.pi0.rubric_evaluation.server.config import EvalConfig
from policy.pi0.rubric_evaluation.server.eval_manager import EvalManager
from policy.pi0.rubric_evaluation.server.webapp import create_app


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", default="blocks_ranking_rgb")
    p.add_argument("--task_config", default="demo_randomized")
    p.add_argument("--train_config_name", default="pi0_base_aloha_robotwin_lora")
    p.add_argument("--model_name", default="blocks_ranking_split")
    p.add_argument("--checkpoint_id", default="latest")
    p.add_argument("--gpus", default="all")
    p.add_argument("--base_port", type=int, default=8000)
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--pi0_step", type=int, default=50)
    p.add_argument("--rollout_workers_per_backend", type=int, default=1)
    p.add_argument("--server_host", default="0.0.0.0")
    p.add_argument("--server_port", type=int, default=8899)
    args = p.parse_args()

    return EvalConfig(
        task_name=args.task_name,
        task_config=args.task_config,
        train_config_name=args.train_config_name,
        model_name=args.model_name,
        checkpoint_id=args.checkpoint_id,
        gpus=args.gpus,
        base_port=args.base_port,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        pi0_step=args.pi0_step,
        rollout_workers_per_backend=args.rollout_workers_per_backend,
        server_host=args.server_host,
        server_port=args.server_port,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    cfg = parse_args()

    current_rubric_path = Path("policy/pi0/rubric_evaluation/rubrics/blocks_ranking_rgb_rubric.py")
    manager = EvalManager(cfg, current_rubric_path=current_rubric_path)
    manager.start()

    app = create_app(cfg=cfg, manager=manager)
    uvicorn.run(app, host=cfg.server_host, port=cfg.server_port, log_level="info")


if __name__ == "__main__":
    main()