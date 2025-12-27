#!/bin/bash
# Test script for running a single episode to verify fixes
# Usage: bash policy/pi0/rubric_evaluation/bin/test_single_episode.sh [GPU_PORT]
# Example: bash policy/pi0/rubric_evaluation/bin/test_single_episode.sh 8004

set -e
cd "$(dirname "$0")/../../../.."

GPU_PORT=${1:-8004}
OUT_DIR="policy/pi0/rubric_evaluation/runs_single_debug/test_fix_$(date +%s)"

source policy/pi0/.venv/bin/activate

python -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
from policy.pi0.rubric_evaluation.server.rollout_worker import WorkerConfig, run_one_episode

result = run_one_episode(
    backend_host='localhost',
    backend_port=${GPU_PORT},
    rubric_path=Path('policy/pi0/rubric_evaluation/rubrics/blocks_ranking_rgb_rubric.py'),
    out_dir=Path('${OUT_DIR}'),
    episode_id=0,
    seed=0,
    worker_cfg=WorkerConfig(
        task_name='blocks_ranking_rgb',
        task_config='demo_randomized',
        pi0_step=50,
        max_steps_fallback=500,
    ),
)
print('Result:', result)
print('Video saved to: ${OUT_DIR}/videos/episode_0000.mp4')
"