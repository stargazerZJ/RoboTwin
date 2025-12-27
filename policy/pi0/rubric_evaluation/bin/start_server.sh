#!/usr/bin/env bash
set -euo pipefail

# Start the rubric evaluation manager + web UI.
#
# Example:
#   bash policy/pi0/rubric_evaluation/bin/start_server.sh \
#     --task_name blocks_ranking_rgb \
#     --task_config demo_randomized \
#     --train_config_name pi0_base_aloha_robotwin_lora \
#     --model_name blocks_ranking_split \
#     --checkpoint_id latest \
#     --gpus all \
#     --base_port 8000 \
#     --num_episodes 500

TASK_NAME="blocks_ranking_rgb"
TASK_CONFIG="demo_randomized"
TRAIN_CONFIG_NAME="pi0_base_aloha_robotwin_lora"
MODEL_NAME="blocks_ranking_split"
CHECKPOINT_ID="latest"
GPUS="all"
BASE_PORT="8000"
NUM_EPISODES="500"
SEED_START="0"
PI0_STEP="50"
WORKERS_PER_BACKEND="1"
SERVER_HOST="0.0.0.0"
SERVER_PORT="8899"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task_name) TASK_NAME="$2"; shift 2;;
    --task_config) TASK_CONFIG="$2"; shift 2;;
    --train_config_name) TRAIN_CONFIG_NAME="$2"; shift 2;;
    --model_name) MODEL_NAME="$2"; shift 2;;
    --checkpoint_id) CHECKPOINT_ID="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --base_port) BASE_PORT="$2"; shift 2;;
    --num_episodes) NUM_EPISODES="$2"; shift 2;;
    --seed_start) SEED_START="$2"; shift 2;;
    --pi0_step) PI0_STEP="$2"; shift 2;;
    --rollout_workers_per_backend) WORKERS_PER_BACKEND="$2"; shift 2;;
    --server_host) SERVER_HOST="$2"; shift 2;;
    --server_port) SERVER_PORT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

source policy/pi0/.venv/bin/activate
cd /home/ubuntu/RoboTwin

python -m policy.pi0.rubric_evaluation.server.main \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --train_config_name "${TRAIN_CONFIG_NAME}" \
  --model_name "${MODEL_NAME}" \
  --checkpoint_id "${CHECKPOINT_ID}" \
  --gpus "${GPUS}" \
  --base_port "${BASE_PORT}" \
  --num_episodes "${NUM_EPISODES}" \
  --seed_start "${SEED_START}" \
  --pi0_step "${PI0_STEP}" \
  --rollout_workers_per_backend "${WORKERS_PER_BACKEND}" \
  --server_host "${SERVER_HOST}" \
  --server_port "${SERVER_PORT}"