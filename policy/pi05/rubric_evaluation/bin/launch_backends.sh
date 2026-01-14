#!/usr/bin/env bash
set -euo pipefail

# Launch one websocket policy server per GPU.
# Uses policy/pi05/scripts/serve_policy.py (openpi websocket server).
#
# IMPORTANT (tyro subcommand ordering):
# - `policy:checkpoint` is a subcommand.
# - `--policy.config` and `--policy.dir` are ONLY valid for that subcommand.
# - `--port`, `--default-prompt`, and `--robotwin-repo-id` are top-level args.
#
# Correct form (matches `serve_policy.py --help`):
#   serve_policy.py --port 8000 --default-prompt "..." [--robotwin-repo-id ...] policy:checkpoint --policy.config ... --policy.dir ...
#
# Example:
#   bash policy/pi05/rubric_evaluation/bin/launch_backends.sh \
#     --train_config_name pi0_base_aloha_robotwin_lora \
#     --model_name blocks_ranking_split \
#     --checkpoint_id latest \
#     --gpus all \
#     --base_port 8000 \
#     --repo_id my_custom_dataset

TRAIN_CONFIG_NAME="pi0_base_aloha_robotwin_lora"
MODEL_NAME="blocks_ranking_split"
CHECKPOINT_ID="latest"
GPUS="all"
BASE_PORT="8000"
REPO_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_config_name) TRAIN_CONFIG_NAME="$2"; shift 2;;
    --model_name) MODEL_NAME="$2"; shift 2;;
    --checkpoint_id) CHECKPOINT_ID="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --base_port) BASE_PORT="$2"; shift 2;;
    --repo_id) REPO_ID="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# Determine repo root from script location (script is in policy/pi05/rubric_evaluation/bin/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# Activate venv (relative to repo root)
if [[ -f "policy/pi05/.venv/bin/activate" ]]; then
  source "policy/pi05/.venv/bin/activate"
fi

# Resolve GPU list
if [[ "$GPUS" == "all" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_IDS=$(nvidia-smi -L | sed -n 's/^GPU \([0-9]\+\):.*/\1/p' | tr '\n' ' ')
  else
    GPU_IDS="0"
  fi
else
  GPU_IDS=$(echo "$GPUS" | tr ',' ' ')
fi

CKPT_ROOT="policy/pi05/checkpoints/${TRAIN_CONFIG_NAME}/${MODEL_NAME}"

# NOTE: serve_policy.py internally resolves checkpoints via openpi.shared.download.maybe_download().
# In this repo, that function maps local paths under policy/pi05/checkpoints/* to /data/ckpts/*.
# So CKPT_DIR should remain under policy/pi05/checkpoints/ (not /data/ckpts).

if [[ "${CHECKPOINT_ID}" == "latest" ]]; then
  # Pick the numerically-largest step directory under CKPT_ROOT (e.g., 30000).
  if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "Checkpoint root not found: ${CKPT_ROOT}" >&2
    exit 1
  fi
  LATEST_STEP="$(ls -1 "${CKPT_ROOT}" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -n 1 || true)"
  if [[ -z "${LATEST_STEP}" ]]; then
    echo "No numeric checkpoint steps found under: ${CKPT_ROOT}" >&2
    echo "Available entries:" >&2
    ls -la "${CKPT_ROOT}" >&2 || true
    exit 1
  fi
  CKPT_DIR="${CKPT_ROOT}/${LATEST_STEP}"
else
  CKPT_DIR="${CKPT_ROOT}/${CHECKPOINT_ID}"
fi

echo "Launching backends for ckpt: ${CKPT_DIR}"
echo "GPU_IDS: ${GPU_IDS}"
echo "BASE_PORT: ${BASE_PORT}"
if [[ -n "$REPO_ID" ]]; then
  echo "REPO_ID: ${REPO_ID}"
fi

# Build extra arguments for serve_policy
EXTRA_ARGS=""
if [[ -n "$REPO_ID" ]]; then
  EXTRA_ARGS="--robotwin-repo-id ${REPO_ID}"
fi

for gid in ${GPU_IDS}; do
  port=$((BASE_PORT + gid))
  echo "Starting backend on GPU ${gid} port ${port}"

  CUDA_VISIBLE_DEVICES="${gid}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
    python -m policy.pi05.scripts.serve_policy \
      --port "${port}" \
      --default-prompt "Rank the blocks by color: blue left, green middle, red right." \
      ${EXTRA_ARGS} \
      policy:checkpoint \
      --policy.config "${TRAIN_CONFIG_NAME}" \
      --policy.dir "${CKPT_DIR}" \
      > "${REPO_ROOT}/policy/pi05/rubric_evaluation/backend_gpu${gid}.log" 2>&1 &
done

echo "Backends launched in background. Logs: ${REPO_ROOT}/policy/pi05/rubric_evaluation/backend_gpu*.log"
echo "PIDs:"
jobs -p || true