#!/usr/bin/env bash
set -euo pipefail

# Usage: finetune.sh <train_config_name> <model_name> <gpu_use> [--repo_id <dataset_repo_id>]
#
# Example:
#   ./finetune.sh pi0_base_aloha_robotwin_lora blocks_ranking_split 0,1,2,3
#   ./finetune.sh pi0_base_aloha_robotwin_lora blocks_ranking_split 0,1 --repo_id my_custom_dataset

train_config_name=$1
model_name=$2
gpu_use=$3
shift 3

# Parse optional arguments
REPO_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo_id) REPO_ID="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

export CUDA_VISIBLE_DEVICES=$gpu_use
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Build extra arguments for tyro CLI
EXTRA_ARGS=""
if [[ -n "$REPO_ID" ]]; then
  # Override the data.repo_id in the training config
  EXTRA_ARGS="--data.repo-id=$REPO_ID"
  echo "Overriding dataset repo_id: $REPO_ID"
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite $EXTRA_ARGS