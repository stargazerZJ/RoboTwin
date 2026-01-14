data_dir=${1}
repo_id=${2}
shift 2

# Forward any extra args to the converter (e.g. --episodes 0 1 2).
extra_args=("$@")

# Hide HuggingFace Datasets progress bars ("Map", "Creating parquet from Arrow format")
# while keeping the episode-level tqdm from the converter.
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export DATASETS_VERBOSITY=error

# NOTE: tyro expects --raw-dir / --repo-id (dashes), not --raw_dir / --repo_id (underscores).
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw-dir "$data_dir" --repo-id "$repo_id" "${extra_args[@]}"
