"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.

Supports two modes:
1. Fast parquet-direct mode (default): Reads only numeric columns, skips image decoding
2. Full dataset mode: Uses LeRobotDataset with all transforms (slower but more accurate)

Parallel data loading via ProcessPoolExecutor for better CPU utilization.
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def _get_parquet_files(repo_id: str) -> list[Path]:
    """Get all parquet files for a LeRobot dataset."""
    # LeRobot stores datasets in HF_LEROBOT_HOME or ~/.cache/huggingface/lerobot
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    dataset_path = HF_LEROBOT_HOME / repo_id / "data"
    if not dataset_path.exists():
        # Try alternate location
        dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id / "data"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    parquet_files = sorted(dataset_path.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_path}")

    return parquet_files


def _read_parquet_chunk(parquet_path: str, use_delta_actions: bool = True) -> dict:
    """Read state and action columns from a parquet file, skipping images.

    Args:
        parquet_path: Path to parquet file
        use_delta_actions: If True, convert actions to delta (relative to state)

    Returns:
        Dict with 'state' and 'actions' numpy arrays
    """
    # Read only numeric columns, skip all image columns
    table = pq.read_table(
        parquet_path,
        columns=["observation.state", "action"]
    )

    # Convert to numpy
    states_list = table["observation.state"].to_pylist()
    actions_list = table["action"].to_pylist()

    states = np.array(states_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.float32)

    # Apply delta action transform (same as DeltaActions transform)
    # For Aloha: delta for joints (first 6), absolute for gripper (7th)
    # Left arm: dims 0-5 delta, dim 6 absolute
    # Right arm: dims 7-12 delta, dim 13 absolute
    if use_delta_actions and states.shape[-1] == 14 and actions.shape[-1] == 14:
        # Create delta mask: True for joint dims, False for gripper dims
        delta_mask = np.array([True, True, True, True, True, True, False,  # left arm
                               True, True, True, True, True, True, False], dtype=bool)  # right arm
        # actions = actions - state for masked dimensions
        actions[:, delta_mask] = actions[:, delta_mask] - states[:, delta_mask]

    return {
        "state": states,
        "actions": actions,
    }


def _parallel_read_parquet(
    parquet_files: list[Path],
    num_workers: int,
    use_delta_actions: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Read all parquet files in parallel, extracting only state and actions."""

    all_states = []
    all_actions = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all files
        futures = {
            executor.submit(_read_parquet_chunk, str(pf), use_delta_actions): pf
            for pf in parquet_files
        }

        # Collect results with progress bar
        with tqdm.tqdm(total=len(parquet_files), desc="Reading parquet files") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_states.append(result["state"])
                    all_actions.append(result["actions"])
                except Exception as e:
                    pf = futures[future]
                    print(f"Error reading {pf}: {e}")
                pbar.update(1)

    # Concatenate all data
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    return states, actions


# ============== Full dataset mode (slower, uses LeRobotDataset) ==============

# Global variable to hold dataset in worker processes
_worker_dataset = None


def _init_worker(config_name: str):
    """Initialize worker process with its own dataset instance."""
    global _worker_dataset
    # Disable JAX GPU preallocation in workers
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    config = _config.get_config(config_name)
    _, _worker_dataset = create_dataset(config)


def _load_batch(indices: list[int]) -> dict:
    """Load a batch of samples in worker process."""
    global _worker_dataset
    states = []
    actions = []
    for idx in indices:
        sample = _worker_dataset[idx]
        state = np.asarray(sample["state"], dtype=np.float32).flatten()
        action = np.asarray(sample["actions"], dtype=np.float32)
        states.append(state)
        actions.append(action.reshape(-1, action.shape[-1]))
    return {
        "state": np.stack(states, axis=0),
        "actions": np.concatenate(actions, axis=0),
    }


def main(
    config_name: str,
    max_frames: int | None = None,
    num_workers: int | None = None,
    fast_mode: bool = True,
    use_delta_actions: bool = True,
    batch_size: int = 100,
):
    """Compute normalization statistics with parallel data loading.

    Args:
        config_name: Name of the training config.
        max_frames: Maximum number of frames to process. If None, process all frames.
        num_workers: Number of parallel workers. Defaults to min(cpu_count, 64).
        fast_mode: If True, read parquet files directly (skip image decoding).
                   If False, use full LeRobotDataset with transforms.
        use_delta_actions: If True (and fast_mode), convert actions to delta space.
        batch_size: Number of samples per batch (only used when fast_mode=False).
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # Determine number of workers
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 64)

    print(f"Config: {config_name}")
    print(f"Repo ID: {data_config.repo_id}")
    print(f"Fast mode: {fast_mode}, Delta actions: {use_delta_actions}")
    print(f"Workers: {num_workers}")

    if fast_mode:
        # Fast parquet-direct mode
        parquet_files = _get_parquet_files(data_config.repo_id)
        print(f"Found {len(parquet_files)} parquet files")

        states, actions = _parallel_read_parquet(
            parquet_files,
            num_workers=num_workers,
            use_delta_actions=use_delta_actions,
        )

        if max_frames is not None and max_frames < len(states):
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(len(states), size=max_frames, replace=False)
            states = states[indices]
            # For actions, we need to handle the horizon dimension
            # In fast mode, actions are per-frame (no horizon expansion)
            actions = actions[indices]
    else:
        # Full dataset mode (slower, loads images)
        _, dataset = create_dataset(config)
        num_frames = len(dataset)
        indices = list(range(num_frames))

        if max_frames is not None and max_frames < num_frames:
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(num_frames, size=max_frames, replace=False).tolist()
            num_frames = max_frames

        print(f"Processing {num_frames} frames")

        # Split indices into batches
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        all_states = []
        all_actions = []

        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(config_name,),
        ) as executor:
            futures = {executor.submit(_load_batch, batch): i for i, batch in enumerate(batches)}

            with tqdm.tqdm(total=len(batches), desc="Loading data") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    all_states.append(result["state"])
                    all_actions.append(result["actions"])
                    pbar.update(1)

        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)

    print(f"State shape: {states.shape}, Actions shape: {actions.shape}")
    print("Computing statistics...")

    # Compute statistics using NumPy's optimized functions
    norm_stats = {
        "state": normalize.NormStats(
            mean=np.mean(states, axis=0),
            std=np.std(states, axis=0),
            q01=np.percentile(states, 1, axis=0),
            q99=np.percentile(states, 99, axis=0),
        ),
        "actions": normalize.NormStats(
            mean=np.mean(actions, axis=0),
            std=np.std(actions, axis=0),
            q01=np.percentile(actions, 1, axis=0),
            q99=np.percentile(actions, 99, axis=0),
        ),
    }

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print("Done!")


if __name__ == "__main__":
    tyro.cli(main)
