"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.

Supports parallel data loading via ProcessPoolExecutor for better CPU utilization.
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
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


# Global variable to hold dataset in worker processes (avoid recreating per sample)
_worker_dataset = None
_worker_config_name = None


def _init_worker(config_name: str):
    """Initialize worker process with its own dataset instance."""
    global _worker_dataset, _worker_config_name
    # Disable JAX GPU preallocation in workers
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    _worker_config_name = config_name
    config = _config.get_config(config_name)
    _, _worker_dataset = create_dataset(config)


def _load_sample(idx: int) -> dict:
    """Load a single sample in worker process."""
    global _worker_dataset
    sample = _worker_dataset[idx]
    state = np.asarray(sample["state"], dtype=np.float32).flatten()
    actions = np.asarray(sample["actions"], dtype=np.float32)
    # Reshape actions to (horizon, action_dim) -> flatten to (horizon * action_dim,) for stacking
    # We'll reshape properly when concatenating
    return {
        "state": state,
        "actions": actions,
    }


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
    batch_size: int = 100,
):
    """Compute normalization statistics with parallel data loading.

    Args:
        config_name: Name of the training config.
        max_frames: Maximum number of frames to process. If None, process all frames.
        num_workers: Number of parallel workers. Defaults to min(32, cpu_count).
        batch_size: Number of samples per batch sent to each worker.
    """
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    indices = list(range(num_frames))

    if max_frames is not None and max_frames < num_frames:
        # Random sample if max_frames specified
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(num_frames, size=max_frames, replace=False).tolist()
        num_frames = max_frames

    # Determine number of workers
    if num_workers is None:
        num_workers = min(32, os.cpu_count() or 1)

    print(f"Processing {num_frames} frames with {num_workers} workers (batch_size={batch_size})")

    # Split indices into batches
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    # Collect all data in parallel
    all_states = []
    all_actions = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(config_name,),
    ) as executor:
        # Submit all batches
        futures = {executor.submit(_load_batch, batch): i for i, batch in enumerate(batches)}

        # Collect results with progress bar
        with tqdm.tqdm(total=len(batches), desc="Loading data") as pbar:
            for future in as_completed(futures):
                result = future.result()
                all_states.append(result["state"])
                all_actions.append(result["actions"])
                pbar.update(1)

    print("Concatenating arrays...")
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
