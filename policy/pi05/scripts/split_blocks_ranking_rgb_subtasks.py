#!/usr/bin/env python3
"""
Split Pi0 processed blocks_ranking_rgb episodes into 3 sub-episodes using gripper release events.

Input dataset format (per episode):
- episode_{i}/episode_{i}.hdf5
- episode_{i}/instructions.json

HDF5 schema (processed Pi0 format):
- /action: (T, 14) float32
- /observations/qpos: (T, 14) float32
- /observations/left_arm_dim: (T,) int
- /observations/right_arm_dim: (T,) int
- /observations/images/cam_high: (T,) bytes (jpeg)
- /observations/images/cam_left_wrist: (T,) bytes (jpeg)
- /observations/images/cam_right_wrist: (T,) bytes (jpeg)

Splitting rule:
- Compute gripper "open" boolean for left and right gripper from qpos using a threshold.
- Find "release" events: closed -> open transitions.
- Merge releases from both grippers, sort, expect exactly 3 releases.
- Define segments:
    seg0: [0, t1]
    seg1: [t1, t2]
    seg2: [t2, t3]   (default) or [t2, end] if --last-to-end

Output dataset format:
- out_root/episode_{k}/episode_{k}.hdf5
- out_root/episode_{k}/instructions.json

Where k is a running global index (0..3*N-1) unless --preserve-episode-ids is used.

Notes:
- We keep qpos/action/images aligned by slicing the same [start:end] range (end exclusive).
- We also slice left_arm_dim/right_arm_dim to keep shapes consistent.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class EpisodePaths:
    episode_dir: Path
    hdf5_path: Path
    instructions_path: Path
    episode_id: int


def _iter_episode_dirs(root: Path) -> List[EpisodePaths]:
    eps: List[EpisodePaths] = []
    for p in sorted(root.glob("episode_*")):
        if not p.is_dir():
            continue
        try:
            episode_id = int(p.name.split("_", 1)[1])
        except Exception:
            continue
        hdf5_path = p / f"episode_{episode_id}.hdf5"
        instructions_path = p / "instructions.json"
        if not hdf5_path.exists():
            raise FileNotFoundError(f"Missing HDF5: {hdf5_path}")
        if not instructions_path.exists():
            raise FileNotFoundError(f"Missing instructions.json: {instructions_path}")
        eps.append(EpisodePaths(p, hdf5_path, instructions_path, episode_id))
    if not eps:
        raise FileNotFoundError(f"No episode_* directories found under: {root}")
    return eps


def _read_episode_arrays(hdf5_path: Path) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        action = f["action"][()]
        qpos = f["observations/qpos"][()]
        left_arm_dim = f["observations/left_arm_dim"][()]
        right_arm_dim = f["observations/right_arm_dim"][()]
        cam_high = f["observations/images/cam_high"][()]
        cam_left = f["observations/images/cam_left_wrist"][()]
        cam_right = f["observations/images/cam_right_wrist"][()]

    return {
        "action": action,
        "qpos": qpos,
        "left_arm_dim": left_arm_dim,
        "right_arm_dim": right_arm_dim,
        "cam_high": cam_high,
        "cam_left_wrist": cam_left,
        "cam_right_wrist": cam_right,
    }


def _write_episode_arrays(out_hdf5_path: Path, arrays: dict) -> None:
    out_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_hdf5_path, "w") as f:
        f.create_dataset("action", data=arrays["action"])
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=arrays["qpos"])
        obs.create_dataset("left_arm_dim", data=arrays["left_arm_dim"])
        obs.create_dataset("right_arm_dim", data=arrays["right_arm_dim"])
        images = obs.create_group("images")
        # Preserve dtype/shape for bytes arrays
        images.create_dataset("cam_high", data=arrays["cam_high"])
        images.create_dataset("cam_left_wrist", data=arrays["cam_left_wrist"])
        images.create_dataset("cam_right_wrist", data=arrays["cam_right_wrist"])


def _load_instructions(instructions_path: Path) -> List[str]:
    with open(instructions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    instr = data.get("instructions")
    if not isinstance(instr, list) or not all(isinstance(x, str) for x in instr):
        raise ValueError(f"Invalid instructions.json schema: {instructions_path}")
    return instr


def _save_instructions(out_path: Path, instructions: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"instructions": instructions}, f, indent=2, ensure_ascii=False)


def _compute_release_indices(qpos: np.ndarray, left_arm_dim: int, right_arm_dim: int, threshold: float) -> List[int]:
    """
    Returns sorted list of release indices (closed->open) across both grippers.
    """
    lg = qpos[:, left_arm_dim]
    rg = qpos[:, left_arm_dim + 1 + right_arm_dim]

    l_open = lg >= threshold
    r_open = rg >= threshold

    # closed -> open at t means (not open at t-1) and (open at t)
    l_rel = np.where((~l_open[:-1]) & (l_open[1:]))[0] + 1
    r_rel = np.where((~r_open[:-1]) & (r_open[1:]))[0] + 1

    rel = np.unique(np.concatenate([l_rel, r_rel])).tolist()
    rel.sort()
    return rel


def _slice_arrays(arrays: dict, start: int, end: int) -> dict:
    # end is exclusive
    return {
        "action": arrays["action"][start:end],
        "qpos": arrays["qpos"][start:end],
        "left_arm_dim": arrays["left_arm_dim"][start:end],
        "right_arm_dim": arrays["right_arm_dim"][start:end],
        "cam_high": arrays["cam_high"][start:end],
        "cam_left_wrist": arrays["cam_left_wrist"][start:end],
        "cam_right_wrist": arrays["cam_right_wrist"][start:end],
    }


def _default_subtask_instructions(subtask_idx: int) -> List[str]:
    # Keep it simple and consistent; you can later replace with richer templates.
    if subtask_idx == 0:
        return [
            "Pick up the blue block and place it at the right position in the row.",
            "Move the blue block to the rightmost position.",
        ]
    if subtask_idx == 1:
        return [
            "Pick up the green block and place it at the middle position in the row.",
            "Move the green block to the middle position.",
        ]
    if subtask_idx == 2:
        return [
            "Pick up the red block and place it at the left position in the row.",
            "Move the red block to the leftmost position.",
        ]
    raise ValueError(f"Invalid subtask_idx: {subtask_idx}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-root",
        type=str,
        required=True,
        help="Input dataset root, e.g. policy/pi05/training_data/blocks_ranking_rgb/blocks_ranking_rgb-aloha-agilex_randomized_500-500",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Output dataset root, e.g. policy/pi05/training_data/blocks_ranking_rgb/blocks_ranking_rgb-aloha-agilex_randomized_split_500-500",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Gripper open threshold applied to qpos gripper scalar (default: 0.5).",
    )
    parser.add_argument(
        "--last-to-end",
        action="store_true",
        default=True,
        help="Subtask 3 is [t2:end] instead of [t2:t3]. (default: True)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of timesteps to overlap between consecutive subtasks (default: 50 = 1s at 50Hz).",
    )
    parser.add_argument(
        "--preserve-episode-ids",
        action="store_true",
        help="If set, output episodes are named episode_{orig}_{sub} instead of a global running index.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for debugging (process only first N episodes).",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default="default",
        choices=["default", "copy"],
        help="Instruction generation: 'default' uses fixed per-subtask instructions; 'copy' copies original instructions.json.",
    )
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    episodes = _iter_episode_dirs(in_root)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    out_root.mkdir(parents=True, exist_ok=True)

    global_out_idx = 0
    bad: List[Tuple[int, str]] = []

    for ep in episodes:
        arrays = _read_episode_arrays(ep.hdf5_path)
        qpos = arrays["qpos"]
        left_arm_dim = int(np.asarray(arrays["left_arm_dim"])[0])
        right_arm_dim = int(np.asarray(arrays["right_arm_dim"])[0])

        releases = _compute_release_indices(qpos, left_arm_dim, right_arm_dim, threshold=args.threshold)

        if len(releases) != 3:
            bad.append((ep.episode_id, f"expected 3 releases, got {len(releases)}: {releases}"))
            continue

        t1, t2, t3 = releases
        if args.last_to_end:
            segments = [(0, t1 + 1), (t1, t2 + 1), (t2, qpos.shape[0])]
        else:
            segments = [(0, t1 + 1), (t1, t2 + 1), (t2, t3 + 1)]

        # Add overlap between consecutive segments (clamped to valid range).
        ov = int(args.overlap)
        if ov < 0:
            bad.append((ep.episode_id, f"overlap must be >= 0, got {ov}"))
            continue

        seg0_s, seg0_e = segments[0]
        seg1_s, seg1_e = segments[1]
        seg2_s, seg2_e = segments[2]

        seg1_s = max(0, seg1_s - ov)
        seg2_s = max(0, seg2_s - ov)

        segments = [(seg0_s, seg0_e), (seg1_s, seg1_e), (seg2_s, seg2_e)]

        # Basic sanity: non-empty segments
        if any(e <= s for s, e in segments):
            bad.append((ep.episode_id, f"empty segment(s): {segments} from releases {releases}"))
            continue

        orig_instructions = _load_instructions(ep.instructions_path)

        for sub_idx, (s, e) in enumerate(segments):
            if args.preserve_episode_ids:
                out_ep_name = f"episode_{ep.episode_id}_{sub_idx}"
            else:
                out_ep_name = f"episode_{global_out_idx}"
            out_ep_dir = out_root / out_ep_name

            out_hdf5 = out_ep_dir / f"{out_ep_name}.hdf5"
            out_instr = out_ep_dir / "instructions.json"

            sliced = _slice_arrays(arrays, s, e)
            _write_episode_arrays(out_hdf5, sliced)

            if args.instructions == "copy":
                _save_instructions(out_instr, orig_instructions)
            else:
                _save_instructions(out_instr, _default_subtask_instructions(sub_idx))

            global_out_idx += 1

        print(f"episode_{ep.episode_id}: releases={releases} segments={segments} -> wrote 3 sub-episodes")

    if bad:
        print("----")
        print(f"WARNING: {len(bad)} episodes were skipped due to split issues:")
        for ep_id, reason in bad[:50]:
            print(f"  episode_{ep_id}: {reason}")
        if len(bad) > 50:
            print(f"  ... and {len(bad) - 50} more")
        # Non-zero exit to make failures visible in scripts
        raise SystemExit(2)

    print("Done.")


if __name__ == "__main__":
    main()