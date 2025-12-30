#!/usr/bin/env python3
"""
Split Pi0 processed put_object_cabinet episodes into 3 sub-episodes based on arm movements.

The put_object_cabinet task has three subtasks:
  - Subtask 0: Grasp the object with the correct arm (first arm to move)
  - Subtask 1: Open the drawer with the other arm
  - Subtask 2: Move the object into the drawer (with the first arm again)

Splitting strategy:
  Since we only have arm positions and camera videos (no object/drawer state), we detect
  subtask boundaries by analyzing arm movement patterns:

  1. The first arm to move is the arm used in subtask 0 (grasping object)
  2. Split 1 (end of subtask 0): The first time the OTHER arm starts moving
  3. Split 2 (end of subtask 1): The first time the FIRST arm moves again, after split 1
  4. The rest is subtask 2 (placing object in drawer)

Input dataset format (per episode):
- episode_{i}/episode_{i}.hdf5
- episode_{i}/instructions.json

HDF5 schema (processed Pi0 format):
- /action: (T, 14) float32
- /observations/qpos: (T, 14) float32
- /observations/left_arm_dim: (T,) int  (typically 6, so index 6 is left gripper)
- /observations/right_arm_dim: (T,) int (typically 6, so index 13 is right gripper)
- /observations/images/cam_high: (T,) bytes (jpeg)
- /observations/images/cam_left_wrist: (T,) bytes (jpeg)
- /observations/images/cam_right_wrist: (T,) bytes (jpeg)

Output dataset format:
- out_root/episode_{k}/episode_{k}.hdf5
- out_root/episode_{k}/instructions.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np


# Known object names from put_object_cabinet task (from envs/put_object_cabinet.py)
KNOWN_OBJECTS = [
    "mouse", "stapler", "toycar", "rubikscube", "bread",
    "phone", "playingcards", "tea-box", "coffee-box", "soap",
    # Also include common variants
    "rubik's cube", "toy car", "playing cards", "tea box", "coffee box",
]


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


def _extract_object_description(instructions: List[str]) -> str:
    """
    Extract the full object description from the original instructions.

    Based on description/task_instruction/put_object_cabinet.json, the prompts use:
    - {A} = object description (e.g., "rubikscube featuring white, green, orange tiles")
    - {B} = cabinet description (e.g., "the gray and wood cabinet")
    - {a} = arm that opens the drawer

    Common patterns:
    - "place {A} inside" -> extract what's being placed
    - "put {A} into" -> extract what's being put
    - "set {A} inside" -> extract what's being set
    - "drop {A} into" -> extract what's being dropped
    - "stick {A} inside" -> extract what's being stuck
    - "move {A} into" -> extract what's being moved
    - "transfer {A} into" -> extract what's being transferred

    Returns the extracted object description or "object" if not found.
    """
    # Patterns that capture the object being placed into the drawer
    # The object appears BEFORE "inside/into/in" and AFTER action verbs
    patterns = [
        # "place/put/set/drop/stick/move/transfer the X inside/into/in"
        r'(?:place|put|set|drop|stick|move|transfer|insert)\s+(?:the\s+)?(.+?)\s+(?:inside|into|in\s+)',
        # "the X inside" at end of sentence
        r'(?:place|put|set|drop|stick|move|transfer|insert)\s+(?:the\s+)?(.+?)\s+inside\s*[.,]?$',
        # "{A} into {B}'s drawer" pattern
        r'(?:place|put|set|drop|stick|move|transfer|insert)\s+(.+?)\s+into\s+(?:the\s+drawer|it)',
    ]

    for instr in instructions:
        for pattern in patterns:
            match = re.search(pattern, instr, re.IGNORECASE)
            if match:
                obj_desc = match.group(1).strip()
                # Clean up: remove trailing punctuation
                obj_desc = re.sub(r'[.,;:!?]+$', '', obj_desc)
                # Don't return if it's too short or just "it"
                if len(obj_desc) > 2 and obj_desc.lower() != "it":
                    return obj_desc

    # Fallback: try to find known objects
    for instr in instructions:
        instr_lower = instr.lower()
        for obj in KNOWN_OBJECTS:
            if obj in instr_lower:
                # Return the canonical name
                if obj in ["rubik's cube", "rubikscube"]:
                    return "rubikscube"
                if obj in ["toy car", "toycar"]:
                    return "toycar"
                if obj in ["playing cards", "playingcards"]:
                    return "playingcards"
                if obj in ["tea box", "tea-box"]:
                    return "tea-box"
                if obj in ["coffee box", "coffee-box"]:
                    return "coffee-box"
                return obj

    return "object"


def _save_instructions(out_path: Path, instructions: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"instructions": instructions}, f, indent=2, ensure_ascii=False)


def _compute_arm_velocities(qpos: np.ndarray, left_arm_dim: int, right_arm_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity magnitude for left and right arms (joints only, excluding gripper).
    Returns smoothed velocity signals.
    """
    # Left arm: indices 0 to left_arm_dim-1 (excluding gripper at left_arm_dim)
    left_joints = qpos[:, :left_arm_dim]
    # Right arm: indices left_arm_dim+1 to left_arm_dim+1+right_arm_dim-1
    right_start = left_arm_dim + 1
    right_joints = qpos[:, right_start:right_start + right_arm_dim]

    # Compute velocities (difference between consecutive frames)
    left_vel = np.diff(left_joints, axis=0)
    right_vel = np.diff(right_joints, axis=0)

    # Compute magnitude
    left_vel_mag = np.linalg.norm(left_vel, axis=1)
    right_vel_mag = np.linalg.norm(right_vel, axis=1)

    # Pad to match original length
    left_vel_mag = np.concatenate([[0], left_vel_mag])
    right_vel_mag = np.concatenate([[0], right_vel_mag])

    # Smooth with moving average
    window = 5
    left_vel_smooth = np.convolve(left_vel_mag, np.ones(window)/window, mode='same')
    right_vel_smooth = np.convolve(right_vel_mag, np.ones(window)/window, mode='same')

    return left_vel_smooth, right_vel_smooth


def _find_first_movement(vel: np.ndarray, threshold: float, start_from: int = 0) -> Optional[int]:
    """
    Find the first frame where velocity exceeds threshold, starting from start_from.
    Returns the frame index or None if not found.
    """
    for i in range(start_from, len(vel)):
        if vel[i] > threshold:
            return i
    return None


def _find_split_points(qpos: np.ndarray, left_arm_dim: int, right_arm_dim: int,
                       velocity_threshold: float = 0.002, verbose: bool = False) -> Tuple[Optional[int], Optional[int], str]:
    """
    Find the two split points based on arm movement patterns.

    New criteria:
    1. The first arm to move is the arm used in subtask 0
    2. Split 1 (end of subtask 0): The first time the OTHER arm starts moving
    3. Split 2 (end of subtask 1): The first time the FIRST arm moves again, after split 1
    4. The rest is subtask 2

    Returns:
        (split1, split2, which_arm_first)
        - split1: first time second arm moves (subtask 0 -> 1)
        - split2: first time first arm moves again after split1 (subtask 1 -> 2)
        - which_arm_first: "left" or "right"
    """
    left_vel, right_vel = _compute_arm_velocities(qpos, left_arm_dim, right_arm_dim)

    if verbose:
        print(f"  Left vel range: [{left_vel.min():.4f}, {left_vel.max():.4f}]")
        print(f"  Right vel range: [{right_vel.min():.4f}, {right_vel.max():.4f}]")

    # Find when each arm first moves
    left_first = _find_first_movement(left_vel, velocity_threshold)
    right_first = _find_first_movement(right_vel, velocity_threshold)

    if verbose:
        print(f"  Left first movement: {left_first}")
        print(f"  Right first movement: {right_first}")

    # Determine which arm moves first
    if left_first is None and right_first is None:
        return None, None, "none"

    if left_first is None:
        first_arm = "right"
        first_vel = right_vel
        second_vel = left_vel
    elif right_first is None:
        first_arm = "left"
        first_vel = left_vel
        second_vel = right_vel
    elif left_first <= right_first:
        first_arm = "left"
        first_vel = left_vel
        second_vel = right_vel
    else:
        first_arm = "right"
        first_vel = right_vel
        second_vel = left_vel

    if verbose:
        print(f"  First arm to move: {first_arm}")

    # Split 1: First time the SECOND arm moves (end of subtask 0)
    split1 = _find_first_movement(second_vel, velocity_threshold)

    if split1 is None:
        if verbose:
            print(f"  Could not find split1: second arm never moves")
        return None, None, first_arm

    if verbose:
        print(f"  Split 1 (second arm starts): {split1}")

    # Split 2: First time the FIRST arm moves again, after split1 (end of subtask 1)
    split2 = _find_first_movement(first_vel, velocity_threshold, start_from=split1)

    if split2 is None:
        if verbose:
            print(f"  Could not find split2: first arm doesn't move after split1")
        return split1, None, first_arm

    if verbose:
        print(f"  Split 2 (first arm resumes): {split2}")

    return split1, split2, first_arm


def _default_subtask_instructions(subtask_idx: int, first_arm: str = "left", object_name: str = "object") -> List[str]:
    """Generate subtask-specific instructions based on the rubric."""
    second_arm = "right" if first_arm == "left" else "left"

    if subtask_idx == 0:
        return [
            f"Use your {first_arm} arm to grasp the {object_name} on the table.",
            f"Pick up the {object_name} with your {first_arm} arm.",
            f"Grasp the {object_name} using the {first_arm} arm.",
        ]
    if subtask_idx == 1:
        return [
            f"Use your {second_arm} arm to open the cabinet drawer.",
            f"Open the drawer of the cabinet with your {second_arm} arm.",
            f"Pull open the cabinet drawer using your {second_arm} arm.",
        ]
    if subtask_idx == 2:
        return [
            f"Move the {object_name} into the open drawer and release it.",
            f"Place the {object_name} inside the drawer.",
            f"Put the {object_name} into the drawer.",
        ]
    raise ValueError(f"Invalid subtask_idx: {subtask_idx}")


def _slice_arrays(arrays: dict, start: int, end: int) -> dict:
    """Slice all arrays from start to end (exclusive)."""
    return {
        "action": arrays["action"][start:end],
        "qpos": arrays["qpos"][start:end],
        "left_arm_dim": arrays["left_arm_dim"][start:end],
        "right_arm_dim": arrays["right_arm_dim"][start:end],
        "cam_high": arrays["cam_high"][start:end],
        "cam_left_wrist": arrays["cam_left_wrist"][start:end],
        "cam_right_wrist": arrays["cam_right_wrist"][start:end],
    }


def analyze_episode(ep: EpisodePaths, velocity_threshold: float = 0.01, verbose: bool = True) -> dict:
    """Analyze a single episode and return split information."""
    arrays = _read_episode_arrays(ep.hdf5_path)
    qpos = arrays["qpos"]
    left_arm_dim = int(np.asarray(arrays["left_arm_dim"])[0])
    right_arm_dim = int(np.asarray(arrays["right_arm_dim"])[0])

    split1, split2, first_arm = _find_split_points(
        qpos, left_arm_dim, right_arm_dim,
        velocity_threshold=velocity_threshold, verbose=verbose
    )

    # Extract object description from instructions
    orig_instructions = _load_instructions(ep.instructions_path)
    object_desc = _extract_object_description(orig_instructions)

    return {
        "episode_id": ep.episode_id,
        "total_frames": len(qpos),
        "split1": split1,
        "split2": split2,
        "first_arm": first_arm,
        "left_arm_dim": left_arm_dim,
        "right_arm_dim": right_arm_dim,
        "object_desc": object_desc,
    }


def generate_visualization_video(
    ep: EpisodePaths,
    output_path: Path,
    split1: Optional[int],
    split2: Optional[int],
    velocity_threshold: float = 0.01,
    fps: int = 25
) -> None:
    """Generate a video with split points marked."""
    import cv2
    import io
    from PIL import Image

    arrays = _read_episode_arrays(ep.hdf5_path)
    qpos = arrays["qpos"]
    left_arm_dim = int(np.asarray(arrays["left_arm_dim"])[0])
    right_arm_dim = int(np.asarray(arrays["right_arm_dim"])[0])

    # Compute velocities for overlay
    left_vel, right_vel = _compute_arm_velocities(qpos, left_arm_dim, right_arm_dim)

    # Get first frame to determine video size
    first_img = Image.open(io.BytesIO(arrays["cam_high"][0]))
    width, height = first_img.size

    # Add space for velocity plot at bottom
    plot_height = 150
    total_height = height + plot_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, total_height))

    max_vel = max(left_vel.max(), right_vel.max(), velocity_threshold * 2)

    for t in range(len(arrays["cam_high"])):
        # Decode image
        img = Image.open(io.BytesIO(arrays["cam_high"][t]))
        frame = np.array(img)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add subtask overlay
        subtask = 0
        if split1 is not None and t >= split1:
            subtask = 1
        if split2 is not None and t >= split2:
            subtask = 2

        # Draw subtask label
        label = f"Subtask {subtask} | Frame {t}/{len(qpos)}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw split indicators
        if split1 is not None:
            color = (0, 0, 255) if t == split1 else (100, 100, 255)
            cv2.putText(frame, f"Split1: {split1}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if split2 is not None:
            color = (255, 0, 0) if t == split2 else (255, 100, 100)
            cv2.putText(frame, f"Split2: {split2}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Create velocity plot
        plot = np.ones((plot_height, width, 3), dtype=np.uint8) * 255

        # Draw velocity curves
        for i in range(1, len(left_vel)):
            x1 = int((i-1) / len(left_vel) * width)
            x2 = int(i / len(left_vel) * width)
            # Left arm (blue)
            y1 = plot_height - int(left_vel[i-1] / max_vel * (plot_height - 20))
            y2 = plot_height - int(left_vel[i] / max_vel * (plot_height - 20))
            cv2.line(plot, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # Right arm (red)
            y1 = plot_height - int(right_vel[i-1] / max_vel * (plot_height - 20))
            y2 = plot_height - int(right_vel[i] / max_vel * (plot_height - 20))
            cv2.line(plot, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # Draw current position
        x_pos = int(t / len(qpos) * width)
        cv2.line(plot, (x_pos, 0), (x_pos, plot_height), (0, 255, 0), 2)

        # Draw split lines
        if split1 is not None:
            x_split1 = int(split1 / len(qpos) * width)
            cv2.line(plot, (x_split1, 0), (x_split1, plot_height), (0, 0, 255), 2)
        if split2 is not None:
            x_split2 = int(split2 / len(qpos) * width)
            cv2.line(plot, (x_split2, 0), (x_split2, plot_height), (255, 0, 0), 2)

        # Draw threshold line
        y_thresh = plot_height - int(velocity_threshold / max_vel * (plot_height - 20))
        cv2.line(plot, (0, y_thresh), (width, y_thresh), (0, 150, 0), 1)

        # Add legend
        cv2.putText(plot, "Blue: Left arm", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(plot, "Red: Right arm", (120, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Combine frame and plot
        combined = np.vstack([frame, plot])
        out.write(combined)

    out.release()
    print(f"  Saved video: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split put_object_cabinet episodes into subtasks")
    parser.add_argument(
        "--in-root",
        type=str,
        default="training_data/pi0_baseline/put_object_cabinet-baseline-500",
        help="Input dataset root",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="training_data/pi0_baseline/put_object_cabinet-split-500",
        help="Output dataset root",
    )
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=0.002,
        help="Velocity threshold for detecting arm movement (default: 0.002)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of timesteps to overlap between consecutive subtasks (default: 50 = 1s at 50Hz)",
    )
    parser.add_argument(
        "--last-to-end",
        action="store_true",
        default=True,
        help="Subtask 2 extends to the end of the episode",
    )
    parser.add_argument(
        "--preserve-episode-ids",
        action="store_true",
        help="Name output episodes as episode_{orig}_{sub} instead of global index",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Process only first N episodes (for debugging)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze and print split points without writing output",
    )
    parser.add_argument(
        "--generate-videos",
        action="store_true",
        help="Generate visualization videos with split points marked",
    )
    parser.add_argument(
        "--video-out-dir",
        type=str,
        default="split_visualization_videos",
        help="Directory for visualization videos",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default="default",
        choices=["default", "copy"],
        help="Instruction generation: 'default' uses subtask-specific prompts; 'copy' copies original",
    )
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    video_out_dir = Path(args.video_out_dir)

    episodes = _iter_episode_dirs(in_root)
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    print(f"Found {len(episodes)} episodes in {in_root}")

    if args.analyze_only or args.generate_videos:
        # Analysis mode
        for ep in episodes:
            print(f"\nAnalyzing episode_{ep.episode_id}:")
            info = analyze_episode(ep, velocity_threshold=args.velocity_threshold, verbose=True)
            print(f"  Total frames: {info['total_frames']}")
            print(f"  First arm: {info['first_arm']}")
            print(f"  Split points: {info['split1']}, {info['split2']}")
            print(f"  Object: {info['object_desc']}")

            if args.generate_videos:
                video_path = video_out_dir / f"episode_{ep.episode_id}_splits.mp4"
                generate_visualization_video(
                    ep, video_path,
                    info['split1'], info['split2'],
                    velocity_threshold=args.velocity_threshold
                )
        return

    # Splitting mode
    out_root.mkdir(parents=True, exist_ok=True)
    global_out_idx = 0
    bad: List[Tuple[int, str]] = []

    for ep in episodes:
        print(f"\nProcessing episode_{ep.episode_id}:")
        arrays = _read_episode_arrays(ep.hdf5_path)
        qpos = arrays["qpos"]
        left_arm_dim = int(np.asarray(arrays["left_arm_dim"])[0])
        right_arm_dim = int(np.asarray(arrays["right_arm_dim"])[0])

        split1, split2, first_arm = _find_split_points(
            qpos, left_arm_dim, right_arm_dim,
            velocity_threshold=args.velocity_threshold, verbose=True
        )

        if split1 is None or split2 is None:
            bad.append((ep.episode_id, f"Could not find split points: split1={split1}, split2={split2}"))
            continue

        # Define segments (all extend to end)
        T = qpos.shape[0]
        ov = args.overlap

        segments = [
            (0, T),  # Subtask 0: from start to end
            (max(0, split1 - ov), T),  # Subtask 1: from split1 to end
            (max(0, split2 - ov), T),  # Subtask 2: from split2 to end
        ]

        print(f"  Split points: {split1}, {split2}")
        print(f"  Segments: {segments}")

        # Validate segments
        if any(e <= s for s, e in segments):
            bad.append((ep.episode_id, f"Invalid segments: {segments}"))
            continue

        orig_instructions = _load_instructions(ep.instructions_path)

        # Extract object description from original instructions
        object_name = _extract_object_description(orig_instructions)
        print(f"  Extracted object: {object_name}")

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
                _save_instructions(out_instr, _default_subtask_instructions(sub_idx, first_arm, object_name))

            global_out_idx += 1

        print(f"  -> Wrote 3 sub-episodes")

    if bad:
        print("\n" + "="*60)
        print(f"WARNING: {len(bad)} episodes were skipped due to split issues:")
        for ep_id, reason in bad[:50]:
            print(f"  episode_{ep_id}: {reason}")
        if len(bad) > 50:
            print(f"  ... and {len(bad) - 50} more")
        raise SystemExit(2)

    print(f"\nDone. Created {global_out_idx} sub-episodes in {out_root}")


if __name__ == "__main__":
    main()