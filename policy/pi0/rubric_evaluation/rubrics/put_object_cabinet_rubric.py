"""
Baseline rubric for put_object_cabinet (no staging).

RUBRIC_SUMMARY (parsed by server; keep this block up to date)
- Goal: complete put_object_cabinet task (open drawer with one arm, place object inside with other arm).
- NO staging / state machine - randomly samples one prompt per episode.
- Completion condition: object is inside the drawer (within xy tolerance) AND holding arm gripper is open.
- Prompting: randomly selects from unseen prompts at reset (like standard eval.sh).
- Debug overlay: object position, target position, distances, drawer state, gripper states.

Notes:
- This is a baseline rubric for comparison with staged/hierarchical prompting.
- Uses ground-truth simulator state (object pose, cabinet functional point) and robot gripper state.
- The task involves bimanual coordination: one arm opens the drawer, the other places the object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


# Unseen prompt templates from description/task_instruction/put_object_cabinet.json
# Template variables: {A}=object, {B}=cabinet, {a}=arm to open drawer
UNSEEN_PROMPT_TEMPLATES = [
    "Use {a} to open {B}'s drawer and the other arm to place {A} inside.",
    "Open {B}'s drawer with {a} and use the other arm to put {A} inside.",
    "Open the drawer of {B} with {a} and place {A} inside using the other arm.",
    "Use {a} to pull open {B}'s drawer and the other arm to move {A} into it.",
    "Open {B}'s drawer with {a} and place {A} inside using the other arm.",
    "Pull out the drawer of {B} and set {A} into it.",
    "Open the drawer of {B} with {a}, then the other arm places {A} inside.",
    "Open {B}'s drawer and set {A} inside with the other arm.",
    "Open {B}'s drawer using {a} and place {A} inside with the other arm.",
    "Set {A} in the drawer of {B} after opening it with {a}.",
]


@dataclass
class RubricConfig:
    eps_xy: Tuple[float, float] = (0.05, 0.05)  # (x,y) tolerance to target position
    z_min: float = 0.007  # minimum z lift from origin (object must be lifted)
    z_max: float = 0.12   # maximum z height (object should be in drawer, not too high)


@dataclass
class RubricState:
    """State holds the randomly selected prompt and template substitutions for this episode."""
    prompt: str = ""
    object_name: str = ""
    cabinet_name: str = "cabinet"
    arm_tag: str = ""


def _get_object_xyz(env: Any) -> np.ndarray:
    """Get current object position (x, y, z)."""
    return np.asarray(env.object.get_pose().p[:3], dtype=np.float32)


def _get_target_xyz(env: Any) -> np.ndarray:
    """Get cabinet drawer functional point (target position for object)."""
    target = env.cabinet.get_functional_point(0)
    return np.asarray(target[:3], dtype=np.float32)


def _get_origin_z(env: Any) -> float:
    """Get the original z position of the object (stored during play_once)."""
    return float(getattr(env, "origin_z", 0.0))


def _within_eps_xy(xy: np.ndarray, target_xy: np.ndarray, eps_xy: Tuple[float, float]) -> bool:
    """Check if xy position is within tolerance of target."""
    d = np.abs(xy - target_xy)
    return bool((d[0] < eps_xy[0]) and (d[1] < eps_xy[1]))


def _get_gripper_open(env: Any, arm_tag: str) -> bool:
    """Check if specified arm's gripper is open."""
    if arm_tag == "left":
        return bool(env.robot.is_left_gripper_open())
    else:
        return bool(env.robot.is_right_gripper_open())


def _format_object_name(model_name: str) -> str:
    """Convert model name like '047_mouse' to readable 'mouse'."""
    # Remove numeric prefix and underscores
    parts = model_name.split("_")
    if len(parts) > 1 and parts[0].isdigit():
        return " ".join(parts[1:])
    return model_name.replace("_", " ")


def reset() -> RubricState:
    """Reset rubric state - will be fully initialized in first step() call."""
    return RubricState()


def step(env: Any, observation: Dict[str, Any], state: RubricState, cfg: RubricConfig | None = None) -> Dict[str, Any]:
    """
    Called every simulator step.

    Returns:
      {
        "prompt": str,              # randomly selected prompt (same for entire episode)
        "subtask_state": int,       # always 0 (no staging)
        "done": bool,               # object at target AND gripper open AND z in valid range
        "debug": { ... },           # overlay-friendly debug info
      }
    """
    cfg = cfg or RubricConfig()

    # Initialize prompt on first step (when env has object info)
    if not state.prompt:
        # Get object and arm info from env
        object_model_name = getattr(env, "selected_modelname", "object")
        arm_tag = str(getattr(env, "arm_tag", "left"))

        # Format object name for natural language
        object_name = _format_object_name(object_model_name)

        # Template substitutions
        subs = {
            "{A}": f"the {object_name}",
            "{B}": "the cabinet",
            "{a}": f"{arm_tag.opposite if hasattr(arm_tag, 'opposite') else ('right' if arm_tag == 'left' else 'left')} arm",
        }

        # Handle arm_tag which might be an ArmTag object
        arm_str = str(arm_tag)
        if arm_str == "left":
            drawer_arm = "right"
        else:
            drawer_arm = "left"

        subs["{a}"] = f"{drawer_arm} arm"

        # Select and substitute prompt
        template = np.random.choice(UNSEEN_PROMPT_TEMPLATES)
        prompt = template
        for key, value in subs.items():
            prompt = prompt.replace(key, value)

        state.prompt = prompt
        state.object_name = object_name
        state.arm_tag = arm_str

    # Get current positions
    object_xyz = _get_object_xyz(env)
    target_xyz = _get_target_xyz(env)
    origin_z = _get_origin_z(env)

    # Compute distances
    dist_xy = np.abs(object_xyz[:2] - target_xyz[:2])
    z_lift = object_xyz[2] - origin_z

    # Check conditions
    xy_at_target = _within_eps_xy(object_xyz[:2], target_xyz[:2], cfg.eps_xy)
    z_valid = (z_lift > cfg.z_min) and (z_lift < cfg.z_max)

    # Get gripper state for the arm that placed the object
    arm_str = state.arm_tag if state.arm_tag else "left"
    gripper_open = _get_gripper_open(env, arm_str)

    # Task is done when object is at target xy, z is in valid range, and gripper is open
    done = xy_at_target and z_valid and gripper_open

    debug = {
        "subtask_state": 0,  # no staging
        "focus": None,       # no focus - baseline uses full task prompt
        "eps_xy": list(cfg.eps_xy),
        "object_xyz": [float(x) for x in object_xyz],
        "target_xyz": [float(x) for x in target_xyz],
        "dist_xy": [float(dist_xy[0]), float(dist_xy[1])],
        "z_lift": float(z_lift),
        "z_range": [float(cfg.z_min), float(cfg.z_max)],
        "xy_at_target": bool(xy_at_target),
        "z_valid": bool(z_valid),
        "gripper_open": bool(gripper_open),
        "arm_tag": arm_str,
        "object_name": state.object_name,
    }

    return {
        "prompt": state.prompt,
        "subtask_state": 0,  # no staging
        "done": done,
        "debug": debug,
    }