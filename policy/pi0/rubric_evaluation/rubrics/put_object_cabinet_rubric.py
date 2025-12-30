"""
Hierarchical subtask rubric for put_object_cabinet.

RUBRIC_SUMMARY (parsed by server; keep this block up to date)
- Goal: complete put_object_cabinet task with hierarchical prompting.
- THREE subtasks with state machine transitions:
  - Subtask 0: Grasp the object with the correct arm
  - Subtask 1: Open the drawer with the other arm
  - Subtask 2: Move the object into the drawer
- Each subtask has a focused prompt to guide the model.
- Transitions are based on observable state (gripper near object + closed, drawer open, etc.)
- Debug overlay: subtask state, transition conditions, distances, gripper states.

Motivation:
- VLA models trained on successful demonstrations cannot recover from failures.
- Hierarchical prompting provides focused instructions for each subtask.
- This allows studying whether decomposed prompts improve success rate.

Subtask Completion Criteria:
- Subtask 0 (grasp object): gripper of correct arm is near object AND closed
- Subtask 1 (open drawer): drawer joint position > threshold (drawer is open)
- Subtask 2 (place in drawer): object at target AND gripper open (same as final success)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np


# Subtask prompt templates - focused on specific actions
# These are designed based on observed failure modes in baseline evaluation

SUBTASK_0_PROMPTS = [
    "Use your {grasp_arm} arm to grasp the {object_name} on the table.",
    "Pick up the {object_name} with your {grasp_arm} arm.",
    "Grasp the {object_name} using the {grasp_arm} arm.",
    "Use the {grasp_arm} arm to pick up the {object_name}.",
    "With your {grasp_arm} arm, grasp the {object_name}.",
]

SUBTASK_1_PROMPTS = [
    "Use your {drawer_arm} arm to open the cabinet drawer.",
    "Open the drawer of the cabinet with your {drawer_arm} arm.",
    "Pull open the cabinet drawer using your {drawer_arm} arm.",
    "With your {drawer_arm} arm, open the cabinet drawer.",
    "Use the {drawer_arm} arm to pull the drawer open.",
]

SUBTASK_2_PROMPTS = [
    "Move the {object_name} into the open drawer and release it.",
    "Place the {object_name} inside the drawer.",
    "Put the {object_name} into the drawer.",
    "Drop the {object_name} into the open drawer.",
    "Release the {object_name} into the drawer.",
]


@dataclass
class RubricConfig:
    # Subtask 0: gripper near object threshold
    grasp_dist_threshold: float = 0.08  # gripper must be within 8cm of object

    # Subtask 1: drawer open threshold (joint position)
    drawer_open_threshold: float = 0.12  # drawer pulled out ~12cm (4 pulls of 0.04m each in demo)

    # Subtask 2 (final): object placement tolerance
    eps_xy: Tuple[float, float] = (0.05, 0.05)
    z_min: float = 0.007
    z_max: float = 0.12


@dataclass
class RubricState:
    """State machine for subtask progression."""
    subtask: int = 0  # Current subtask: 0=grasp, 1=open_drawer, 2=place
    prompt: str = ""
    object_name: str = ""
    grasp_arm: str = ""  # Arm that grasps the object
    drawer_arm: str = ""  # Arm that opens the drawer (opposite of grasp_arm)
    initialized: bool = False


def _format_object_name(model_name: str) -> str:
    """Convert model name like '047_mouse' to readable 'mouse'."""
    parts = model_name.split("_")
    if len(parts) > 1 and parts[0].isdigit():
        return " ".join(parts[1:])
    return model_name.replace("_", " ")


def _get_object_xyz(env: Any) -> np.ndarray:
    """Get current object position (x, y, z)."""
    return np.asarray(env.object.get_pose().p[:3], dtype=np.float32)


def _get_ee_pos(env: Any, arm: str) -> np.ndarray:
    """Get end effector position for specified arm."""
    if arm == "left":
        pose = env.robot.get_left_ee_pose()
    else:
        pose = env.robot.get_right_ee_pose()
    return np.asarray(pose[:3], dtype=np.float32)


def _get_gripper_close(env: Any, arm: str) -> bool:
    """Check if specified arm's gripper is closed."""
    if arm == "left":
        return bool(env.robot.is_left_gripper_close())
    else:
        return bool(env.robot.is_right_gripper_close())


def _get_gripper_open(env: Any, arm: str) -> bool:
    """Check if specified arm's gripper is open."""
    if arm == "left":
        return bool(env.robot.is_left_gripper_open())
    else:
        return bool(env.robot.is_right_gripper_open())


def _get_drawer_qpos(env: Any) -> float:
    """Get cabinet drawer joint position (how far it's pulled out)."""
    qpos = env.cabinet.get_qpos()
    # Cabinet has one degree of freedom (drawer slide)
    return float(qpos[0]) if len(qpos) > 0 else 0.0


def _get_target_xyz(env: Any) -> np.ndarray:
    """Get cabinet drawer functional point (target position for object)."""
    target = env.cabinet.get_functional_point(0)
    return np.asarray(target[:3], dtype=np.float32)


def _get_origin_z(env: Any) -> float:
    """Get the original z position of the object."""
    return float(getattr(env, "origin_z", 0.0))


def _within_eps_xy(xy: np.ndarray, target_xy: np.ndarray, eps_xy: Tuple[float, float]) -> bool:
    """Check if xy position is within tolerance of target."""
    d = np.abs(xy - target_xy)
    return bool((d[0] < eps_xy[0]) and (d[1] < eps_xy[1]))


def _generate_prompt(state: RubricState, subtask: int) -> str:
    """Generate prompt for specified subtask."""
    if subtask == 0:
        template = np.random.choice(SUBTASK_0_PROMPTS)
        return template.format(
            grasp_arm=state.grasp_arm,
            object_name=state.object_name
        )
    elif subtask == 1:
        template = np.random.choice(SUBTASK_1_PROMPTS)
        return template.format(drawer_arm=state.drawer_arm)
    else:  # subtask == 2
        template = np.random.choice(SUBTASK_2_PROMPTS)
        return template.format(object_name=state.object_name)


def reset() -> RubricState:
    """Reset rubric state - will be fully initialized in first step() call."""
    return RubricState()


def step(env: Any, observation: Dict[str, Any], state: RubricState, cfg: RubricConfig | None = None) -> Dict[str, Any]:
    """
    Called every simulator step. Manages subtask state machine and generates appropriate prompts.

    Returns:
      {
        "prompt": str,              # subtask-specific focused prompt
        "subtask_state": int,       # current subtask (0, 1, or 2)
        "done": bool,               # task fully complete
        "debug": { ... },           # overlay-friendly debug info
      }
    """
    cfg = cfg or RubricConfig()

    # Initialize state on first step (when env has object info)
    if not state.initialized:
        object_model_name = getattr(env, "selected_modelname", "object")
        state.object_name = _format_object_name(object_model_name)

        # Determine arm assignments based on object position
        arm_tag = str(getattr(env, "arm_tag", "left"))
        state.grasp_arm = arm_tag  # Arm on same side as object grasps it
        state.drawer_arm = "right" if arm_tag == "left" else "left"  # Other arm opens drawer

        # Generate initial prompt for subtask 0
        state.prompt = _generate_prompt(state, 0)
        state.subtask = 0
        state.initialized = True

    # Get current state observations
    object_xyz = _get_object_xyz(env)
    target_xyz = _get_target_xyz(env)
    origin_z = _get_origin_z(env)

    grasp_ee_pos = _get_ee_pos(env, state.grasp_arm)
    drawer_ee_pos = _get_ee_pos(env, state.drawer_arm)

    grasp_gripper_close = _get_gripper_close(env, state.grasp_arm)
    grasp_gripper_open = _get_gripper_open(env, state.grasp_arm)

    drawer_qpos = _get_drawer_qpos(env)

    # Compute distances
    grasp_dist = np.linalg.norm(grasp_ee_pos - object_xyz)
    dist_xy = np.abs(object_xyz[:2] - target_xyz[:2])
    z_lift = object_xyz[2] - origin_z

    # Subtask completion conditions
    subtask_0_complete = grasp_gripper_close and grasp_dist < cfg.grasp_dist_threshold
    subtask_1_complete = drawer_qpos > cfg.drawer_open_threshold

    xy_at_target = _within_eps_xy(object_xyz[:2], target_xyz[:2], cfg.eps_xy)
    z_valid = (z_lift > cfg.z_min) and (z_lift < cfg.z_max)
    subtask_2_complete = xy_at_target and z_valid and grasp_gripper_open

    # State machine transitions
    prev_subtask = state.subtask

    if state.subtask == 0 and subtask_0_complete:
        state.subtask = 1
        state.prompt = _generate_prompt(state, 1)
    elif state.subtask == 1 and subtask_1_complete:
        state.subtask = 2
        state.prompt = _generate_prompt(state, 2)

    # Check if fully done
    done = subtask_2_complete

    debug = {
        "subtask_state": state.subtask,
        "prev_subtask": prev_subtask,
        "focus": f"subtask_{state.subtask}",

        # Subtask 0 info
        "grasp_arm": state.grasp_arm,
        "grasp_dist": float(grasp_dist),
        "grasp_dist_threshold": float(cfg.grasp_dist_threshold),
        "grasp_gripper_close": bool(grasp_gripper_close),
        "subtask_0_complete": bool(subtask_0_complete),

        # Subtask 1 info
        "drawer_arm": state.drawer_arm,
        "drawer_qpos": float(drawer_qpos),
        "drawer_open_threshold": float(cfg.drawer_open_threshold),
        "subtask_1_complete": bool(subtask_1_complete),

        # Subtask 2 info
        "object_xyz": [float(x) for x in object_xyz],
        "target_xyz": [float(x) for x in target_xyz],
        "dist_xy": [float(dist_xy[0]), float(dist_xy[1])],
        "z_lift": float(z_lift),
        "xy_at_target": bool(xy_at_target),
        "z_valid": bool(z_valid),
        "grasp_gripper_open": bool(grasp_gripper_open),
        "subtask_2_complete": bool(subtask_2_complete),

        # General info
        "object_name": state.object_name,
        "eps_xy": list(cfg.eps_xy),
        "z_range": [float(cfg.z_min), float(cfg.z_max)],
    }

    return {
        "prompt": state.prompt,
        "subtask_state": state.subtask,
        "done": done,
        "debug": debug,
    }