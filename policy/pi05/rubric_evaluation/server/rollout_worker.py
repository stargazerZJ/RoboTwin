from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import importlib
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse existing evaluation utilities / envs
from envs import CONFIGS_PATH  # type: ignore
from envs.utils.create_actor import UnStableError  # type: ignore

from openpi_client import websocket_client_policy  # type: ignore


@dataclass(frozen=True)
class WorkerConfig:
    task_name: str
    task_config: str
    instruction_type: str = "unseen"
    pi0_step: int = 50
    max_steps_fallback: int = 500
    rubric_variant: str = "baseline"


def _load_task_args(task_config: str) -> Dict[str, Any]:
    import yaml  # local import to keep worker import light

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_embodiment_config(robot_file: str) -> Dict[str, Any]:
    import yaml  # local import

    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _resolve_embodiment_files(args: Dict[str, Any]) -> None:
    """
    Populate the same keys that script/eval_policy.py sets before calling env.setup_demo():
      - left_robot_file, right_robot_file, dual_arm_embodied, embodiment_dis (optional)
      - left_embodiment_config, right_embodiment_config
    """
    import yaml  # local import

    embodiment_type = args.get("embodiment")
    if embodiment_type is None:
        raise KeyError("embodiment")

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(emb_type: str) -> str:
        robot_file = _embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])


def _make_env(task_name: str) -> Any:
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def _get_camera_config(camera_type: str) -> Dict[str, Any]:
    import yaml  # local import

    camera_config_path = os.path.abspath("task_config/_camera_config.yml")
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args[camera_type]


def _encode_obs(observation: Dict[str, Any]) -> Tuple[list[np.ndarray], np.ndarray]:
    # Same as policy/pi05/deploy_policy.py:encode_obs
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def _to_observation_window(input_rgb_arr: list[np.ndarray], state: np.ndarray, prompt: str) -> Dict[str, Any]:
    # Same schema as policy/pi05/pi_model.py:update_observation_window
    img_front, img_right, img_left = input_rgb_arr[0], input_rgb_arr[1], input_rgb_arr[2]
    img_front = np.transpose(img_front, (2, 0, 1))
    img_right = np.transpose(img_right, (2, 0, 1))
    img_left = np.transpose(img_left, (2, 0, 1))
    return {
        "state": state,
        "images": {
            "cam_high": img_front,
            "cam_left_wrist": img_left,
            "cam_right_wrist": img_right,
        },
        "prompt": prompt,
    }


def _save_attention_plot(data: np.ndarray, path: Path):
    try:
        if data.dtype == 'bfloat16':
            data = data.astype(np.float32)
        if data.shape[0] == 1:
            data = np.squeeze(data, axis=0)
        
        # Aggregate to 2D
        while data.ndim > 2:
            data = np.mean(data, axis=0)

        # Assumptions for PaliGemma/Pi0:
        # 3 images (224x224) -> 256 tokens each = 768 tokens total
        # followed by text prompt tokens.
        # Check if shape matches this expectation
        num_keys = data.shape[1]
        
        plt.figure(figsize=(10, 6))
        plt.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
        
        # Draw separator if realistic
        if num_keys > 768:
            plt.axvline(x=768, color='red', linestyle='--', linewidth=1, label='Image/Text Boundary')
            # Add text labels if space permits
            if num_keys < 1024: # arbitrary cutoff to avoid clutter
                plt.text(384, -2, 'Images', ha='center', va='bottom', color='red', fontsize=8)
                plt.text((768 + num_keys)/2, -2, 'Text', ha='center', va='bottom', color='red', fontsize=8)

        plt.axis('off') # Keep axis off for clean web view, but maybe add small markers?
        # For now, just the line is good help.
        
        plt.tight_layout(pad=0)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Failed to save attention plot: {e}")


def _load_rubric_module(rubric_path: Path):
    """
    Load rubric.py from an arbitrary path (versioned folder).

    Important: register the module in sys.modules before exec_module,
    otherwise dataclasses can crash when resolving type annotations.
    """
    module_name = f"rubric_module_{rubric_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, str(rubric_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load rubric module from {rubric_path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _start_ffmpeg(video_path: Path, video_size: str) -> subprocess.Popen:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )


def run_one_episode(
    *,
    backend_host: str,
    backend_port: int,
    rubric_path: Path,
    out_dir: Path,
    episode_id: int,
    seed: int,
    worker_cfg: WorkerConfig,
    client: "websocket_client_policy.WebsocketClientPolicy | None" = None,
) -> Dict[str, Any]:
    """
    Runs a single rollout episode and writes:
      - videos/episode_XXXX.mp4
      - appends one line to metrics.jsonl (handled by manager)

    Args:
        client: Optional persistent websocket client. If None, a new one is created
                (and closed) per episode. Passing a persistent client avoids rapid
                connect/disconnect cycles that can corrupt CUDA graph capture state.

    Returns a dict suitable for JSONL.
    """
    t0 = time.time()

    # Load rubric module (version-specific)
    rubric = _load_rubric_module(rubric_path)
    rubric_state = rubric.reset()
    rubric_cfg_cls = getattr(rubric, "RubricConfig", None)
    rubric_cfg = rubric_cfg_cls() if rubric_cfg_cls is not None else None
    
    if rubric_cfg is not None:
        # Inject rubric_variant if the rubric config supports it (or dynamically if it allows)
        # We assume RubricConfig might have this field, or we can set it if it's a standard class.
        # If it's a frozen dataclass, this might fail unless we pre-configure it.
        # But RubricConfig in rubric files is usually a Mutable dataclass.
        try:
            rubric_cfg.rubric_variant = worker_cfg.rubric_variant
        except Exception:
            # If we can't set it (e.g. frozen or slot), we might ignore or log.
            # But we are designing the rubrics, so we will make sure they have the field.
            pass

    # Create env and args (mostly copied from script/eval_policy.py)
    args = _load_task_args(worker_cfg.task_config)
    args["task_name"] = worker_cfg.task_name
    args["task_config"] = worker_cfg.task_config
    args["eval_mode"] = True

    # Critical: env.setup_demo expects embodiment configs populated
    _resolve_embodiment_files(args)

    # Video config: follow script/eval_policy.py behavior.
    # Let the env decide eval_video_path based on eval_video_save_dir.
    video_size = None
    if args.get("eval_video_log", True):
        args["eval_video_save_dir"] = out_dir
        camera_type = args["camera"]["head_camera_type"]
        cam_cfg = _get_camera_config(camera_type)
        video_size = f"{cam_cfg['w']}x{cam_cfg['h']}"


    env = _make_env(worker_cfg.task_name)

    # Setup demo; skip unstable seeds by raising to manager
    try:
        env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **args)
    except UnStableError:
        env.close_env()
        raise

    # Instruction: we will override with rubric prompt each step, but env still expects an instruction set once.
    env.set_instruction(instruction="Rank the blocks by color: blue left, green middle, red right.")

    # Start video (only if env actually configured a path)
    ffmpeg = None
    video_path = out_dir / "videos" / f"episode_{episode_id:04d}.mp4"
    if video_size is not None and getattr(env, "eval_video_path", None) is not None:
        # Match baseline naming convention but keep our folder structure.
        ffmpeg = _start_ffmpeg(video_path, video_size)
        env._set_eval_video_ffmpeg(ffmpeg)

    # Policy client: use provided persistent client or create temporary one
    owns_client = client is None
    if owns_client:
        client = websocket_client_policy.WebsocketClientPolicy(host=backend_host, port=backend_port)

    # Rollout loop
    succ = False
    step_count = 0
    subtask_state = 0
    debug_last: Dict[str, Any] = {}
    prompt = ""

    try:
        while env.take_action_cnt < getattr(env, "step_lim", worker_cfg.max_steps_fallback):
            observation = env.get_obs()

            # Rubric decides prompt based on env + observation + internal state
            r = rubric.step(env, observation, rubric_state, rubric_cfg)
            prompt = r["prompt"]
            subtask_state = int(r.get("subtask_state", 0))
            debug_last = dict(r.get("debug", {}))

            # Build observation window and query policy once per chunk
            input_rgb_arr, input_state = _encode_obs(observation)
            obs_window = _to_observation_window(input_rgb_arr, input_state, prompt)
            
            response = client.infer(obs_window)
            actions = response["actions"][: worker_cfg.pi0_step]

            if "attention" in response and response["attention"] is not None:
                attention = response["attention"]
                attention_dir = out_dir / "attention" / f"episode_{episode_id:04d}"
                attention_dir.mkdir(parents=True, exist_ok=True)
                np.save(attention_dir / f"step_{step_count:04d}.npy", attention)
                _save_attention_plot(attention, attention_dir / f"step_{step_count:04d}.png")

            for action in actions:
                env.take_action(action)
                step_count += 1
                # Check success after each action (env might set eval_success internally)
                if env.check_success():
                    succ = True
                    break
            if succ:
                break

        # Final success check after all actions
        if env.check_success():
            succ = True

    finally:
        if ffmpeg is not None:
            # Ensure ffmpeg flushes and writes moov atom properly.
            try:
                env._del_eval_video_ffmpeg()
            finally:
                try:
                    if ffmpeg.stdin is not None:
                        ffmpeg.stdin.close()
                except Exception:
                    pass
                try:
                    ffmpeg.wait(timeout=5)
                except Exception:
                    ffmpeg.kill()
        env.close_env()
        # Only close client if we created it (not persistent)
        if owns_client and client is not None:
            try:
                client.close()
            except Exception:
                pass

    t1 = time.time()
    return {
        "episode_id": int(episode_id),
        "seed": int(seed),
        "success": bool(succ),
        "steps": int(step_count),
        "subtask_state_final": int(subtask_state),
        "debug_last": debug_last,
        "prompt": prompt,
        "video_path": str(video_path),
        "backend": {"host": backend_host, "port": int(backend_port)},
        "time_sec": float(t1 - t0),
    }