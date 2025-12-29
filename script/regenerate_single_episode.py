#!/usr/bin/env python3
"""
Regenerate a single episode HDF5 file.

Usage:
    python script/regenerate_single_episode.py put_object_cabinet demo_randomized 315 4564
"""
import os
import sys
import json
sys.path.append("./")

def main(task_name, task_config, episode_idx, seed):
    """Regenerate a single episode."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Set ASSETS_PATH for curobo configuration files
    workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ["ASSETS_PATH"] = workspace_path

    import yaml
    import importlib
    from sapien.render import clear_cache
    from envs import CONFIGS_PATH

    def class_decorator(task_name):
        envs_module = importlib.import_module(f"envs.{task_name}")
        try:
            env_class = getattr(envs_module, task_name)
            env_instance = env_class()
        except:
            raise SystemExit("No such task")
        return env_instance

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args

    # Load config
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
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

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_path = os.path.join(args["save_path"], task_name, task_config)

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = save_path
    args["need_plan"] = True  # Generate trajectory from scratch
    args["render_freq"] = 0
    args["save_data"] = True

    TASK_ENV = class_decorator(task_name)

    print(f"\033[92mRegenerating episode {episode_idx} with seed {seed}\033[0m")
    print(f"Save path: {save_path}")

    # Retry planning multiple times since it can be non-deterministic
    # GPU-based motion planning can have significant variance between machines
    max_retries = 50
    success = False

    for attempt in range(max_retries):
        print(f"\033[93mAttempt {attempt + 1}/{max_retries}\033[0m")
        try:
            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
            info = TASK_ENV.play_once()

            print(f"  plan_success={TASK_ENV.plan_success}")
            print(f"  left_joint_path length={len(TASK_ENV.left_joint_path)}")
            print(f"  right_joint_path length={len(TASK_ENV.right_joint_path)}")

            if TASK_ENV.plan_success:
                check_result = TASK_ENV.check_success()
                print(f"  check_success={check_result}")
                if check_result:
                    # Save trajectory data for future use
                    TASK_ENV.save_traj_data(episode_idx)
                    success = True
                    break
                else:
                    print(f"\033[93mAttempt {attempt + 1} planning succeeded but check_success failed\033[0m")
            else:
                print(f"\033[93mAttempt {attempt + 1} planning failed\033[0m")

            TASK_ENV.close_env()
            if args.get("render_freq"):
                TASK_ENV.viewer.close()
        except Exception as e:
            print(f"\033[91mAttempt {attempt + 1} error: {e}\033[0m")
            import traceback
            traceback.print_exc()
            try:
                TASK_ENV.close_env()
            except:
                pass

    if not success:
        print(f"\033[91mFailed to generate episode {episode_idx} after {max_retries} attempts\033[0m")
        return

    try:
        info = info  # Already have info from successful attempt

        # Save scene info
        info_file_path = os.path.join(save_path, "scene_info.json")
        if os.path.exists(info_file_path):
            with open(info_file_path, "r") as f:
                info_db = json.load(f)
        else:
            info_db = {}

        info_db[f"episode_{episode_idx}"] = info

        with open(info_file_path, "w") as f:
            json.dump(info_db, f, ensure_ascii=False, indent=4)

        TASK_ENV.close_env(clear_cache=True)
        TASK_ENV.merge_pkl_to_hdf5_video()
        TASK_ENV.remove_data_cache()

        if not TASK_ENV.check_success():
            print(f"\033[91mEpisode {episode_idx} check_success failed!\033[0m")
        else:
            print(f"\033[92mEpisode {episode_idx} completed successfully!\033[0m")

        # Verify the file was created
        hdf5_path = os.path.join(save_path, 'data', f'episode{episode_idx}.hdf5')
        if os.path.exists(hdf5_path):
            print(f"\033[92mHDF5 file created: {hdf5_path}\033[0m")
        else:
            print(f"\033[91mHDF5 file NOT found: {hdf5_path}\033[0m")

    except Exception as e:
        print(f"\033[91mError regenerating episode {episode_idx}: {e}\033[0m")
        import traceback
        traceback.print_exc()
        try:
            TASK_ENV.close_env()
        except:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script/regenerate_single_episode.py <task_name> <task_config> <episode_idx> <seed>")
        print("Example: python script/regenerate_single_episode.py put_object_cabinet demo_randomized 315 4564")
        sys.exit(1)

    task_name = sys.argv[1]
    task_config = sys.argv[2]
    episode_idx = int(sys.argv[3])
    seed = int(sys.argv[4])

    main(task_name, task_config, episode_idx, seed)