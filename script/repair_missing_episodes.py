#!/usr/bin/env python
# repair_missing_episodes.py

import os
import sys
import yaml
import json
sys.path.append("./")

def repair_missing_episodes(task_name, task_config, gpu_id=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import importlib
    from envs import CONFIGS_PATH

    def class_decorator(task_name):
        envs_module = importlib.import_module(f"envs.{task_name}")
        env_class = getattr(envs_module, task_name)
        return env_class()

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    # Load config
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    save_path = os.path.join(args["save_path"], task_name, task_config)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise Exception("missing embodiment files")
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

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = save_path
    args["need_plan"] = True

    # Load seeds
    seed_file = os.path.join(save_path, "seed.txt")
    with open(seed_file, "r") as f:
        seed_list = [int(s) for s in f.read().split()]

    episode_num = args["episode_num"]
    traj_path = os.path.join(save_path, "_traj_data")

    # Find missing pkl files
    missing_episodes = []
    for i in range(episode_num):
        pkl_file = os.path.join(traj_path, f"episode{i}.pkl")
        if not os.path.exists(pkl_file):
            missing_episodes.append(i)

    if not missing_episodes:
        print("\033[92mNo missing episodes found!\033[0m")
        return

    print(f"\033[93mFound {len(missing_episodes)} missing episodes: {missing_episodes}\033[0m")

    TASK_ENV = class_decorator(task_name)

    for episode_idx in missing_episodes:
        seed = seed_list[episode_idx]
        print(f"\033[34mRepairing episode {episode_idx} with seed {seed}...\033[0m")

        try:
            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
            TASK_ENV.play_once()

            if TASK_ENV.plan_success and TASK_ENV.check_success():
                TASK_ENV.save_traj_data(episode_idx)
                print(f"\033[92mEpisode {episode_idx} repaired successfully\033[0m")
            else:
                print(f"\033[91mEpisode {episode_idx} failed planning/check - seed may be bad\033[0m")

            TASK_ENV.close_env()
            if args.get("render_freq"):
                TASK_ENV.viewer.close()

        except Exception as e:
            print(f"\033[91mError repairing episode {episode_idx}: {e}\033[0m")
            try:
                TASK_ENV.close_env()
            except:
                pass


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    repair_missing_episodes(args.task_name, args.task_config, args.gpu_id)
