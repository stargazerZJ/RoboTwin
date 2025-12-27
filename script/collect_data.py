import os
import sys
import time
import json
import filelock
from argparse import ArgumentParser
import multiprocessing as mp
sys.path.append("./")

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def worker_seed_collection(gpu_id, task_name, task_config, save_path, target_count):
    """Worker for parallel seed collection - GPU set before imports"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import after setting GPU
    import yaml
    import importlib
    import traceback
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

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = save_path
    args["need_plan"] = True

    TASK_ENV = class_decorator(task_name)

    lock_file = os.path.join(save_path, "seed_collection.lock")
    state_file = os.path.join(save_path, "seed_state.json")
    seed_file = os.path.join(save_path, "seed.txt")

    print(f"\033[92m[GPU {gpu_id}] Seed collection worker started\033[0m")

    while True:
        # Atomically get next seed to try
        with filelock.FileLock(lock_file):
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
            else:
                state = {"next_seed": 0, "success_count": 0}

            if state["success_count"] >= target_count:
                print(f"\033[92m[GPU {gpu_id}] Target reached, exiting\033[0m")
                return

            seed_to_try = state["next_seed"]
            current_success = state["success_count"]
            state["next_seed"] += 1

            with open(state_file, "w") as f:
                json.dump(state, f)

        # Try this seed (outside lock)
        success = False
        try:
            TASK_ENV.setup_demo(now_ep_num=current_success, seed=seed_to_try, **args)
            TASK_ENV.play_once()

            if TASK_ENV.plan_success and TASK_ENV.check_success():
                success = True

            TASK_ENV.close_env()
            if args.get("render_freq"):
                TASK_ENV.viewer.close()

        except Exception as e:
            print(f"\033[91m[GPU {gpu_id}] Error with seed {seed_to_try}: {e}\033[0m")
            try:
                TASK_ENV.close_env()
                if args.get("render_freq"):
                    TASK_ENV.viewer.close()
            except:
                pass
            time.sleep(0.5)

        # If successful, atomically record it
        if success:
            with filelock.FileLock(lock_file):
                with open(state_file, "r") as f:
                    state = json.load(f)

                if state["success_count"] >= target_count:
                    print(f"\033[92m[GPU {gpu_id}] Target already reached\033[0m")
                    return

                episode_num = state["success_count"]
                state["success_count"] += 1

                with open(state_file, "w") as f:
                    json.dump(state, f)

                # Append to seed file
                with open(seed_file, "a") as f:
                    f.write(f"{seed_to_try} ")

                # Save trajectory data
                TASK_ENV.setup_demo(now_ep_num=episode_num, seed=seed_to_try, **args)
                TASK_ENV.play_once()
                TASK_ENV.save_traj_data(episode_num)
                TASK_ENV.close_env()

            print(f"\033[92m[GPU {gpu_id}] Seed {seed_to_try} succeeded (episode {episode_num})\033[0m")
        else:
            print(f"\033[93m[GPU {gpu_id}] Seed {seed_to_try} failed\033[0m")


def worker_data_collection(gpu_id, task_name, task_config, save_path, seed_list, total_episodes):
    """Worker for parallel data collection - GPU set before imports"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import after setting GPU
    import yaml
    import importlib
    import traceback
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

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = save_path
    args["need_plan"] = False
    args["render_freq"] = 0
    args["save_data"] = True

    TASK_ENV = class_decorator(task_name)

    lock_file = os.path.join(save_path, "data_collection.lock")
    state_file = os.path.join(save_path, "data_state.json")
    info_file_path = os.path.join(save_path, "scene_info.json")

    clear_cache_freq = args.get("clear_cache_freq", 10)
    local_count = 0

    print(f"\033[92m[GPU {gpu_id}] Data collection worker started\033[0m")

    while True:
        # Atomically get next episode to collect
        with filelock.FileLock(lock_file):
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
            else:
                state = {"next_episode": 0}

            # Skip already existing episodes
            while state["next_episode"] < total_episodes:
                hdf5_path = os.path.join(save_path, 'data', f'episode{state["next_episode"]}.hdf5')
                if not os.path.exists(hdf5_path):
                    break
                state["next_episode"] += 1

            if state["next_episode"] >= total_episodes:
                print(f"\033[92m[GPU {gpu_id}] All episodes done, exiting\033[0m")
                return

            episode_idx = state["next_episode"]
            state["next_episode"] += 1

            with open(state_file, "w") as f:
                json.dump(state, f)

        # Collect this episode (outside lock)
        print(f"\033[34m[GPU {gpu_id}] Collecting episode {episode_idx}\033[0m")

        try:
            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info = TASK_ENV.play_once()

            # Save scene info atomically
            with filelock.FileLock(lock_file):
                if not os.path.exists(info_file_path):
                    with open(info_file_path, "w") as f:
                        json.dump({}, f)

                with open(info_file_path, "r") as f:
                    info_db = json.load(f)

                info_db[f"episode_{episode_idx}"] = info

                with open(info_file_path, "w") as f:
                    json.dump(info_db, f, ensure_ascii=False, indent=4)

            local_count += 1
            TASK_ENV.close_env(clear_cache=(local_count % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()

            if not TASK_ENV.check_success():
                print(f"\033[91m[GPU {gpu_id}] Episode {episode_idx} check_success failed!\033[0m")
            else:
                print(f"\033[92m[GPU {gpu_id}] Episode {episode_idx} completed\033[0m")

        except Exception as e:
            print(f"\033[91m[GPU {gpu_id}] Error at episode {episode_idx}: {e}\033[0m")
            try:
                TASK_ENV.close_env()
            except:
                pass


def main(task_name, task_config, gpu_ids):
    import yaml

    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    save_path = os.path.join(args["save_path"], task_name, task_config)
    os.makedirs(save_path, exist_ok=True)

    episode_num = args["episode_num"]

    print(f"Task: \033[34m{task_name}\033[0m")
    print(f"GPUs: \033[34m{gpu_ids}\033[0m")
    print(f"Episodes: \033[34m{episode_num}\033[0m")

    # =========== Parallel Seed Collection ===========
    if not args.get("use_seed", False):
        print("\n\033[93m[Starting Parallel Seed Collection]\033[0m")

        # Initialize state files
        state_file = os.path.join(save_path, "seed_state.json")
        seed_file = os.path.join(save_path, "seed.txt")

        # Check existing progress
        if os.path.exists(seed_file):
            with open(seed_file, "r") as f:
                existing_seeds = f.read().split()
            existing_count = len(existing_seeds)
            if existing_count > 0:
                # Resume from existing
                max_seed = max(int(s) for s in existing_seeds)
                with open(state_file, "w") as f:
                    json.dump({"next_seed": max_seed + 1, "success_count": existing_count}, f)
                print(f"Resuming from {existing_count} existing seeds")
        else:
            with open(state_file, "w") as f:
                json.dump({"next_seed": 0, "success_count": 0}, f)

        # Start workers
        processes = []
        for gpu_id in gpu_ids:
            p = mp.Process(
                target=worker_seed_collection,
                args=(gpu_id, task_name, task_config, save_path, episode_num)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("\033[92m[Seed Collection Complete]\033[0m")

    # Load seed list
    seed_file = os.path.join(save_path, "seed.txt")
    with open(seed_file, "r") as f:
        seed_list = [int(s) for s in f.read().split()]

    print(f"Loaded {len(seed_list)} seeds")

    # =========== Parallel Data Collection ===========
    if args.get("collect_data", True):
        print("\n\033[93m[Starting Parallel Data Collection]\033[0m")

        # Initialize state file
        state_file = os.path.join(save_path, "data_state.json")
        with open(state_file, "w") as f:
            json.dump({"next_episode": 0}, f)

        # Start workers
        processes = []
        for gpu_id in gpu_ids:
            p = mp.Process(
                target=worker_data_collection,
                args=(gpu_id, task_name, task_config, save_path, seed_list, episode_num)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("\033[92m[Data Collection Complete]\033[0m")

        # Generate instructions
        language_num = args.get("language_num", 1)
        command = f"cd description && bash gen_episode_instructions.sh {task_name} {task_config} {language_num}"
        os.system(command)


if __name__ == "__main__":
    # from test_render import Sapien_TEST
    # Sapien_TEST()

    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    main(args.task_name, args.task_config, gpu_ids)
