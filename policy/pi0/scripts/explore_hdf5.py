#!/usr/bin/env python3
"""Quick script to explore HDF5 structure."""
import h5py
import numpy as np

# Explore HDF5 structure
with h5py.File('training_data/pi0_baseline/put_object_cabinet-baseline-500/episode_0/episode_0.hdf5', 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f'{name}: shape={obj.shape}, dtype={obj.dtype}')
        else:
            print(f'{name}/')
    f.visititems(print_structure)

    print()
    print('--- Exploring qpos ---')
    qpos = f['observations/qpos'][:]
    print(f'qpos shape: {qpos.shape}')
    print(f'qpos[0]: {qpos[0]}')
    print(f'left_arm_dim: {f["observations/left_arm_dim"][0]}')
    print(f'right_arm_dim: {f["observations/right_arm_dim"][0]}')

    print()
    print('--- Action data ---')
    action = f['action'][:]
    print(f'action shape: {action.shape}')
    print(f'action[0]: {action[0]}')

    print()
    print('--- Gripper analysis (end values of each arm section) ---')
    left_arm_dim = f["observations/left_arm_dim"][0]
    right_arm_dim = f["observations/right_arm_dim"][0]

    # Gripper values over time
    left_gripper = qpos[:, left_arm_dim]  # Last value of left arm is gripper
    right_gripper = qpos[:, left_arm_dim + 1 + right_arm_dim]  # After left arm (7) + 1 gap + right arm (6)

    print(f'Left gripper index: {left_arm_dim}')
    print(f'Right gripper index: {left_arm_dim + 1 + right_arm_dim}')
    print(f'Left gripper range: min={left_gripper.min():.4f}, max={left_gripper.max():.4f}')
    print(f'Right gripper range: min={right_gripper.min():.4f}, max={right_gripper.max():.4f}')

    print()
    print('--- First 20 timesteps gripper values ---')
    for t in range(min(20, len(qpos))):
        print(f't={t:3d}: left={left_gripper[t]:.4f}, right={right_gripper[t]:.4f}')

    print()
    print('--- Sample gripper values over episode ---')
    for t in range(0, len(qpos), max(1, len(qpos)//20)):
        print(f't={t:3d}: left={left_gripper[t]:.4f}, right={right_gripper[t]:.4f}')