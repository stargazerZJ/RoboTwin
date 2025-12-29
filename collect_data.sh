#!/bin/bash

task_name=${1}
task_config=${2}
gpu_ids=${3:-"0"}  # Default to GPU 0

./script/.update_path.sh > /dev/null 2>&1

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config --gpu_ids $gpu_ids

rm -rf data/${task_name}/${task_config}/.cache