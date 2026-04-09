#!/usr/bin/env bash
set -e

cwd=$(dirname "$(realpath "$0")")
root=$(dirname "$cwd")

envName=${1:-"yolov10"}
sourcePath="$root/3rdparty/yolov10"
projectPath="$root/projects/yolov10"

# create conda env
conda env create -y -f "$root/yolov10.yml" -n "$envName"
eval "$(conda shell.bash hook)"
conda activate "$envName"

# install dependencies in requirements.txt
pip install -r "$sourcePath/requirements.txt"

# install yolov10
pip install -e "$sourcePath"

# configure yolo settings
# NOTE: runs_dir will only work if project setting is not specified! Please check the project/yolov10/README.md
yolo settings datasets_dir="$projectPath/datasets" weights_dir="$projectPath/weights" runs_dir="$projectPath/runs"

echo -e "\033[0;32mInstallation Successfully!"
