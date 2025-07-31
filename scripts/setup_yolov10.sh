#!/usr/bin/env bash
set -e

envName="yolov10"
sourcePath="3rdparty/yolov10"

# create conda env
conda env create -y -f yolov10.yml -n "$envName"
eval "$(conda shell.bash hook)"
conda activate "$envName"

# install dependencies in requirements.txt
pip install -r "$sourcePath/requirements.txt"

# install yolov10
pip install -e "$sourcePath"
