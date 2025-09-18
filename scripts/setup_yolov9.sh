#!/usr/bin/env bash
set -e

envName="yolov9"
sourcePath="3rdparty/yolov9"

# create conda env
conda env create -y -f yolov9.yml -n "$envName"
eval "$(conda shell.bash hook)"
conda activate "$envName"

# install yolov9
pip install -e "$sourcePath"

echo -e "\033[0;32mInstallation Successfully!"
