#!/usr/bin/env bash
set -e

cwd=$(dirname "$0")
root=$(dirname "$cwd")

envName=${1:-"rtdetrv2"}
sourcePath="$root/3rdparty/RT-DETR/rtdetrv2_pytorch"

# create conda env
conda env create -y -f "$root/rtdetrv2.yml" -n "$envName"
eval "$(conda shell.bash hook)"
conda activate "$envName"

# install rtdetrv2
pip install -e "$sourcePath"

echo -e "\033[0;32mInstallation Successfully!"
