#!/usr/bin/env bash
set -e

cwd=$(dirname "$0")
root=$(dirname "$cwd")

envName="detrex"
sourcePath="$root/3rdparty/detrex"

# create conda env
conda env create -y -f "$root/detrex.yml" -n "$envName"
eval "$(conda shell.bash hook)"
conda activate "$envName"

# install detectron2
pip install -e "$sourcePath/detectron2"

# install detrex
pip install -e "$sourcePath"

echo -e "\033[0;32mInstallation Successfully!"
