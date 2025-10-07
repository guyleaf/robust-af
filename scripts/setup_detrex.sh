#!/usr/bin/env bash
set -e

envName="detrex"
detrexPath="3rdparty/detrex"

# create conda env
conda env create -y -f detrex.yml -n "$envName"
eval "$(conda shell.bash hook)"
conda activate "$envName"

# install detectron2
pip install -e "$detrexPath/detectron2"

# install detrex
pip install -e "$detrexPath"

echo -e "\033[0;32mInstallation Successfully!"
