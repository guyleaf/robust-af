#!/usr/bin/env bash
set -e

envName="rtdetrv2"
rtdetrv2Path="3rdparty/RT-DETR/rtdetrv2_pytorch"

# create conda env
conda env create -y -f rtdetrv2.yml -n "$envName"
eval "$(conda shell.bash hook)"
conda activate "$envName"

# install rtdetrv2
pip install -e "$rtdetrv2Path"
