#!/usr/bin/env bash
set -eu

cwd=$(dirname "$0")
dataset_root=$1
subset=$2
degradation=$3

echo "Degradation: $degradation"
python "$cwd/make_degraded_coco_dataset.py" --annotation-files "$subset" --tgt-images-name "degraded_$degradation" --degradations "$degradation" "${@:4}" -- "$dataset_root"
