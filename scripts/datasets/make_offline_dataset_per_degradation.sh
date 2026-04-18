#!/usr/bin/env bash
set -eu

degradations=('snow' 'fog' 'rain' 'gaussian_noise' 'iso_noise' 'multiplicative_noise' 'resampling_blur' 'motion_blur' 'zoom_blur' 'color_jitter' 'compression' 'elastic' 'glass_blur' 'brightness' 'contrast')
dataset_root=$1

cwd=$(dirname "$0")

for degradation in "${degradations[@]}"; do
    echo "Degradation: $degradation"
    python "$cwd/make_degraded_coco_dataset.py" --tgt-images-name "degraded_$degradation" --degradations "$degradation" "${@:2}" -- "$dataset_root"
done
