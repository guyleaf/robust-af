#!/usr/bin/env bash
set -eu

cwd=$(dirname "$0")
degradations=('snow' 'fog' 'rain' 'gaussian_noise' 'iso_noise' 'multiplicative_noise' 'resampling_blur' 'motion_blur' 'zoom_blur' 'color_jitter' 'compression' 'elastic' 'glass_blur' 'brightness' 'contrast')
dataset_root=$1
subset=$2

for degradation in "${degradations[@]}"; do
    bash "$cwd/make_offline_subset.sh" "$dataset_root" "$subset" "$degradation" "${@:3}"
done
