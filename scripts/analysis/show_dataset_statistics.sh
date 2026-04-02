#!/usr/bin/env bash
set -e

cwd=$(dirname "$0")
root=$(dirname "$(dirname "$cwd")")
out_dir=${1:-$root/images}

python "$cwd/show_dataset_statistics.py" \
    --out-file "$out_dir/dut_anti_uav_detection.png" \
    --title "DUT Anti-UAV" \
    ~/data/UAV/DUT_Anti_UAV/detection

python "$cwd/show_dataset_statistics.py" \
    --out-file "$out_dir/det_fly.png" \
    --title "Det-Fly" \
    ~/data/UAV/Det_Fly

python "$cwd/show_dataset_statistics.py" \
    --out-file "$out_dir/mav_vid.png" \
    --title "MAV-VID" \
    ~/data/UAV/MAV_VID

python "$cwd/show_dataset_statistics.py" \
    --out-file "$out_dir/dds.png" \
    --title "DDS" \
    ~/data/UAV/DDS

python "$cwd/show_dataset_statistics.py" \
    --out-file "$out_dir/sim2air/uav_eagle.png" \
    --title "UAV-Eagle" \
    ~/data/UAV/Sim2Air/UAV_Eagle

python "$cwd/show_dataset_statistics.py" \
    --out-file "$out_dir/sim2air/s_uav_t.png" \
    --title "S-UAV-T" \
    ~/data/UAV/Sim2Air/S_UAV_T
