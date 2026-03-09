#!/usr/bin/env bash
set -e

cwd=$(dirname "$0")
root=$(dirname "$(dirname "$cwd")")
out_dir=${1:-$root/images}

# python "$cwd/show_dataset_statistics.py" --out-file "$out_dir/dut_anti_uav.png" --root-dirs ~/data/UAV/DUT_Anti_UAV/detection/images --annotations ~/data/UAV/DUT_Anti_UAV/detection/annotations/train.json ~/data/UAV/DUT_Anti_UAV/detection/annotations/val.json ~/data/UAV/DUT_Anti_UAV/detection/annotations/test.json --title "DUT Anti-UAV"
# python "$cwd/show_dataset_statistics.py" --out-file "$out_dir/det_fly.png" --root-dirs ~/data/UAV/Det_Fly/images --annotations ~/data/UAV/Det_Fly/annotations/train.json ~/data/UAV/Det_Fly/annotations/val.json --title "Det-Fly"
# python "$cwd/show_dataset_statistics.py" --out-file "$out_dir/mav_vid.png" --root-dirs ~/data/UAV/MAV_VID/images --annotations ~/data/UAV/MAV_VID/annotations/train.json ~/data/UAV/MAV_VID/annotations/val.json --title "MAV-VID"
# python "$cwd/show_dataset_statistics.py" --out-file "$out_dir/dds.png" --root-dirs ~/data/UAV/DDS/images --annotations ~/data/UAV/DDS/annotations/train.json ~/data/UAV/DDS/annotations/val.json --title "DDS"
# python "$cwd/show_dataset_statistics.py" --out-file "$out_dir/dds_10_interval.png" --root-dirs ~/data/UAV/DDS_10_interval/images --annotations ~/data/UAV/DDS_10_interval/annotations/train.json ~/data/UAV/DDS_10_interval/annotations/val.json --title "DDS (10 interval)"
