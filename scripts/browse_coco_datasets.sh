#!/usr/bin/env bash
set -e

cwd=$(dirname "$0")
root=$(dirname "$cwd")
out_dir=${1:-$root/images/vis}

python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir/dut_anti_uav" ~/data/UAV/DUT_Anti_UAV/detection
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir/dut_anti_uav" ~/data/UAV/DUT_Anti_UAV/tracking
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir" ~/data/UAV/Det_Fly
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir" ~/data/UAV/MAV_VID
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir" ~/data/UAV/DDS
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir" ~/data/UAV/Sim2Air/UAV_Eagle
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir" ~/data/UAV/Sim2Air/S_UAV_T
