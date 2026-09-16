#!/usr/bin/env bash
set -e

cwd=$(dirname "$0")
root=$(dirname "$(dirname "$cwd")")
out_dir=${1:-$root/images/vis}

python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir/dut_anti_uav" ~/data/UAV/DUT_Anti_UAV/detection
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir/dut_anti_uav" ~/data/UAV/DUT_Anti_UAV/tracking
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir" ~/data/UAV/Det_Fly
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir" ~/data/UAV/MAV_VID
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir" ~/data/UAV/DDS
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir" ~/data/UAV/Sim2Air/UAV_Eagle
python "$cwd/browse_coco_dataset.py" --out-dir "$out_dir" ~/data/UAV/Sim2Air/S_UAV_T

# paper demo
# python browse_coco_dataset.py --annotations "val.json" "degraded_val.json" --out-dir robust_anti_uav --seed 1234 ~/data/UAV/Robust_Anti_UAV_Low

# qualitative results (rtdetrv2_r18vd+SpatialAFR)
# python browse_coco_dataset.py --annotations "test.json" "degraded_test.json" --out-dir "dut_anti_uav" --ids 1381 297 361 13 417 229 1605 1077 1925 --single -- ~/data/UAV/DUT_Anti_UAV/detection
# python tools/visualize_samples.py --image-ids 1077 1973 1181 149 417 1709 473 2133 1801 1745 877 1349 177 285 13 1617 889 17 1125 309 1493 373 1557 1605 1009 185 1001 241 953 1193 697 1925 865 725 593 1697 2065 9 1993 1137 1353 297 105 229 73 589 2113 1381 -- $CONFIG $CHECKPOINT

# dino_r50+SpatialAFR
# python tools/base/visualize_json_results.py --input $INPUT --output $OUTPUT --dataset dut_anti_uav_test_degraded --no-show-gt --image-ids 1077 653 269 1321 129 1181 617 741 113 1777 337 2133 2177 1841 833 1709 877 1745 429 1889 97 285 13 177 1325 1493 885 1605 185 1009 1217 1101 1997 241 1041 1257 1765 953 725 1925 697 865 1193 985 57 1405 945 1621 1597 817 2065 9 1885 1045 1637 1397 1401 1137 25 189 1353 297 105 1089 997 497 145 717 1165 41 217 229 1285 509 589 909
