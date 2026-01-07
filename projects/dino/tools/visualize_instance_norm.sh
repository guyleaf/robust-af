#!/usr/bin/env bash
set -eu

for i in {5..50..5}; do
  python tools/visualize_instance_norm.py --n-components 2 --max-num-samples 50 --perplexity "$i" --IN --legend --feat-size 8 --out-dir /home/leafying/work/work_dirs/detrex/dino_r50_4scale/dut_anti_uav/dut_anti_uav/IN_tsne_50_v2 configs/dut_anti_uav/test/dino_r50_4scale.py /home/leafying/work/work_dirs/detrex/dino_r50_4scale/dut_anti_uav/dino_r50_4scale_36ep_5e-5_lr/model_best_0021449.pth
  # python tools/visualize_instance_norm.py --n-components 2 --max-num-samples 50 --perplexity "$i" --IN --legend --feat-size 4 --out-dir /home/leafying/work/work_dirs/detrex/dino_swin_small_224_4scale/dut_anti_uav/dut_anti_uav/IN_tsne_50 configs/dut_anti_uav/test/dino_swin_small_224_4scale.py /home/leafying/git/robust-au-od/projects/dino/outputs/dino_swin_small_224_4scale/dut_anti_uav/dino_swin_small_224_4scale_24ep_1e-4_lr_warmup/model_best_0013649.pth
done
