# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RobustAF is a plug-and-play adapter-based framework for robust UAV detection in degraded conditions (snow, fog, rain, noise). It integrates robust feature restoration modules into three object detection frameworks: **YOLOv10**, **RT-DETR v2**, and **DINO (detrex)**. Each framework uses a separate conda environment.

## Environment Setup

Each framework has its own conda environment and setup script:

```bash
# YOLOv10 (Python 3.9, PyTorch 2.0.1, CUDA 11.8)
./scripts/setup_yolov10.sh

# RT-DETR v2 (Python 3.10, PyTorch 2.7.1, CUDA 11.8)
./scripts/setup_rtdetrv2.sh

# DINO/detrex (Python 3.9, PyTorch 1.13.1, CUDA 11.7)
./scripts/setup_detrex.sh
```

The conda environment YML files (`yolov10.yml`, `rtdetrv2.yml`, `detrex.yml`) set critical environment variables:
- `ULTRALYTICS_ENV_MODULE=robust_af.yolov10` — pre-loads robust modules before YOLOv10
- `RTDETRV2_ENV_MODULE=robust_af.rtdetrv2` — pre-loads robust modules before RT-DETR v2
- `DETECTRON2_ENV_MODULE=robust_af.detrex` — pre-loads robust modules before detrex
- `NVIDIA_TF32_OVERRIDE=0` — disables TF32 for cross-GPU consistency

## Training and Evaluation Commands

### YOLOv10

Commands run from `projects/yolov10/`. Configs use YOLO YAML format.

```bash
# Train (reads config YAML directly)
python run.py configs/dut_anti_uav/robust_yolov10n.yaml

# Validate (test split configs are in configs/*/test/)
python run.py configs/dut_anti_uav/test/robust_yolov10n_dut_anti_uav.yaml
```

### RT-DETR v2

Commands run from `projects/rtdetrv2/`. Uses the 3rdparty train script.

```bash
# Single GPU train
python 3rdparty/RT-DETR/rtdetrv2_pytorch/tools/train.py \
    -c projects/rtdetrv2/configs/dut_anti_uav/robust_rtdetrv2_r18vd_120e.yml

# Multi-GPU train (torchrun)
torchrun --nproc_per_node=4 \
    3rdparty/RT-DETR/rtdetrv2_pytorch/tools/train.py \
    -c projects/rtdetrv2/configs/dut_anti_uav/robust_rtdetrv2_r18vd_120e.yml

# Test only
python 3rdparty/RT-DETR/rtdetrv2_pytorch/tools/train.py \
    -c projects/rtdetrv2/configs/dut_anti_uav/test/robust_rtdetrv2_r18vd_dut_anti_uav.yml \
    --test-only -t /path/to/checkpoint.pth
```

The `-u key=value` flag overrides any YAML config field inline.

### DINO (detrex)

Commands run from `projects/dino/`. Configs are Python LazyConfig files.

```bash
# Multi-GPU train
python train_net.py --config-file configs/dut_anti_uav/robust_dino_r50_v2_4scale_12ep.py \
    --num-gpus 4

# Eval only
python train_net.py --config-file configs/dut_anti_uav/test/robust_dino_r50_v2_4scale.py \
    --eval-only

# Resume training
python train_net.py --config-file configs/dut_anti_uav/robust_dino_r50_v2_4scale_12ep.py \
    --num-gpus 4 --resume
```

Override config values via `train.key=value` positional args (detectron2 LazyConfig style).

## Linting

```bash
ruff check .
ruff check --fix .
```

Configured rules: E4, E7, E9 (errors), F (pyflakes), I (isort), NPY201 (numpy 2.0 compat).

## Project Structure

```
robust_af/              # Core package (pip install -e .)
├── models/
│   └── robust_layers/  # AFR, AMFG, AMFGv2, SimpleNN robust adapter modules
├── yolov10/            # YOLOv10 integration (trainer, validator, dataset, loss)
├── rtdetrv2/           # RT-DETR v2 integration (solver, dataset, zoo/models)
├── detrex/             # DINO integration (engine, modeling, data)
├── transforms/         # Degradation augmentations (snow, fog, rain, noise)
└── utils/              # Shared utilities (seeds, frozen BN, COCO tools)
projects/               # Experiment configs (one subfolder per framework)
├── yolov10/
│   ├── configs/        # Train/val YAMLs; test/ subfolder holds eval configs
│   ├── datasets -> /path/to/data   (symlink)
│   └── weights -> /path/to/pretrained  (symlink)
├── rtdetrv2/
│   ├── configs/        # Hierarchical YAML configs using __include__
│   ├── dataset -> /path/to/data    (symlink)
│   └── output -> /path/to/workdir  (symlink)
└── dino/
    ├── configs/        # Python LazyConfig files (detectron2 style)
    ├── modeling/       # DINO model variants (robust_dino.py, robust_dino_v2.py)
    ├── train_net.py    # Training entry point
    ├── datasets -> /path/to/data   (symlink)
    └── outputs -> /path/to/workdir (symlink)
3rdparty/               # Git submodules (yolov10, RT-DETR, detrex forks)
tools/                  # Standalone scripts (COCO↔YOLO conversion, checkpoint remapping)
scripts/                # Environment setup scripts
```

## Architecture

### Robust Adapter Modules (`robust_af/models/robust_layers/`)

The plug-and-play adapters sit between backbone feature maps and the detection head:

- **AFR** (Anti-degradation Feature Restoration): Combined spatial + frequency domain restoration
  - **SpatialAFR**: Spatial-only variant (most commonly used in configs)
  - **FrequencyAFR**: Frequency-only variant
- **AMFG** / **AMFGv2**: Adaptive Multi-scale Feature Guidance variants
- **SimpleNN**: Lightweight baseline

For RT-DETR v2, adapters are instantiated as a `MultiScaleProcessor` that wraps three modules (one per FPN scale: 128, 256, 512 dims). The module type is configured directly in the experiment YAML:

```yaml
MultiScaleProcessor:
  modules:
    - { type: SpatialAFR, embed_dims: 128, activation: ReLU }
    - { type: SpatialAFR, embed_dims: 256, activation: ReLU }
    - { type: SpatialAFR, embed_dims: 512, activation: ReLU }
```

### Degradation Augmentation (`robust_af/transforms/degradations.py`)

Uses `albumentations` + `imgaug` to simulate: snow, fog, rain, Gaussian noise, ISO noise, multiplicative noise, resampling blur. Controlled via config:

```yaml
degradation:
  enabled: True
  seed: 2025          # fixed for reproducibility
  identity: True      # include clean pass-through in augmentation pool
  ignored_degradations: []
```

### Training Strategy

The standard robust training approach:
1. Start from a pretrained clean detector checkpoint (`pretrained:` in config)
2. Freeze backbone + head layers (only robust adapters train)
3. Apply degradation augmentation to training images
4. Use CST loss (Content-Style Transfer, `weight: 20`) alongside detection loss to supervise feature restoration

### Framework Integration Mechanism

Each framework integration uses a pre-load hook via environment variable so that custom classes are registered before the framework initializes:

- `ULTRALYTICS_ENV_MODULE` → calls `robust_af.yolov10.setup_environment()`
- `RTDETRV2_ENV_MODULE` → calls `robust_af.rtdetrv2.setup_environment()`

This is why the conda environment YML files set these variables — they must be active before any framework import.

### Config Hierarchy (RT-DETR v2)

RT-DETR v2 configs use `__include__` for composability:
- `common/dataset/` — dataset paths and splits
- `common/runtime.yml` — logging, checkpointing
- `common/optimizer.yml` — AdamW defaults
- `common/models/` — backbone + adapter architecture
- Top-level experiment YAML — overrides (epochs, lr, `MultiScaleProcessor` module type)

### Dataset Setup

Datasets and output directories are symlinked into each project folder:

```bash
# YOLOv10
cd projects/yolov10
ln -s /path/to/datasets datasets
ln -s /path/to/weights weights
ln -s /path/to/workdir yolov10  # matches project: yolov10 in config

# RT-DETR v2
cd projects/rtdetrv2
ln -s /path/to/datasets dataset
ln -s /path/to/workdir output

# DINO
cd projects/dino
ln -s /path/to/datasets datasets
ln -s /path/to/workdir outputs
```

The DUT Anti-UAV dataset splits used: `train`, `val`, `test`, `degraded_val`, `degraded_test`.

## Key Notes
### YOLOv10
- **Validation discrepancy**: mAP results from inline validation during `model.train()` may differ slightly from standalone `model.val()`. Published results use standalone validation.
- **AMP disabled if training with frequency domain restoration**: Use `amp: False` in YOLOv10 configs if using frequency domain restoration to avoid FFT size limitation — do not enable without testing.
- **Deterministic mode disabled**: `deterministic: False` for performance; seed is still set for reproducibility.
- **Checkpoint remapping**: After package refactoring (rename), add the map to the `_MODULE_PREFIX_MAP` variable in `robust_af/utils/setup.py` to remap old checkpoints automatically while loading. Using `tools/ultralytics/map_checkpoint_to_new_package.py` to remap old checkpoints is optional.
