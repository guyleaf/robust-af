## Setup

### Dataset Folder
```bash
ln -s /path/to/datasets datasets
```

### Weight Folder
```bash
ln -s /path/to/weights weights
```

### Experiment Folder
```bash
# NOTE: if project setting is specified, the runs_dir will be ignored. But project setting is important for wandb.
# So, for consistency, we use yolov10 as the name of folder to match the project setting.

# if project setting is specified (default to our configs)
ln -s /path/to/workdir yolov10
# else
ln -s /path/to/workdir runs
```

## Dataset (val)
1. Evaluate degraded images (default)
2. Evaluate clear and degraded images
   1. Make your own subset with our dataset script
   2. Change the dataset config
3. Evaluate with online augmentation
   1. Change the dataset config
   2. Replace `robust=mode == "train"` with `robust=True` to allow augment val subset in `RobustYOLOv10DetectionTrainer`.

## Notes
1. The performances may slightly differ between validation after training (`model.train(...)`) and standalone validation (`model.val(...)`).
   To be consistent, **the results on the table are based on the latter**. (I already check the implementation and match the hyperparameters but still failure. Feel free to send a Pull Request if you found)
