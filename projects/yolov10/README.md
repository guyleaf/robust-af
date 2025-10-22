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
