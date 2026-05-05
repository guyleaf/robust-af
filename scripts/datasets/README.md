# datasets - README.md

## Datasets
Follow these steps below.

### Sources

### Preparation

### Augment with synthetic degradations (offline, evaluation only)
```bash
# Follow these steps based on comments.
# Total size: ~59GB

# val + test subset
## all datasets (15 degradations)
python make_degraded_coco_dataset.py -- <dataset_root>

# val subset
## DUT Anti-UAV (foggy)
bash make_offline_subset.sh <dataset_root> val.json fog

# test subset
## DUT Anti-UAV (per degradation, including foggy)
bash make_offline_subset_per_degradation.sh <dataset_root> test.json

## others (foggy)
### Robust Anti-UAV Low
bash make_offline_subset.sh <dataset_root> val.json fog
### others (excluding DUT Anti-UAV)
bash make_offline_subset.sh <dataset_root> test.json fog

# NOTE: each subset uses an independent dataloader, so val and val+test runs produce identical val images (verified by binary diff).
```


## Usage
### Offline dataset for val and test (15 degradations)
`python make_degraded_coco_dataset.py -- <dataset_root>`

### Offline dataset for specific subset (specific degradation)
`bash make_offline_subset.sh <dataset_root> <subset> <degradation>`

### Offline dataset for specific subset (per degradation)
`bash make_offline_subset_per_degradation.sh <dataset_root> <subset>`

### Others
Please look at the arguments in `make_degraded_coco_dataset.py`.
