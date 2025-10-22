from detrex.config import get_config as get_upstream_config

# Common training-related configs that are designed for "tools/train_net.py"
# You can use your own instead, together with your own train_net.py
train = get_upstream_config("common/train.py").train

# support SyncBN
train.sync_bn = False

# support saving the best checkpoint based on metric
train.best_checkpointer = dict(val_metric="bbox/AP", mode="max")

# disable TF32 to get consistent results between old and Ampere (and later) GPU devices
train.allow_tf32 = False
