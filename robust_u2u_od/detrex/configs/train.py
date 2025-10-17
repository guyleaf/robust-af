from detrex.config import get_config as get_upstream_config

# Common training-related configs that are designed for "tools/train_net.py"
# You can use your own instead, together with your own train_net.py
train = get_upstream_config("common/train.py").train

# support SyncBN in custom train_net.py
train.sync_bn = False
