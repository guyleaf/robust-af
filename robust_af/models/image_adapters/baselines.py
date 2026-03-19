from denet.core import DENet as _DENet

# if interface is compatible, re-export directly.
# Otherwise, make a thin adapter for it.
DENet = _DENet
