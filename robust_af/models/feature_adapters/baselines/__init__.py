# ruff: noqa: F401

# if interface is compatible, re-export here directly.
# Otherwise, make a thin adapter for it in the other new file.

__all__ = list(globals().keys())
