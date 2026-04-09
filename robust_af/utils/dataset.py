from typing import Optional, TypeVar

from sklearn.model_selection import train_test_split

T = TypeVar("T")


def split_into_train_val_test(
    x: list[T], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[T]]:
    if test_ratio < 1.0:
        train, test = train_test_split(x, test_size=test_ratio, random_state=seed)
        ratio_remaining = 1.0 - test_ratio
        val_ratio = val_ratio / ratio_remaining
        if val_ratio < 1.0:
            train, val = train_test_split(train, test_size=val_ratio, random_state=seed)
        else:
            val = train
            train = []
    else:
        test = x
        train = val = []

    return {"train": train, "val": val, "test": test}


def split_into_train_val(
    x: list[T], val_ratio: float, seed: int, labels: Optional[list] = None
) -> dict[str, list[T]]:
    if val_ratio < 1.0:
        train, val = train_test_split(
            x, test_size=val_ratio, random_state=seed, stratify=labels
        )
    else:
        val = x
        train = []

    return {"train": train, "val": val}
