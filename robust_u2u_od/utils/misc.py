import sys
from pathlib import Path

THIRD_PARTY_ROOT = Path(__file__).resolve().parents[2] / "3rdparty"


# copied from https://github.com/open-mmlab/mmengine/blob/main/mmengine/analysis/print_helper.py#L20
def format_size(x: int, sig_figs: int = 3, hide_zero: bool = False) -> str:
    """Formats an integer for printing in a table or model representation.

    Expresses the number in terms of 'kilo', 'mega', etc., using
    'K', 'M', etc. as a suffix.

    Args:
        x (int): The integer to format.
        sig_figs (int): The number of significant figures to keep.
            Defaults to 3.
        hide_zero (bool): If True, x=0 is replaced with an empty string
            instead of '0'. Defaults to False.

    Returns:
        str: The formatted string.
    """
    if hide_zero and x == 0:
        return ""

    def fmt(x: float) -> str:
        # use fixed point to avoid scientific notation
        return f"{{:.{sig_figs}f}}".format(x).rstrip("0").rstrip(".")

    if abs(x) > 1e14:
        return fmt(x / 1e15) + "P"
    if abs(x) > 1e11:
        return fmt(x / 1e12) + "T"
    if abs(x) > 1e8:
        return fmt(x / 1e9) + "G"
    if abs(x) > 1e5:
        return fmt(x / 1e6) + "M"
    if abs(x) > 1e2:
        return fmt(x / 1e3) + "K"
    return str(x)


def find_3rdparty_submodule(name: str):
    """Find the 3rd-party submodule by name

    Args:
        name (str): Name of the 3rd-party submodule

    Raises:
        ValueError: The 3rd-party submodule is not found.

    Returns:
        pathlib.Path: The root of the 3rd-party submodule
    """
    path = THIRD_PARTY_ROOT / name
    if not path.is_dir():
        raise ValueError(f"The submodule is not found in {path}.")
    return path


def add_3rdparty_submodule(name: str):
    """Prepend 3rd-party submodule to sys.path

    Args:
        name (str): Name of the 3rd-party submodule
    """
    path = str(find_3rdparty_submodule(name))
    if path not in sys.path:
        sys.path.insert(0, path)
