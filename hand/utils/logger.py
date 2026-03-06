"""Simple colored logger."""

import os


def get_rank():
    """Get the current process rank."""
    return int(os.environ.get("LOCAL_RANK", 0))


def _get_color_code(color: str) -> str:
    """Convert color name to ANSI color code."""
    color_map = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }
    return color_map.get(color, "0")


def log(message: str, color: str = ""):
    """Log message only for rank 0 process."""
    if get_rank() != 0:
        return

    if color:
        print(f"\033[1;{_get_color_code(color)}m{message}\033[0m")
    else:
        print(f"\033[1m{message}\033[0m")
