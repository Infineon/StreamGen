"""🐍📦 a framework and implementation for simulating stream datasets."""

import importlib

from beartype import beartype

EXTRAS = {
    "examples": ["perlin_numpy", "polars", "ipymlp"],
    "cl": [
        # "avalanche",
        "continuum",
    ],
    "all": [
        "perlin_numpy",
        "polars",
        "ipympl",
        "avalanche",
        "continuum",
    ],
}
"""🧾 Dictionary mapping pip EXTRAS to their corresponding dependencies.

This dictionary has to stay in sync with `[tool.poetry.EXTRAS]` in `pyproject.toml`.
"""


@beartype
def is_extra_installed(extra: str) -> bool:
    """📦❓ check if setuptools/pip extra is installed in the current Python environment.

    Args:
        extra (str): name of the extra.

    Returns:
        bool: True if extra is installed.
    """
    return all(importlib.util.find_spec(package) is not None for package in EXTRAS[extra])
