"""pymgcv: Generalized Additive Models in Python."""

from importlib.metadata import version

from .gam import GAM

__all__ = [
    "GAM",
]

# Version information
__version__ = version("pymgcv")
__author__ = "Daniel Ward"
__email__ = "danielward27@outlook.com"
