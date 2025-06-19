"""pymgcv: Generalized Additive Models in Python."""

from .gam import GAM
from .terms import Interaction, Linear, Offset, Smooth, TensorSmooth

__all__ = [
    "GAM",
    "Linear",
    "Smooth",
    "TensorSmooth",
    "Interaction",
    "Offset",
]

# Version information
__version__ = "0.0.0"
__author__ = "Daniel Ward"
__email__ = "danielward27@outlook.com"
