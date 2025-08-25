"""Custom types for pymgcv."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import pandas as pd

S = TypeVar("S", np.ndarray, pd.DataFrame)


@dataclass
class FitAndSE(Generic[S]):
    """Container for predictions or partial effects with standard errors.

    Attributes:
        fit: Predicted values or partial effect.
        se: Standard errors of the predictions.
    """

    fit: S
    se: S
