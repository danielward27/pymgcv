from dataclasses import dataclass

import numpy as np
import pandas as pd




@dataclass
class FitAndSEArrays:
    """Container for predictions or individual partial effects with optional standard errors.

    Used for predictions or the partial effect of a single variable.

    Attributes:
        fit: Predicted values or partial effect.
        se: Standard errors of the predictions.
    """

    fit: np.ndarray
    se: np.ndarray


@dataclass
class FitAndSEDataFrames:
    """Container for partial effects across multiple variables with optional standard errors.

    Used for tabular results, such as partial effects for multiple variables or observations.

    Attributes:
        fit: Partial effects as a pandas DataFrame.
        se: Standard errors of the partial effects, if available.
    """

    fit: pd.DataFrame
    se: pd.DataFrame
