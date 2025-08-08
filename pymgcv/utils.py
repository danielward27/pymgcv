from collections.abc import Mapping
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy as np
from pymgcv.rpy_utils import to_py
import pandas as pd
rutils = importr("utils")


def get_data(name: str):
    """Get built-in R dataset.

    Currently assumes that the dataset is a dataframe.
    """
    with ro.local_context() as lc:
        rutils.data(ro.rl(name), envir=lc)
        return to_py(lc[name])


def data_len(data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray]):
    """Get the length of the data.

    If the data is a dictionary, returns the maximum value of the shape along axis 0.
    """
    if isinstance(data, pd.DataFrame):
        return len(data)
    return max([d.shape[0] for d in data.values()])
