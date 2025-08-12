from collections.abc import Mapping

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.rpy_utils import to_py

rutils = importr("utils")
rbase = importr("base")


def load_rdata_dataframe_from_url(url: str) -> pd.DataFrame:
    """Load an RData (.rda) file from URL expecting a single dataframe.

    None-dataframe data is not supported.

    !!! note

        When downloading from GitHub, use the "raw" URL (e.g., `https://github.com/.../raw/.../file.rda`).
        Avoid "blob" URLs, as they serve HTML pages instead of the actual file.

    """
    temp_file = rbase.tempfile(fileext=".RData")
    rutils.download_file(url, destfile=temp_file, mode="wb")
    loaded_names = rbase.load(temp_file)
    loaded_names_py = list(loaded_names)

    if len(loaded_names_py) != 1:
        raise ValueError(
            f"Expected one object, got {len(loaded_names_py)}: {loaded_names_py}",
        )

    data_r = ro.r[loaded_names_py[0]]
    data = to_py(data_r)
    rbase.file_remove(temp_file)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a dataframe, got {type(data)}")

    return data


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
