import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.rpy_utils import to_py

rutils = importr("utils")


def get_data(name: str):
    """Get built-in R dataset.

    Currently assumes that the dataset is a dataframe.
    """
    with ro.local_context() as lc:
        rutils.data(ro.rl(name), envir=lc)
        return to_py(lc[name])
