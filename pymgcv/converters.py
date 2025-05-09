"""More convenient versions."""

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

base = importr("base")


def to_rpy(x):
    """Convert python object to rpy."""
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(x)


def to_py(x):
    """Convert rpy object to python."""
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(x)


def rlistvec_to_dict(x: ro.ListVector) -> dict:
    """Convert a list vector to a dict, with conversion using to_py.

    Dots in names are replaced with underscores, to promote more pythonic naming.
    """
    if len(x.names) != len(set(x.names)):
        raise ValueError(
            "List vector contained duplicate names, so cannot be "
            "converted to a python dictionary.",
        )
    return {k.replace(".", "_"): to_py(v) for k, v in zip(x.names, x, strict=True)}


def data_to_rdf(
    data: pd.DataFrame | dict[str, pd.Series | np.ndarray],
) -> ro.vectors.DataFrame:
    """Convert data to an rpy2 dataframe.

    Data can be either a dataframe, or a dictionary mapping from strings to arrays
    or pandas series. The latter is occasionally useful when users wish to have
    matrix variables which cannot be added as a dataframe column.
    """
    if isinstance(data, pd.DataFrame):
        return to_rpy(data)

    shapes = [arr.shape for arr in data.values()]

    if any(len(s) < 1 or len(s) > 2 for s in shapes):
        raise ValueError("All data must be 1D or 2D (i.e. vector or matrix).")

    if not all(s[0] == shapes[0][0] for s in shapes):
        raise ValueError("All data must match on axis 0.")

    data = {k: to_rpy(v) for k, v in data.items()}

    with ro.local_context() as env:
        env["list_vec"] = ro.ListVector(data)
        return ro.r(
            """
            data.frame(lapply(list_vec, function(x) {
                if (is.matrix(x)) I(x) else x
            }))
            """,
        )  # type: ignore
