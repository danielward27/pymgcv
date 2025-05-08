"""More convenient versions."""

import numpy as np
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


def list_vec_to_dict(x: ro.ListVector) -> dict:
    """Convert a list vector to a dict, with conversion using to_py.

    Dots in names are replaced with underscores, to promote more pythonic naming.
    """
    if len(x.names) != len(set(x.names)):
        raise ValueError(
            "List vector contained duplicate names, so cannot be "
            "converted to a python dictionary.",
        )
    return {k.replace(".", "_"): to_py(v) for k, v in zip(x.names, x, strict=True)}


def dict_to_rdf(d: dict[str, np.ndarray]) -> ro.vectors.DataFrame:
    """Convert a dictionary of arrays to a rpy dataframe."""
    shapes = [arr.shape for arr in d.values()]

    if any(len(s) < 1 or len(s) > 2 for s in shapes):
        raise ValueError("All arrays must be 1D or 2D.")

    if not all(s[0] == shapes[0][0] for s in shapes):
        raise ValueError("All arrays must match on axis 0.")

    d = {k: to_rpy(v) for k, v in d.items()}
    ro.r(
        """
        to_df <- function(a) {
            return(data.frame(lapply(a, function(x) {
                if (is.matrix(x)) I(x) else x
            })))
        }
    """,
    )
    return ro.r["to_df"](ro.ListVector(d))  # type: ignore
