"""More convenient versions."""

from collections.abc import Iterable

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
    data: pd.DataFrame,
    as_array_prefixes: Iterable[str] = (),
) -> ro.vectors.DataFrame:
    """Convert pandas dataframe to an rpy2 dataframe.

    Args:
    data: pandas dataframe to convert.
    as_array_prefixes: prefixes of columns to combine into arrays, which
        e.g. are interpreted as functional effects by mgcv.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    if any(data.dtypes == "object") or any(data.dtypes == "string"):
        raise TypeError("DataFrame contains unsupported object or string types.")

    not_array_colnames = [
        col
        for col in data.columns
        if not any(col.startswith(prefix) for prefix in as_array_prefixes)
    ]
    rpy_df = to_rpy(data[not_array_colnames])

    matrices = {}
    for prefix in as_array_prefixes:
        subset = data.filter(like=prefix)
        matrices[prefix] = base.I(to_rpy(subset.to_numpy()))
    matrices_df = base.data_frame(**matrices)
    if rpy_df.nrow == 0:
        return matrices_df
    if matrices_df.nrow == 0:
        return rpy_df
    return base.cbind(rpy_df, matrices_df)
