"""Data conversion utilities for Python-R interoperability.

This module provides convenient functions for converting data between Python
and R representations, particularly for use with rpy2. It handles the conversion
of pandas DataFrames to R data frames and various other Python objects to their
R equivalents, with proper handling of numpy arrays and pandas-specific features.

The conversions are essential for seamless integration with R's mgcv library
while maintaining pythonic data structures on the Python side.
"""

from collections.abc import Iterable

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

base = importr("base")


def to_rpy(x):
    """Convert Python object to R object using rpy2.

    Handles automatic conversion of pandas DataFrames, numpy arrays, and
    other Python objects to their R equivalents using the appropriate
    rpy2 converters.

    Args:
        x: Python object to convert (DataFrame, array, list, etc.)

    Returns:
        R object equivalent of the input, ready for use in R function calls

    Examples:
        ```python
        import pandas as pd
        import numpy as np

        # Convert DataFrame
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        r_df = to_rpy(df)

        # Convert numpy array
        arr = np.array([1, 2, 3])
        r_vec = to_rpy(arr)
        ```
    """
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(x)


def to_py(x):
    """Convert R object to Python object using rpy2.

    Handles automatic conversion of R data structures back to their Python
    equivalents, including R data frames to pandas DataFrames and R vectors
    to numpy arrays.

    Args:
        x: R object to convert (data.frame, vector, list, etc.)

    Returns:
        Python object equivalent of the input

    Examples:
        ```python
        # Convert R vector to numpy array
        r_vector = ro.IntVector([1, 2, 3])
        py_array = to_py(r_vector)  # Returns numpy array

        # Convert R data frame to pandas DataFrame
        r_df = robjects.r('data.frame(x=1:3, y=4:6)')
        py_df = to_py(r_df)  # Returns pandas DataFrame
        ```
    """
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(x)


def rlistvec_to_dict(x: ro.ListVector) -> dict:
    """Convert R ListVector to Python dictionary with pythonic naming.

    Converts an R named list (ListVector) to a Python dictionary, with each
    element converted to appropriate Python types. Dots in R names are replaced
    with underscores to follow Python naming conventions.

    Args:
        x: R ListVector with named elements

    Returns:
        Dictionary with string keys (dots replaced with underscores) and
        values converted to Python equivalents

    Raises:
        ValueError: If the ListVector contains duplicate names, which would
            create an invalid dictionary

    Examples:
        ```python
        # R list with elements: $coef, $fitted.values, $residuals
        r_list = robjects.r('list(coef=1:3, fitted.values=4:6, residuals=7:9)')
        py_dict = rlistvec_to_dict(r_list)
        # Returns: {'coef': array([1,2,3]), 'fitted_values': array([4,5,6]), ...}
        ```

    Note:
        This function is particularly useful for converting mgcv model output,
        which often contains lists with R-style naming conventions.
    """
    if len(x.names) != len(set(x.names)):
        raise ValueError(
            "List vector contained duplicate names, so cannot be "
            "converted to a python dictionary.",
        )
    return {k.replace(".", "_"): to_py(v) for k, v in zip(x.names, x, strict=True)}


def data_to_rdf(
    data: pd.DataFrame | pd.Series,
    as_array_prefixes: Iterable[str] = (),
) -> ro.vectors.DataFrame:
    """Convert pandas DataFrame to R data.frame for use with mgcv.

    Certain columns can be combined into arrays for functional smooth terms.

    Args:
        data: Pandas DataFrame to convert.
        as_array_prefixes: Prefixes of column names to group into arrays.
            Columns matching these prefixes will be combined into R arrays,
            which mgcv can interpret as functional data for specialized
            smooth terms.

    Returns:
        R data.frame object ready for use with mgcv functions

    Raises:
        TypeError: If input is not a pandas DataFrame or Series
    """
    data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame | pd.Series):
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
