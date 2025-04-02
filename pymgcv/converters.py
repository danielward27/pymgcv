"More convenient versions."

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri


def to_rpy(x):
    """Convert python object to rpy."""
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(x)


def to_py(x):
    """Convert rpy object to python."""
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(x)
