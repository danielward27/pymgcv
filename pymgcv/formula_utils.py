"""Formula utilities."""

from dataclasses import dataclass
from typing import Any

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import to_rpy

rbase = importr("base")
rutils = importr("utils")


@dataclass
class _Var:
    name: str


def _to_r_constructor_string(arg: Any) -> str:
    """Converts an object to R string representation.

    _Var acts as a placeholder for a variable name in R.

    Args:
        arg: The object to convert. If not already an RObject, it is first passed to
            `to_rpy`.
    """
    if isinstance(arg, _Var):
        return arg.name

    if not isinstance(arg, ro.RObject):
        arg = to_rpy(arg)

    connection = rbase.textConnection("__r_obj_str", "w")
    rbase.dput(arg, file=connection)
    rbase.close(connection)
    assert len(ro.r["__r_obj_str"]) == 1
    return ro.r["__r_obj_str"][0]
