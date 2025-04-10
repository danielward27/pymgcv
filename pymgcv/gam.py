from typing import Literal

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv import Smooth
from pymgcv.converters import to_rpy
from pymgcv.smooth import Smooth, TensorSmooth

mgcv = importr("mgcv")
importr("base")


# TODO, passing arguments to family?
type FamilyOptions = Literal[
    "gaussian",
    "Tweedie",
    "negbin",
    "negbin",
    "betar",
    "cnorm",
    "nb",
    "ocat",
    "scat",
    "tw",
    "ziP",
    "gfam",
    "cox.ph",
    "gammals",
    "gaulss",
    "gevlss",
    "gumbls",
    "multinom",
    "mvn",
    "shash",
    "twlss",
    "ziplss",
]


def variables_to_formula(
    dependent: str, independent: tuple[str | Smooth | TensorSmooth, ...]
) -> str:
    return f"{dependent}~" + "+".join(str(var) for var in independent)


def gam(
    dependent: str,
    independent: tuple[str, Smooth | TensorSmooth, ...],
    data: pd.DataFrame,
    family: FamilyOptions = "gaussian",
):  # TODO many more options not included currently.
    # TODO purpose for families as functions? e.g. gaussian()
    formula = variables_to_formula(dependent, independent)
    return mgcv.gam(ro.Formula(formula), data=to_rpy(data), family=family)
    # TODO result format.


# class GamResult:

#     def __init__(self, gam: ro.vectors.ListVector):
#         self.gam = gam

#     def coefficients
