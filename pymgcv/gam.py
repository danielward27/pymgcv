from typing import Literal

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import to_rpy
from pymgcv.smooths import Smooth

mgcv = importr("mgcv")


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


def variables_to_formula(dependent: str, independent: tuple[str | Smooth, ...]) -> str:
    return f"{dependent}~" + "+".join(str(var) for var in independent)


def gam(
    dependent: str,
    independent: tuple[str | Smooth, ...],
    data: pd.DataFrame,
    family: FamilyOptions = "gaussian",
):  # TODO many more options not included currently.
    # TODO purpose for families as functions? e.g. gaussian()
    formula = variables_to_formula(dependent, independent)
    return mgcv.gam(ro.Formula(formula), data=to_rpy(data), family=family)
    # TODO result format.
