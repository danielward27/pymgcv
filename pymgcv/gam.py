from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import to_py, to_rpy

mgcv = importr("mgcv")
rbase = importr("base")
rutils = importr("utils")
rstats = importr("stats")


# TODO, passing arguments to family?
FamilyOptions = Literal[
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
    dependent: str,
    independent: Iterable[str],
) -> str:
    """Convert the variables to an mgcv style formula."""
    return f"{dependent}~" + "+".join(independent)


def gam(
    dependent: str,
    independent: tuple[str, ...],
    data: pd.DataFrame,
    family: FamilyOptions = "gaussian",
):  # TODO missing options.
    # TODO families as functions? e.g. gaussian()
    formula = variables_to_formula(dependent, independent)
    return FittedGAM(mgcv.gam(ro.Formula(formula), data=to_rpy(data), family=family))


@dataclass
class FittedGAM:
    """The result object from fittin a GAM."""

    def __init__(self, gam: ro.vectors.ListVector):
        self.rgam = gam

    def predict(self, data):  # TODO many more options here
        return to_py(rstats.predict(self.rgam, newdata=to_rpy(data)))

    def summary(self) -> str:
        strvec = rutils.capture_output(rbase.summary(self.rgam))
        return "\n".join(tuple(strvec))
