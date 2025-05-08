from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import dict_to_rdf, list_vec_to_dict

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
    data: dict[str, np.ndarray],
    family: FamilyOptions = "gaussian",
):

    # Note, if your data is a pandas dataframe, you can convert before fitting, e.g.
    # using {k: v.to_numpy() for k, v in df.items()}
    # TODO missing options.
    # TODO families as functions? e.g. gaussian()

    formula = variables_to_formula(dependent, independent)
    return FittedGAM(
        mgcv.gam(ro.Formula(formula), data=dict_to_rdf(data), family=family),
    )


@dataclass
class FittedGAM:
    """The result object from fittin a GAM."""

    def __init__(self, gam: ro.vectors.ListVector):
        self.rgam = gam

    def predict(
        self,
        data: dict[str, np.ndarray],
        type: Literal["link", "terms"] = "link",  # TODO: add other types when tested.
    ):
        """Compute predictions and standard errors."""
        predictions = rstats.predict(
            self.rgam,
            newdata=dict_to_rdf(data),
            type=type,
            se=True,
        )
        return list_vec_to_dict(predictions)

    def summary(self) -> str:
        strvec = rutils.capture_output(rbase.summary(self.rgam))
        return "\n".join(tuple(strvec))

    def formula(self) -> str:
        """Get the mgcv-style formula used to fit the model."""
        return str(self.rgam.rx2("formula"))
