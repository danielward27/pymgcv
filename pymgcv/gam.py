from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import data_to_rdf, rlistvec_to_dict

mgcv = importr("mgcv")
rbase = importr("base")
rutils = importr("utils")
rstats = importr("stats")


# TODO, passing arguments to family?


def variables_to_formula(
    dependent: str,
    independent: Iterable[str],
) -> str:
    """Convert the variables to an mgcv style formula."""
    return f"{dependent}~" + "+".join(independent)


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
            newdata=data_to_rdf(data),
            type=type,
            se=True,
        )
        return rlistvec_to_dict(predictions)

    def summary(self) -> str:
        strvec = rutils.capture_output(rbase.summary(self.rgam))
        return "\n".join(tuple(strvec))

    def formula(self) -> str:
        """Get the mgcv-style formula used to fit the model."""
        return str(self.rgam.rx2("formula"))

    @property
    def coefficients(self) -> np.ndarray:
        """The coefficients from the fit."""
        return rlistvec_to_dict(self.rgam)["coefficients"]


def gam(
    dependent: str,
    independent: Iterable[str],
    data: pd.DataFrame | dict[str, pd.Series | np.ndarray],
    family: str = "gaussian",
) -> FittedGAM:
    """Fit a gam model.

    Args:
        dependent: The dependent variable.
        independent: The independent variables.
        data: A ``pandas.DataFrame`` or a dictionary mapping variable names to arrays or
            pandas series. Using a dictionary is useful for passing matrix variables.
            Factors should be represented using the pandas category dtype.
        family: The family. This is currently passed as a string which will be evaluated
            in R. This can be the name, the name with arguments e.g. ``mvn(d=2)``.
            Defaults to "gaussian".

    Returns:
        The fitted gam model.
    """
    # TODO missing options.
    # TODO families as functions? e.g. gaussian()
    formula = variables_to_formula(dependent, independent)
    family = ro.r(family)
    return FittedGAM(
        mgcv.gam(
            ro.Formula(formula), data=data_to_rdf(data), family="gaussian",
        ),  # TODO
    )


# TODO make data to R and check factor maintained?
