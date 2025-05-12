from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import data_to_rdf, rlistvec_to_dict
from pymgcv.smooth import SmoothProtocol

mgcv = importr("mgcv")
rbase = importr("base")
rutils = importr("utils")
rstats = importr("stats")
base = importr("base")



# TODO, passing arguments to family?


def variables_to_formula(
    dependent: str,
    independent: Iterable[str | SmoothProtocol],
) -> str:
    """Convert the variables to an mgcv style formula."""
    return f"{dependent}~" + "+".join(map(str, independent))


@dataclass
class FittedGAM:
    """The result object from fittin a GAM."""

    def __init__(
            self,
            gam: ro.vectors.ListVector,
            dependent: str = str,
            independent = Iterable[SmoothProtocol],
            ):
        self.rgam = gam
        self.dependent = dependent
        self.independent = independent

    def predict(
        self,
        data: dict[str, np.ndarray],
    ):
        """Compute predictions and standard errors."""
        predictions = rstats.predict(
            self.rgam,
            newdata=data_to_rdf(data),
            se=True,
        )
        return rlistvec_to_dict(predictions)
    
    def predict_term(
        self,
        smooth: SmoothProtocol,
        data: dict[str, np.ndarray],
    ):
        for var in smooth.varnames:
            if var not in data:
                raise ValueError(f"Expected {var} to be provided in data.")
        # TODO should I use exclude or newdata.guaranteed?

        # TODO also check by variables provided? Anyting else?
        # Manually add missing columns as for some reason mgcv wants them
        all_independent_names = [el for indep in self.independent for el in indep.varnames]
        n = list(data.values())[0].shape[0]
        dummy = {k: np.zeros(n) for k in all_independent_names}  # TODO seems very bug prone?
        data = dummy | data
        
        predictions = rstats.predict(
            self.rgam,
            newdata=data_to_rdf(data),
            se=True,
            type="terms",
        )

        # TODO This is pretty horrible and should probably be rewritten:
        predictions = rlistvec_to_dict(predictions)
        zeroed_cols = np.all(predictions["fit"] == 0, axis=-2)  # Do these have names?
        predictions["fit"] = predictions["fit"][..., ~zeroed_cols].squeeze()
        predictions["se_fit"] = predictions["se_fit"][..., ~zeroed_cols].squeeze()
        return predictions
        

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
    independent: Iterable[SmoothProtocol],
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
        dependent=dependent,
        independent=independent,
    )


# TODO make data to R and check factor maintained?
