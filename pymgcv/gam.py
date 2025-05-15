from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import data_to_rdf, rlistvec_to_dict
from pymgcv.terms import TermLike

mgcv = importr("mgcv")
rbase = importr("base")
rutils = importr("utils")
rstats = importr("stats")
base = importr("base")



# TODO, passing arguments to family?


def variables_to_formula(
    dependent: str,
    independent: Iterable[str | TermLike],
) -> str:
    """Convert the variables to an mgcv style formula."""
    return f"{dependent}~" + "+".join(map(str, independent))


@dataclass
class FittedGAM:
    """The result object from fittin a GAM."""

    def __init__(
            self,
            gam: ro.vectors.ListVector,
            dependent: str,
            independent = Iterable[TermLike],
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
        smooth: TermLike,  # TODO rename
        data: dict[str, np.ndarray],
    ):
        for var in smooth.varnames:
            if var not in data:
                raise ValueError(f"Expected {var} to be provided in data.")
        # TODO should I use newdata.guaranteed?
        # TODO also check by variables provided? Anyting else?
        # Manually add missing columns as for some reason mgcv wants them
        all_independent_names = [el for indep in self.independent for el in indep.varnames]
        n = list(data.values())[0].shape[0]
        dummy = {k: np.zeros(n) for k in all_independent_names}  # TODO seems very bug prone?
        data = dummy | data
        exclude = [term.simple_string for term in self.independent if term.simple_string != smooth.simple_string]
    
        # TODO Order not consistent with formula? Do zeroed terms get prepended or somthing?

        predictions = rstats.predict(
            self.rgam,
            newdata=data_to_rdf(data),
            se=True,
            type="terms",
            exclude=exclude,
        )  # Not whether I should expect columns to be removed?

        # TODO This is pretty horrible and should probably be rewritten:
        predictions = rlistvec_to_dict(predictions)  # TODO 
        zeroed_cols = np.all(predictions["fit"] == 0, axis=-2)  # Do these have names?
        predictions["fit"] = predictions["fit"][..., ~zeroed_cols].squeeze()
        predictions["se_fit"] = predictions["se_fit"][..., ~zeroed_cols].squeeze()
        # TODO at least have failsafe that if all columns are zero, to return a vector of zeros?
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
    independent: Iterable[TermLike],
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
