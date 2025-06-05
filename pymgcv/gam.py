from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py
from pymgcv.terms import Smooth, TensorSmooth, TermLike

mgcv = importr("mgcv")
rbase = importr("base")
rutils = importr("utils")
rstats = importr("stats")


# TODO, passing arguments to family?


def terms_to_formula(
    dependent: str,
    terms: Iterable[str | TermLike],
) -> str:
    """Convert the variables to an mgcv style formula."""
    return f"{dependent}~" + "+".join(map(str, terms))


@dataclass
class FittedGAM:
    """The result object from fittin a GAM."""

    rgam: ro.vectors.ListVector
    dependent: str
    terms: tuple[TermLike, ...]
    data: pd.DataFrame

    def __init__(
        self,
        rgam: ro.vectors.ListVector,
        dependent: str,
        terms: Iterable[TermLike],
        data: pd.DataFrame,
    ):
        self.rgam = rgam
        self.dependent = dependent
        self.terms = tuple(terms)
        self.data = data

        for term in self.terms:
            if isinstance(term, Smooth | TensorSmooth) and term.by is not None:
                if data[term.by].dtype == "category":
                    raise TypeError(
                        """Categorical by variables not yet supported. The reason for
                        this is that these implicitly expand into multiple terms in the
                        model, which leads to complications as the model terms no longer
                        have a one to one mapping to the predicted terms. This behaviour
                        will be supported as needed.""",
                    )

    def predict(
        self,
        data: pd.DataFrame,
    ):
        """Compute predictions and standard errors."""
        predictions = rstats.predict(
            self.rgam,
            newdata=data_to_rdf(data),
            se=True,
        )
        return rlistvec_to_dict(predictions)

    def predict_terms(
        self,
        data: pd.DataFrame,
    ):
        """Compute predictions and standard errors."""
        predictions = rstats.predict(
            self.rgam,
            newdata=data_to_rdf(data),
            se=True,
            type="terms",
            newdata_gauranteed=True,
        )
        fit = pd.DataFrame(
            to_py(predictions.rx2["fit"]),
            columns=to_py(rbase.colnames(predictions.rx2["fit"])),
        )
        se = pd.DataFrame(
            to_py(predictions.rx2["se.fit"]),
            columns=to_py(rbase.colnames(predictions.rx2["se.fit"])),
        )
        return {"fit": fit, "se": se}

    def predict_term(
        self,
        term: TermLike,
        data: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Convenience predictor for a single term of the model."""
        for var in term.varnames:
            if var not in data:
                raise ValueError(f"Expected {var} to be provided in data.")
        # TODO should I check the variables provided?
        # Manually add missing columns as for some reason mgcv wants them
        all_independent_names = [el for indep in self.terms for el in indep.varnames]
        for name in all_independent_names:
            if name not in data:
                data[name] = np.zeros(len(data))
        exclude = [
            t.simple_string for t in self.terms if t.simple_string != term.simple_string
        ]

        predictions = rstats.predict(  # TODO exclude vs passing terx
            self.rgam,
            newdata=data_to_rdf(data),
            type="terms",
            exclude=exclude if exclude else ro.NULL,
            se=True,
        )
        fit = pd.DataFrame(
            predictions.rx2["fit"],
            columns=np.array([term.simple_string]),
        )
        se = pd.DataFrame(
            predictions.rx2["se.fit"],
            columns=np.array([term.simple_string]),
        )
        return {"fit": fit, "se": se}

    def summary(self) -> str:
        """Get the summary of the gam model.

        Usually this should be combined with print, i.e. ``print(model.summary())``.
        """
        strvec = rutils.capture_output(rbase.summary(self.rgam))
        return "\n".join(tuple(strvec))

    def formula(self) -> str:
        """Get the mgcv-style formula used to fit the model."""
        return str(self.rgam.rx2("formula"))

    @property
    def coefficients(self) -> np.ndarray:
        """The coefficients from the fit."""
        return rlistvec_to_dict(self.rgam)["coefficients"]

    def partial_residuals(
        self,
        term: TermLike,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """Get the partial residuals for a term."""
        term_predict = self.predict_term(term, data)["fit"]
        total_predict = self.predict(data)["fit"]
        y = data[self.dependent]
        return (y - total_predict) + term_predict


def gam(
    dependent: str,
    terms: Iterable[TermLike],
    data: pd.DataFrame,
    family: str = "gaussian",
) -> FittedGAM:
    """Fit a gam model.

    Args:
        dependent: The dependent variable.
        terms: The terms to use for fitting the model.
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
    formula = terms_to_formula(dependent, terms)
    ro.rl(family)
    return FittedGAM(
        mgcv.gam(
            ro.Formula(formula),
            data=data_to_rdf(data),
            family=family,
        ),  # TODO
        dependent=dependent,
        terms=terms,
        data=deepcopy(data),
    )


# TODO make data to R and check factor maintained?
