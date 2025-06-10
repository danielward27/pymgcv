from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py
from pymgcv.terms import Offset, TermLike

mgcv = importr("mgcv")
rbase = importr("base")
rutils = importr("utils")
rstats = importr("stats")

# TODO, passing arguments to family?


def _get_intercepts_and_se(rgam):
    coef = rgam.rx2("coefficients")
    vp_diag = np.diag(np.array(rgam.rx2("Vp")))
    coef_names = list(coef.names)
    assert len(coef_names) == len(vp_diag)
    intercept_names = [n for n in coef_names if n.startswith("(Intercept)")]

    return {
        name: {
            "fit": coef[coef_names.index(name)],
            "se": np.sqrt(vp_diag[coef_names.index(name)]).item(),
        }
        for name in intercept_names
    }


@dataclass
class ModelSpecification:
    """A class holding the specification of a GAM model.

    Args:
        mean_predictors: A dictionary mapping dependent variable names
            to a list of ``TermLike`` objects used for modelling that
            variable. For univariate models (a single dependent variable), this is a
            dictionary with a single key-value pair.
        other_predictors: A dictionary mapping family parameter labels to
            a list of terms used to model the parameter. The keys of the dictionary
            are used solely as labels/names e.g. during predictions. The dictionary
            should match the order of the parameters in the mgcv family specified.
        family: The MGCV family of the model. This is currently evaluated as literal
            R code.
    """

    mean_predictors: dict[str, list[TermLike]]
    other_predictors: dict[str, list[TermLike]] = field(default_factory=dict)
    family: str = "gaussian"

    def __post_init__(self):
        if len(self.mean_predictors) > 1 and self.other_predictors:
            raise ValueError(  # TODO I assume this is possible in mgcv
                "Simultaneous use of multiple dependent variables and predictors of "
                "family parameters is not yet supported.",
            )

    @property
    def all_formulae(self) -> dict[str, list[TermLike]]:
        """All formulae (response and for family parameters)"""
        return self.mean_predictors | self.other_predictors

    def _check_valid_data(
        self,
        data: pd.DataFrame,
    ) -> None:
        """Checks data is compatible with the specification."""
        all_terms: list[TermLike] = []
        for terms in (self.all_formulae).values():
            all_terms.extend(terms)

        for term in all_terms:
            for varname in term.varnames:
                if varname not in data.columns:
                    raise ValueError(f"Variable {varname} not found in data.")

            if term.by is not None:
                if term.by not in data.columns:
                    raise ValueError(f"Variable {term.by} not found in data.")

                if data[term.by].dtype == "category":
                    raise TypeError(
                        """Categorical by variables not yet supported. The reason for
                        this is that these implicitly expand into multiple terms in the
                        model, which leads to complications as the model terms no longer
                        have a one to one mapping to the predicted terms. This behaviour
                        will be supported as needed. For now, see
                        ``smooth_by_factor``.""",
                    )
        # TODO error if any varnames clash with "intercept".

    def _to_r_formulae(self) -> ro.Formula | list[ro.Formula]:
        formulae = []
        for dependent, terms in self.mean_predictors.items():
            formulae.append(ro.Formula(_terms_to_formula_str(dependent, terms)))

        for terms in self.other_predictors.values():
            formulae.append(ro.Formula(_terms_to_formula_str("", terms)))

        return formulae if len(formulae) > 1 else formulae[0]


def _terms_to_formula_str(
    dependent: str,
    terms: Iterable[TermLike],
) -> str:
    """Convert a list of variables to an mgcv style formula."""
    return f"{dependent}~" + "+".join(map(str, terms))


@dataclass
class FittedGAM:
    """The result object from fitting a GAM."""

    rgam: ro.vectors.ListVector
    data: pd.DataFrame
    model_specification: ModelSpecification

    def predict(
        self,
        data: pd.DataFrame,
    ):
        """Compute predictions and standard errors."""
        self.model_specification._check_valid_data(data)
        predictions = rstats.predict(
            self.rgam,
            newdata=data_to_rdf(data),
            se=True,
        )
        predictions = rlistvec_to_dict(predictions)

        all_targets = (
            self.model_specification.mean_predictors
            | self.model_specification.other_predictors
        ).keys()
        fit = pd.DataFrame(predictions["fit"], columns=list(all_targets))
        se = pd.DataFrame(predictions["se_fit"], columns=list(all_targets))

        # TODO we assume 1 column for each linear predictor
        return pd.concat({"fit": fit, "se": se}, axis=1)

    def partial_effects(
        self,
        data: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        # TODO way to filter terms?
        """Compute partial effects and standard errors.

        The returned structure is a dictionary with keys matching the target
        variables (dependent variables of family parameters), and values are
        hierarcahical dataframes, with two top level columns "se" and "fit",
        and the lower level columns representing the components of the linear
        predictor (including offsets and intercepts).

        """
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
        # Partition results based on formulas
        formulae = (
            self.model_specification.mean_predictors
            | self.model_specification.other_predictors
        )
        results = {}

        formulae_without_offset = {
            k: [term for term in terms if not isinstance(term, Offset)]
            for k, terms in formulae.items()
        }
        for i, (name, terms) in enumerate(formulae_without_offset.items()):
            names_with_idx = [term.simple_string(i) for term in terms]
            names = [term.simple_string() for term in terms]
            rename = {
                with_idx: without
                for with_idx, without in zip(names_with_idx, names, strict=True)
            }
            results[name] = pd.concat(
                {
                    "fit": fit[names_with_idx]
                    .rename(columns=rename, errors="raise")[names]
                    .copy(),
                    "se": se[names_with_idx]
                    .rename(columns=rename, errors="raise")[names]
                    .copy(),
                },
                axis=1,
            )

        # Add offset terms
        for k, terms in formulae.items():
            for term in terms:
                if isinstance(term, Offset):
                    results[k]["fit", term.simple_string()] = data[term.varnames[0]]
                    results[k]["se", term.simple_string()] = 0

        # Add intercepts
        intercepts = _get_intercepts_and_se(self.rgam)
        formula_names = list(results.keys())
        for name, vals in intercepts.items():
            idx = 0 if name == "(Intercept)" else int(name.split(".")[-1])
            formula_name = formula_names[idx]
            results[formula_name]["fit", "intercept"] = vals["fit"]
            results[formula_name]["se", "intercept"] = vals["se"]

        return {k: v.sort_index(axis=1) for k, v in results.items()}

    def partial_effect(
        self,
        target: str,
        term: TermLike,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute the partial effect for a specific term.

        Args:
            target (str): The target variable (response variable name or
                name given to family parameter).
            term (TermLike): The term to compute the partial effect for.
            data (pd.DataFrame): The data to compute the partial effect using.

        Returns:
            pd.DataFrame: Dataframe with columns "fit" (the effect) and "se"
                (standard errors).
        """
        effect, se = term._partial_effect(data, self, target)
        return pd.DataFrame({"fit": effect, "se": se})

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


def gam(
    specification: ModelSpecification,
    data: pd.DataFrame,
) -> FittedGAM:
    """Fit a gam model.

    Args:
        family: The family. This is currently passed as a string which will be evaluated
            in R. This can be the name, the name with arguments e.g. ``mvn(d=2)``.
            Defaults to "gaussian".

    Returns:
        The fitted gam model.
    """
    # TODO missing options.
    # TODO families as functions? e.g. gaussian()
    specification._check_valid_data(data)

    return FittedGAM(
        mgcv.gam(
            specification._to_r_formulae(),
            data=data_to_rdf(data),
            family=ro.rl(specification.family),
        ),
        data=deepcopy(data),
        model_specification=specification,
    )
