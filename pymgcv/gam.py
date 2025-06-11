from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py
from pymgcv.rgam_utils import _get_intercepts_and_se
from pymgcv.terms import Offset, TermLike

mgcv = importr("mgcv")
rbase = importr("base")
rutils = importr("utils")
rstats = importr("stats")

# TODO, passing arguments to family?


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

        all_term_names = set()
        for terms in self.all_formulae.values():
            for term in terms:
                name = term.simple_string()
                if name in all_term_names:
                    raise ValueError(
                        f"Duplicate term name '{name}' found in formulae. "
                        "pymgcv does not support this. If you believe to have "
                        "a legitimate use case for this, try duplicating the term "
                        "and renaming it.",
                    )
                all_term_names.add(term.simple_string())

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
            formulae.append(ro.Formula(f"{dependent}~{'+'.join(map(str, terms))}"))

        for terms in self.other_predictors.values():
            formulae.append(ro.Formula(f"~{'+'.join(map(str, terms))}"))

        return formulae if len(formulae) > 1 else formulae[0]


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
        """Compute predictions and standard errors.

        Predictions are returned on the link scale (i.e. the result of the linear
        predictor).
        """
        self.model_specification._check_valid_data(data)
        predictions = rstats.predict(
            self.rgam,
            newdata=data_to_rdf(data),
            se=True,
        )
        predictions = rlistvec_to_dict(predictions)

        all_targets = self.model_specification.all_formulae.keys()
        fit = pd.DataFrame(predictions["fit"], columns=pd.Index(all_targets))
        se = pd.DataFrame(predictions["se_fit"], columns=pd.Index(all_targets))

        # TODO we assume 1 column for each linear predictor
        return pd.concat({"fit": fit, "se": se}, axis=1)

    def partial_effects(
        self,
        data: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        # TODO way to filter terms?
        """Compute partial effects and standard errors.

        The returned structure is a dictionary with keys matching the target
        variables (dependent variables or family parameters), and values are
        hierarcahical dataframes, with two top level columns "se" and "fit",
        and the lower level columns representing the components of the linear
        predictor (including offsets and intercepts).

        Summing over the columns of the "fit" dataframe will match the values
        produced by the `predict` method, i.e.
        ```
        partial_effects[target]["fit"].sum(axis=1)
        ```

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
        formulae = self.model_specification.all_formulae
        results = {}

        formulae_without_offset = {
            k: [term for term in terms if not isinstance(term, Offset)]
            for k, terms in formulae.items()
        }
        for i, (name, terms) in enumerate(formulae_without_offset.items()):
            rename = {term.simple_string(i): term.simple_string() for term in terms}
            results[name] = pd.concat(
                {
                    "fit": fit.rename(columns=rename, errors="raise")[
                        rename.values()
                    ].copy(),
                    "se": se.rename(columns=rename, errors="raise")[
                        rename.values()
                    ].copy(),
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

        For a single term, this provides a more efficient implementation than
        computing for the full model.

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

    def partial_residuals(
        self,
        target: str,  # TODO dependent/resonse
        term: TermLike,
        data: pd.DataFrame,
    ):
        """Compute the partial residuals for a specific term.

        Args:
            target (str): The target variable (response variable) name.
            term (TermLike): The term to compute the partial effect for.
            data (pd.DataFrame): The data to compute the partial effect using.
        """
        partial_effect = self.partial_effect(target, term, data)["fit"]
        fit = self.predict(data)["fit"][target]
        y = data[target]
        return (y - fit) + partial_effect

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


FitMethodOptions = Literal[
    "GCV.Cp",
    "GACV.Cp",
    "NCV",
    "QNCV",
    "REML",
    "P-REML",
    "ML",
    "P-ML",
]


def gam(
    specification: ModelSpecification,
    data: pd.DataFrame,
    method: FitMethodOptions = "GCV.Cp",
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
    specification._check_valid_data(data)
    _check_valid_varnames(data.columns)

    return FittedGAM(
        mgcv.gam(
            specification._to_r_formulae(),
            data=data_to_rdf(data),
            family=ro.rl(specification.family),
            method=method,
        ),
        data=data.copy(),
        model_specification=specification,
    )


def _check_valid_varnames(varnames: Iterable[str]):
    disallowed = ["Intercept", "intercept", "s(", "te(", "ti(", "t2(", ":", "*"]

    for var in varnames:
        if any(dis in var for dis in disallowed):
            raise ValueError(
                f"Variable name '{var}' risks clashing with terms generated by mgcv, "
                "please rename this variable.",
            )
