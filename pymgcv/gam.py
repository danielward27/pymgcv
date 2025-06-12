"""Core GAM fitting and model specification functionality.

This module provides the main interface for fitting Generalized Additive Models (GAMs)
using R's mgcv library through rpy2. It includes classes for model specification,
fitted model objects, and the main fitting function.

The module handles:
- Model specification with flexible term definitions
- GAM fitting with various smoothing parameter estimation methods
- Predictions and partial effects computation
- Model diagnostics and summaries
"""

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


@dataclass
class ModelSpecification:
    r"""Defines the model to use.

    This class encapsulates the GAM model specification, including the
    family, and the terms for modeling response variable(s) and family parameters.

    Args:
        response_predictors: Dictionary mapping response variable names to lists of
            [`TermLike`][pymgcv.terms.TermLike] objects used to predict $g([\mathbb{E}[Y])$  For single response
            models, use a single key-value pair. For multivariate models, include
            multiple response variables.
        other_predictors: Dictionary mapping family parameter names to lists of
            terms for modeling those parameters. Keys are used as labels during
            prediction and should match the order expected by the mgcv family.
            Common examples include 'scale' for location-scale models.
        family: String specifying the mgcv family for the error distribution.
            This is passed directly to R's mgcv and can include family arguments.
    """

    response_predictors: dict[str, list[TermLike]]
    other_predictors: dict[str, list[TermLike]] = field(default_factory=dict)
    family: str = "gaussian"

    def __post_init__(self):
        if len(self.response_predictors) > 1 and self.other_predictors:
            raise ValueError(  # TODO I assume this is possible in mgcv
                "Simultaneous use of multiple dependent variables and predictors of "
                "family parameters is not yet supported.",
            )

        for terms in self.all_formulae.values():
            within_formula_names = set()
            for term in terms:
                name = term.simple_string()
                if name in within_formula_names:
                    raise ValueError(
                        f"Duplicate term name '{name}' found in formulae. "
                        "pymgcv does not support this. If you believe to have "
                        "a legitimate use case for this, try duplicating the term "
                        "and renaming it.",
                    )
                within_formula_names.add(name)

    @property
    def all_formulae(self) -> dict[str, list[TermLike]]:
        """All formulae (response and for family parameters)"""
        return self.response_predictors | self.other_predictors

    def _check_valid_data(
        self,
        data: pd.DataFrame,
    ) -> None:
        """Validate that data contains all variables required by the model specification.

        Performs comprehensive validation including:
        - Checking that all term variables exist in the data
        - Validating 'by' variables are present
        - Ensuring categorical 'by' variables are not used (not yet supported)
        - Checking for reserved variable names that conflict with mgcv

        Args:
            data: DataFrame containing the modeling data

        Raises:
            ValueError: If required variables are missing from data
            TypeError: If categorical 'by' variables are detected (unsupported)
        """
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
        """Convert the model specification to R formula objects.

        Creates mgcv-compatible formula objects from the Python specification.
        For single-formula models, returns a single Formula object. For
        multi-formula models (multiple responses or family parameters),
        returns a list of Formula objects.

        Returns:
            Single Formula object for simple models, or list of Formula objects
            for multi-formula models. The order matches mgcv's expectations:
            response formulae first, then family parameter formulae.
        """
        formulae = []
        for dependent, terms in self.response_predictors.items():
            formulae.append(ro.Formula(f"{dependent}~{'+'.join(map(str, terms))}"))

        for terms in self.other_predictors.values():
            formulae.append(ro.Formula(f"~{'+'.join(map(str, terms))}"))

        return formulae if len(formulae) > 1 else formulae[0]


@dataclass
class FittedGAM:
    """The fitted GAM model with methods for predicting and analyzing.

    Generally returned by fitting methods (rather than directly).

    Args:
        rgam: The underlying R mgcv model object from fitting
        data: Original DataFrame used for model fitting
        model_specification: The ModelSpecification used to create this model
    """

    rgam: ro.vectors.ListVector
    data: pd.DataFrame
    model_specification: ModelSpecification

    def predict(
        self,
        data: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Compute model predictions with uncertainty estimates.

        Makes predictions for new data using the fitted GAM model. Predictions
        are returned on the link scale (linear predictor scale), not the response
        scale. For response scale predictions, apply the appropriate inverse link
        function to the results.

        Args:
            data: DataFrame containing predictor variables. Must include all
                variables referenced in the original model specification.

        Returns:
            DataFrame with hierarchical columns:
            - Top level: 'fit' (predictions) and 'se' (standard errors)
            - Second level: target variable names (response or family parameter names)

            For single response models:
                columns = [('fit', 'response_name'), ('se', 'response_name')]
            For multi-target models:
                columns = [('fit', 'target1'), ('se', 'target1'),
                          ('fit', 'target2'), ('se', 'target2'), ...]

        Raises:
            ValueError: If required variables are missing from the input data

        Example:
            ```python
            predictions = model.predict(new_data)
            print(predictions[('fit', 'y')])  # Predictions for response 'y'
            print(predictions[('se', 'y')])   # Standard errors for 'y'
            ```
        """
        self.model_specification._check_valid_data(data)
        predictions = rstats.predict(
            self.rgam,
            newdata=data_to_rdf(data),
            se=True,
        )
        predictions = rlistvec_to_dict(predictions)

        all_targets = self.model_specification.all_formulae.keys()

        # TODO we assume 1 column for each linear predictor
        n = data.shape[0]
        return {
            target: pd.DataFrame(
                {
                    "fit": predictions["fit"].reshape(n, -1)[:, i],
                    "se": predictions["se_fit"].reshape(n, -1)[:, i],
                },
            )
            for i, target in enumerate(all_targets)
        }

    def partial_effects(
        self,
        data: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Compute partial effects for all model terms.

        Calculates the contribution of each model term to the overall prediction.
        This decomposition is useful for understanding which terms contribute most
        to predictions and for creating partial effect plots.

        Args:
            data: DataFrame containing predictor variables for evaluation

        Returns:
            Dictionary mapping target variable names to DataFrames with partial effects.
            Each DataFrame has hierarchical columns:
            - Top level: 'fit' (partial effects) and 'se' (standard errors)
            - Second level: term names (e.g., 's(x1)', 'x2', 'intercept')

            The sum of all fit columns equals the total prediction:
            ```python
            effects = model.partial_effects(data)
            total_effect = effects['y']['fit'].sum(axis=1)
            predictions = model.predict(data)[('fit', 'y')]
            assert np.allclose(total_effect, predictions)
            ```

        Example:
            ```python
            effects = model.partial_effects(data)

            # Get contribution of smooth term s(x1) to response y
            smooth_contribution = effects['y'][('fit', 's(x1)')]

            # Get standard error of the smooth term
            smooth_se = effects['y'][('se', 's(x1)')]

            # Plot partial effect
            plt.plot(data['x1'], smooth_contribution)
            plt.fill_between(data['x1'],
                           smooth_contribution - 2*smooth_se,
                           smooth_contribution + 2*smooth_se,
                           alpha=0.3)
            ```

        Note:
            Partial effects include all model components: smooth terms, linear terms,
            interactions, offsets, and intercepts. The effects are centered around
            their mean values following mgcv conventions.
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
            results[formula_name] = results[formula_name].sort_index(axis=1)
            results[formula_name]["se", "intercept"] = vals["se"]

        return {k: v.sort_index(axis=1) for k, v in results.items()}

    def partial_effect(
        self,
        target: str,
        term: TermLike,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute the partial effect for a single model term.

        This method efficiently computes the contribution of one specific term
        to the model predictions. It's more efficient than computing all partial
        effects when you only need one term.

        Args:
            target: Name of the target variable (response variable or family
                parameter name from the model specification)
            term: The specific term to evaluate (must match a term used in the
                original model specification)
            data: DataFrame containing the predictor variables needed for the term

        Returns:
            DataFrame with columns:
            - 'fit': The partial effect values (contribution of this term)
            - 'se': Standard errors of the partial effect

        Raises:
            KeyError: If the target name is not found in the model
            ValueError: If the term is not found in the model specification

        Example:
            ```python
            # Get partial effect of smooth term on response
            effect = model.partial_effect('y', Smooth('x1'), data)

            # Plot the partial effect
            import matplotlib.pyplot as plt
            plt.plot(data['x1'], effect['fit'])
            plt.fill_between(data['x1'],
                           effect['fit'] - 2*effect['se'],
                           effect['fit'] + 2*effect['se'],
                           alpha=0.3)
            plt.xlabel('x1')
            plt.ylabel('Partial effect')
            ```

        Note:
            The partial effect represents the contribution of this term alone,
            with all other terms held at their reference values (typically zero
            for centered smooth terms).
        """
        effect, se = term._partial_effect(data, self, target)
        return pd.DataFrame({"fit": effect, "se": se})

    def partial_residuals(
        self,
        target: str,
        term: TermLike,
        data: pd.DataFrame,
    ) -> pd.Series:
        """Compute partial residuals for model diagnostic plots.

        Partial residuals combine the fitted values from a specific term with
        the overall model residuals. They're useful for assessing whether the
        chosen smooth function adequately captures the relationship, or if a
        different functional form might be more appropriate.

        The partial residuals are calculated as:
        partial_residuals = observed - (total_fitted - term_effect)
                          = observed - total_fitted + term_effect
                          = residuals + term_effect

        Args:
            target: Name of the response variable
            term: The model term to compute partial residuals for
            data: DataFrame containing the data (must include the response variable)

        Returns:
            Series containing the partial residuals for the specified term

        Example:
            ```python
            # Compute partial residuals for smooth term
            partial_resid = model.partial_residuals('y', Smooth('x1'), data)

            # Plot partial residuals to assess model fit
            import matplotlib.pyplot as plt
            plt.scatter(data['x1'], partial_resid, alpha=0.6)

            # Overlay the fitted smooth for comparison
            smooth_effect = model.partial_effect('y', Smooth('x1'), data)
            plt.plot(data['x1'], smooth_effect['fit'], 'red', linewidth=2)
            plt.xlabel('x1')
            plt.ylabel('Partial residuals')
            plt.title('Partial residual plot for s(x1)')
            ```

        Note:
            If the partial residuals show systematic patterns around the fitted
            smooth curve, it may indicate that a different functional form or
            additional model terms are needed.
        """
        partial_effect = self.partial_effect(target, term, data)["fit"]
        fit = self.predict(data)["fit"][target]
        y = data[target]
        return (y - fit) + partial_effect

    def summary(self) -> str:
        """Generate a comprehensive summary of the fitted GAM model.

        Produces a detailed summary including parameter estimates, significance
        tests, smooth term information, model fit statistics, and convergence
        diagnostics. The output matches the format of R's mgcv summary.
        """
        strvec = rutils.capture_output(rbase.summary(self.rgam))
        return "\n".join(tuple(strvec))

    @property
    def coefficients(self) -> np.ndarray:  # TODO consider returning as dict?
        """Extract model coefficients from the fitted GAM."""
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
    """Fit a Generalized Additive Model.

    Args:
        specification: ModelSpecification object defining the model structure,
            including terms for response variables and family parameters, plus
            the error distribution family
        data: DataFrame containing all variables referenced in the specification.
            Variable names must match those used in the model terms.
        method: Method for smoothing parameter estimation, matching the mgcv,
            options, including:
            - "GCV.Cp": Generalized Cross Validation (default, recommended)
            - "REML": Restricted Maximum Likelihood (good for mixed models)

    Returns:
        FittedGAM object containing the fitted model and methods for prediction,
            analysis.
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


def _check_valid_varnames(varnames: Iterable[str]) -> None:
    """Validate variable names don't conflict with mgcv syntax.

    Checks that variable names in the dataset don't contain strings that
    could be interpreted as mgcv formula syntax, which would cause parsing
    errors or unexpected behavior.

    Args:
        varnames: Iterable of variable names to validate

    Raises:
        ValueError: If any variable name contains reserved mgcv syntax
    """
    disallowed = ["Intercept", "intercept", "s(", "te(", "ti(", "t2(", ":", "*"]

    for var in varnames:
        if any(dis in var for dis in disallowed):
            raise ValueError(
                f"Variable name '{var}' risks clashing with terms generated by mgcv, "
                "please rename this variable.",
            )
