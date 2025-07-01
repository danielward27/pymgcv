"""Core GAM fitting and model specification functionality.

This module provides the main interface for fitting Generalized Additive Models (GAMs)
using R's mgcv library through rpy2. It includes classes for model specification,
fitted model objects, and the main fitting function.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py, to_rpy
from pymgcv.terms import Intercept, Offset, TermLike

mgcv = importr("mgcv")
rbase = importr("base")
rutils = importr("utils")
rstats = importr("stats")

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


@dataclass
class GAM:
    r"""Defines the model to use and provides a fit method.

    This class encapsulates the GAM model specification, including the
    family, and the terms for modeling response variable(s) and family parameters.

    Args:
        predictors: Dictionary mapping response variable names to lists of
            [`TermLike`][pymgcv.terms.TermLike] objects used to predict
            $g([\mathbb{E}[Y])$ For single response models, use a single key-value pair.
            For multivariate models, include multiple response variables.
        family_predictors: Dictionary mapping family parameter names to lists of
            terms for modeling those parameters. Keys are used as labels during
            prediction and should match the order expected by the mgcv family.
        family: String specifying the mgcv family for the error distribution.
            This is passed directly to R's mgcv and can include family arguments.
    """

    predictors: dict[str, list[TermLike]]
    family_predictors: dict[str, list[TermLike]]
    family: str

    def __init__(
        self,
        predictors: dict[str, list[TermLike] | TermLike],
        family_predictors: dict[str, list[TermLike] | TermLike] | None = None,
        *,
        family: str = "gaussian",
        add_intercepts: bool = True,
    ):
        predictors, family_predictors = deepcopy((predictors, family_predictors))
        family_predictors = {} if family_predictors is None else family_predictors

        def _ensure_list_of_terms(d):
            return {k: [v] if isinstance(v, TermLike) else v for k, v in d.items()}

        self.predictors = _ensure_list_of_terms(predictors)
        self.family_predictors = _ensure_list_of_terms(family_predictors)
        self.family = family

        if add_intercepts:
            for v in self.all_predictors.values():
                v.append(Intercept())

    def __post_init__(self):
        for terms in self.all_predictors.values():
            identifiers = set()
            labels = set()
            for term in terms:
                mgcv_id = term.mgcv_identifier()
                label = term.label()
                if mgcv_id in identifiers or label in labels:
                    raise ValueError(
                        f"Duplicate term with label '{label}' and mgcv_identifier "
                        f"'{mgcv_id}' found in formula. pymgcv does not support "
                        "duplicate terms. If this is intentional, consider duplicating "
                        "the corresponding variable in your data under a new name and "
                        "using it for one of the terms.",
                    )
                identifiers.add(mgcv_id)
                labels.add(label)

        for k in self.predictors:
            if k in self.family_predictors:
                raise ValueError(
                    f"Cannot have key {k} in both predictors and family_predictors.",
                )

    def fit(
        self,
        data: pd.DataFrame,
        method: FitMethodOptions = "GCV.Cp",
    ):
        """Fit a Generalized Additive GAM.

        Note, this returns a FittedGAM object (does not mutate the model!), so
        assign the result to a variable.

        Args:
            specification: GAM object defining the model structure,
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
        self._check_valid_data(data)

        return FittedGAM(
            mgcv.gam(
                self._to_r_formulae(),
                data=data_to_rdf(data),
                family=ro.rl(self.family),
                method=method,
            ),
            data=data.copy(),
            gam=self,
        )

    @property
    def all_predictors(self) -> dict[str, list[TermLike]]:
        """All predictors (response and for family parameters)."""
        return self.predictors | self.family_predictors

    def _check_valid_data(
        self,
        data: pd.DataFrame,
    ) -> None:
        """Validate that data contains all variables required by the model specification.

        Performs comprehensive validation including:
        - Checking that all term variables exist in the data
        - Validating 'by' variables are present
        - Checking for reserved variable names that conflict with mgcv

        Args:
            data: DataFrame containing the modeling data

        Raises:
            ValueError: If required variables are missing from data
            TypeError: If categorical 'by' variables are detected (unsupported)
        """
        all_terms: list[TermLike] = []
        for terms in (self.all_predictors).values():
            all_terms.extend(terms)

        for term in all_terms:
            for varname in term.varnames:
                if varname not in data.columns:
                    raise ValueError(f"Variable {varname} not found in data.")

            if term.by is not None:
                if term.by not in data.columns:
                    raise ValueError(f"Variable {term.by} not found in data.")

            disallowed = ["Intercept", "s(", "te(", "ti(", "t2(", ":", "*"]

            for var in data.columns:
                if any(dis in var for dis in disallowed):
                    raise ValueError(
                        f"Variable name '{var}' risks clashing with terms generated by mgcv, "
                        "please rename this variable.",
                    )

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
        for target, terms in self.all_predictors.items():
            if target in self.family_predictors:
                target = ""  # no left hand side

            formula_str = f"{target}~{'+'.join(map(str, terms))}"
            if not any(isinstance(term, Intercept) for term in terms):
                formula_str += "-1"

            formulae.append(ro.Formula(formula_str))

        return formulae if len(formulae) > 1 else formulae[0]


@dataclass
class FittedGAM:
    """The fitted GAM model with methods for predicting.

    Generally returned by fitting methods (rather than being created directly).

    Args:
        rgam: The underlying R mgcv model object from fitting
        data: Original DataFrame used for model fitting
        gam: The GAM.
    """

    rgam: ro.vectors.ListVector
    data: pd.DataFrame
    gam: GAM

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

        """
        self.gam._check_valid_data(data)
        predictions = rstats.predict(
            self.rgam,
            newdata=data_to_rdf(data),
            se=True,
        )
        predictions = rlistvec_to_dict(predictions)

        all_targets = self.gam.all_predictors.keys()

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
        results = {}
        for i, (target, terms) in enumerate(self.gam.all_predictors.items()):
            result = {"fit": {}, "se": {}}
            for term in terms:
                match term:
                    case Offset() | Intercept():
                        partial_effect = self.partial_effect(target, term, data)
                        result["fit"][term.label()] = partial_effect["fit"]
                        result["se"][term.label()] = partial_effect["se"]

                    case _ if term.by is not None and data[term.by].dtype == "category":
                        levels = data[term.by].cat.categories.to_list()
                        cols = [f"{term.mgcv_identifier(i)}{lev}" for lev in levels]
                        result["fit"][term.label()] = fit[cols].sum(axis=1)
                        result["se"][term.label()] = se[cols].sum(axis=1)

                    case _:
                        result["fit"][term.label()] = fit[term.mgcv_identifier(i)]
                        result["se"][term.label()] = se[term.mgcv_identifier(i)]

            results[target] = result
            result = pd.concat({k: pd.DataFrame(v) for k, v in result.items()}, axis=1)
            results[target] = result
        return results

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
            DataFrame with columns "fit" and "se":

        Example:
            ```python
            # Get partial effect of smooth term on response
            effect = model.partial_effect('y', S('x1'), data)
            ```

        Note:
            The partial effect represents the contribution of this term alone,
            with all other terms held at their reference values (typically zero
            for centered smooth terms).
        """
        formula_idx = list(self.gam.all_predictors.keys()).index(target)
        effect, se = term._partial_effect(
            data=data,
            rgam=self.rgam,
            formula_idx=formula_idx,
        )
        assert len(data) == len(effect)
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
        """
        partial_effects = self.partial_effects(data)[target]["fit"]  # Link scale
        link_fit = partial_effects.sum(axis=1).to_numpy()
        term_effect = partial_effects.pop(term.label()).to_numpy()

        family = self.gam.family
        if "(" not in family:
            family = f"{family}()"

        rfam = ro.r(family)
        inv_link_fn = rfam.rx2("linkinv")  # TODO this breaks with GAULSS
        d_mu_d_eta_fn = rfam.rx2("mu.eta")
        rpy_link_fit = to_rpy(link_fit)
        response_residual = data[target] - to_py(inv_link_fn(rpy_link_fit))

        # We want to transform residuals to link scale.
        # link(response) - link(response_fit) not sensible (e.g. poisson with log link risks log(0))
        # Instead use first order taylor expansion of link function around the fit
        d_mu_d_eta = to_py(d_mu_d_eta_fn(rpy_link_fit))
        d_mu_d_eta = np.maximum(d_mu_d_eta, 1e-6)  # Numerical stability

        # If ĝ is the f.d. approxmator to link, below is ĝ(response) - ĝ(response_fit)
        link_residual = response_residual / d_mu_d_eta
        return link_residual + term_effect

    def summary(self) -> str:
        """Generate a comprehensive summary of the fitted GAM model.

        Produces a detailed summary including parameter estimates, significance
        tests, smooth term information, model fit statistics, and convergence
        diagnostics. The output matches the format of R's mgcv summary.
        """
        strvec = rutils.capture_output(rbase.summary(self.rgam))
        return "\n".join(tuple(strvec))

    def coefficients(self) -> pd.Series:  # TODO consider returning as dict?
        """Extract model coefficients from the fitted GAM.

        Returns a series where the index if the mgcv-style name of the parameter.
        """
        coef = self.rgam.rx2["coefficients"]
        names = coef.names
        return pd.Series(to_py(coef), index=names)

    def covariance(
        self,
        *,
        sandwich=False,
        freq=False,
        unconditional=False,
    ) -> pd.DataFrame:
        """Extract the covariance matrix from the fitted GAM.

        Extracts the Bayesian posterior covariance matrix of the parameters or
        frequentist covariance matrix of the parameter estimators from the fitted GAM.
        Returns a pandas dataframe, where the column names and index are the mgcv-style
        parameter names.

        Args:
            sandwich: If True, compute sandwich estimate of covariance matrix.
                Currently expensive for discrete bam fits.
            freq: If True, return the frequentist covariance matrix of the parameter
                estimators. If False, return the Bayesian posterior covariance matrix
                of the parameters. The latter option includes the expected squared bias
                according to the Bayesian smoothing prior.
            unconditional: If True (and freq=False), return the Bayesian smoothing
                parameter uncertainty corrected covariance matrix, if available.

        Returns:
            The covariance matrix as a numpy array.

        """
        if unconditional and freq:
            raise ValueError("Unconditional and freq cannot both be True")
        coef_names = self.rgam.rx2["coefficients"].names
        cov = to_py(
            rstats.vcov(
                self.rgam,
                sandwich=sandwich,
                freq=freq,
                unconditional=unconditional,
            ),
        )
        return pd.DataFrame(cov, index=coef_names, columns=coef_names)
