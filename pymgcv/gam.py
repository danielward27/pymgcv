"""Core GAM fitting and model specification functionality."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import rpy2.rinterface as ri
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import data_to_rdf, to_py, to_rpy
from pymgcv.custom_types import PartialEffectsResult, PredictionResult
from pymgcv.terms import Intercept, TermLike

mgcv = importr("mgcv")
rbase = importr("base")
rutils = importr("utils")
rstats = importr("stats")

GAMFitMethods = Literal[
    "GCV.Cp",
    "GACV.Cp",
    "QNCV",
    "REML",
    "P-REML",
    "ML",
    "P-ML",
    "NCV",
]

BAMFitMethods = Literal[
    "fREML",
    "GCV.Cp",
    "GACV.Cp",
    "REML",
    "P-REML",
    "ML",
    "P-ML",
    "NCV",
]


@dataclass
class FitState:
    """The mgcv gam, and the data used for fitting.

    This gets set as an attribute fit_state on the AbstractGAM object after fitting.

    Attributes:
        rgam: The fitted mgcv gam object.
        data: The data used for fitting.
    """

    rgam: ro.ListVector
    data: pd.DataFrame


@dataclass
class AbstractGAM(ABC):
    """Abstract base class for GAM models.

    This class cannot be initialized but provides a common interface for fitting and
    predicting GAM models.
    """

    predictors: dict[str, list[TermLike]]
    family_predictors: dict[str, list[TermLike]]
    family: str
    fit_state: FitState | None

    def __init__(
        self,
        predictors: dict[str, Iterable[TermLike] | TermLike],
        family_predictors: dict[str, Iterable[TermLike] | TermLike] | None = None,
        *,
        family: str = "gaussian",
        add_intercepts: bool = True,
    ):
        r"""Initialize a GAM/BAM model.

        Args:
            predictors: Dictionary mapping response variable names to an iterable of
                [`TermLike`][pymgcv.terms.TermLike] objects used to predict
                $g([\mathbb{E}[Y])$. For single response models, use a single key-value
                pair. For multivariate models, include multiple response variables.
            family_predictors: Dictionary mapping family parameter names to an iterable
                of terms for modeling those parameters. Keys are used as labels during
                prediction and should match the order expected by the mgcv family.
            family: String specifying the mgcv family for the error distribution.
                This is passed directly to R's mgcv and can include family arguments.
            add_intercepts: If True, adds an intercept term to each formula.
                If false, we assume that any [`Intercept`][pymgcv.terms.Intercept]
                terms desired are manually added to the formulae.
        """
        predictors, family_predictors = deepcopy((predictors, family_predictors))
        family_predictors = {} if family_predictors is None else family_predictors

        def _ensure_list_of_terms(d):
            return {
                k: [v] if isinstance(v, TermLike) else list(v) for k, v in d.items()
            }

        self.predictors = _ensure_list_of_terms(predictors)
        self.family_predictors = _ensure_list_of_terms(family_predictors)
        self.family = family
        self.fit_state = None

        if add_intercepts:
            for v in self.all_predictors.values():
                v.append(Intercept())
        self._check_init()

    def _check_init(self):
        # Perform some basic checks
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

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        *args,
        **kwargs,
    ) -> Self:
        """Fit the GAM model to the given data."""
        pass

    @abstractmethod
    def predict(
        self,
        data: pd.DataFrame | None = None,
        *args,
        compute_se: bool = False,
        **kwargs,
    ) -> dict[str, PredictionResult]:
        """Predict the response variable(s) (link scale) for the given data."""
        pass

    @abstractmethod
    def partial_effects(
        self,
        data: pd.DataFrame | None = None,
        *args,
        compute_se: bool = False,
        **kwargs,
    ) -> dict[str, PartialEffectsResult]:
        """Compute the partial effects for the terms in the model."""
        pass

    @property
    def all_predictors(self) -> dict[str, list[TermLike]]:
        """All predictors (response and for family parameters)."""
        return self.predictors | self.family_predictors

    def _check_valid_data(
        self,
        data: pd.DataFrame,
    ) -> None:
        """Validate that data contains all variables required by the model.

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
                        f"Variable name '{var}' risks clashing with terms generated by "
                        "mgcv, please rename this variable.",
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

    def summary(self) -> str:
        """Generate an mgcv-style summary of the fitted GAM model."""
        if self.fit_state is None:
            raise RuntimeError("Cannot print summary of an unfitted model.")
        strvec = rutils.capture_output(rbase.summary(self.fit_state.rgam))
        return "\n".join(tuple(strvec))

    def coefficients(self) -> pd.Series:  # TODO consider returning as dict?
        """Extract model coefficients from the fitted GAM.

        Returns a series where the index if the mgcv-style name of the parameter.
        """
        if self.fit_state is None:
            raise RuntimeError("Cannot extract coefficients from an unfitted model.")
        coef = self.fit_state.rgam.rx2["coefficients"]
        names = coef.names
        return pd.Series(to_py(coef), index=names)

    def covariance(
        self,
        *,
        sandwich: bool = False,
        freq: bool = False,
        unconditional: bool = False,
    ) -> pd.DataFrame:
        """Extract the covariance matrix from the fitted GAM.

        Extracts the Bayesian posterior covariance matrix of the parameters or
        frequentist covariance matrix of the parameter estimators from the fitted GAM.

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
            The covariance matrix as a pandas dataframe where the column names and index
            are the mgcv-style parameter names.

        """
        if self.fit_state is None:
            raise RuntimeError("Cannot extract covariance from an unfitted model.")

        if unconditional and freq:
            raise ValueError("Unconditional and freq cannot both be True")

        coef_names = self.fit_state.rgam.rx2["coefficients"].names
        cov = to_py(
            rstats.vcov(
                self.fit_state.rgam,
                sandwich=sandwich,
                freq=freq,
                unconditional=unconditional,
            ),
        )
        return pd.DataFrame(cov, index=coef_names, columns=coef_names)

    def partial_effect(
        self,
        target: str,
        term: TermLike,
        data: pd.DataFrame,
        *,
        compute_se: bool = False,
    ) -> PredictionResult:
        """Compute the partial effect for a single model term.

        This method efficiently computes the contribution of one specific term
        to the model predictions.

        Args:
            target: Name of the target variable (response variable or family
                parameter name from the model specification)
            term: The specific term to evaluate (must match a term used in the
                original model specification)
            data: DataFrame containing the predictor variables needed for the term
            compute_se: Whether to compute and return standard errors
        """
        data = data if data is not None else self.fit_state.data
        if self.fit_state is None:
            raise RuntimeError(
                "Cannot compute partial effect before fitting the model.",
            )

        formula_idx = list(self.all_predictors.keys()).index(target)
        return term._partial_effect(
            data=data,
            rgam=self.fit_state.rgam,
            formula_idx=formula_idx,
            compute_se=compute_se,
        )

    def partial_residuals(
        self,
        target: str,
        term: TermLike,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.Series:
        """Compute partial residuals for model diagnostic plots.

        Partial residuals combine the fitted values from a specific term with
        the overall model residuals. They're useful for assessing whether the
        chosen smooth function adequately captures the relationship, or if a
        different functional form might be more appropriate.

        Args:
            target: Name of the response variable.
            term: The model term to compute partial residuals for.
            data: DataFrame containing the data (must include the response variable).
            **kwargs: Additional keyword arguments to pass to `partial_effects`.

        Returns:
            Series containing the partial residuals for the specified term
        """
        data = data if data is not None else self.fit_state.data
        partial_effects = self.partial_effects(data, **kwargs)[target].fit  # Link scale
        assert isinstance(partial_effects, pd.DataFrame)
        link_fit = partial_effects.sum(axis=1).to_numpy()
        term_effect = partial_effects.pop(term.label()).to_numpy()

        family = self.family
        if "(" not in family:
            family = f"{family}()"

        rfam: ro.ListVector = ro.r(family)  # type: ignore
        inv_link_fn = rfam.rx2("linkinv")  # TODO this breaks with GAULSS
        d_mu_d_eta_fn = rfam.rx2("mu.eta")

        if rbase.is_null(inv_link_fn)[0] or rbase.is_null(d_mu_d_eta_fn)[0]:
            raise NotImplementedError(
                f"Computing partial residuals for {family} not yet supported.",
            )
        rpy_link_fit = to_rpy(link_fit)
        response_residual = data[target] - to_py(inv_link_fn(rpy_link_fit))

        # We want to transform residuals to link scale.
        # link(response) - link(response_fit) not sensible: poisson + log link -> log(0)
        # Instead use first order taylor expansion of link function around the fit
        d_mu_d_eta = to_py(d_mu_d_eta_fn(rpy_link_fit))
        d_mu_d_eta = np.maximum(d_mu_d_eta, 1e-6)  # Numerical stability

        # If ĝ is the first order approxmation to link, below is:
        # ĝ(response) - ĝ(response_fit)
        link_residual = response_residual / d_mu_d_eta
        return link_residual + term_effect

    def _format_predictions(self, predictions, *, compute_se: bool):
        """Formats output from mgcv predict."""
        all_targets = self.all_predictors.keys()
        result = {}

        if compute_se:
            fit_all = to_py(predictions.rx2["fit"]).reshape(-1, len(all_targets))
            se_all = to_py(predictions.rx2["se.fit"]).reshape(-1, len(all_targets))
        else:
            fit_all = to_py(predictions).reshape(-1, len(all_targets))
            se_all = None

        for i, target in enumerate(all_targets):
            if compute_se:
                assert se_all is not None
                result[target] = PredictionResult(
                    fit=fit_all[:, i],
                    se=se_all[:, i],
                )
            else:
                result[target] = PredictionResult(fit=fit_all[:, i])
        return result

    def _format_partial_effects(
        self,
        predictions,
        data: pd.DataFrame,
        *,
        compute_se: bool,
    ) -> dict[str, PartialEffectsResult]:
        """Formats output from mgcv predict with type="terms" (i.e. partial effects)."""
        if compute_se:
            fit_raw = pd.DataFrame(
                to_py(predictions.rx2["fit"]),
                columns=to_py(rbase.colnames(predictions.rx2["fit"])),
            )
            se_raw = pd.DataFrame(
                to_py(predictions.rx2["se.fit"]),
                columns=to_py(rbase.colnames(predictions.rx2["se.fit"])),
            )
        else:
            fit_raw = pd.DataFrame(
                to_py(predictions),
                columns=to_py(rbase.colnames(predictions)),
            )
            se_raw = None

        # Partition results based on formulas
        results = {}
        for i, (target, terms) in enumerate(self.all_predictors.items()):
            fit = {}
            se = {}

            for term in terms:
                label = term.label()
                identifier = term.mgcv_identifier(i)

                if term.by is not None and data[term.by].dtype == "category":
                    levels = data[term.by].cat.categories.to_list()
                    cols = [f"{identifier}{lev}" for lev in levels]
                    fit[label] = fit_raw[cols].sum(axis=1)

                    if se_raw is not None:
                        se[label] = se_raw[cols].sum(axis=1)

                elif identifier in fit_raw.columns:
                    fit[label] = fit_raw[identifier]

                    if se_raw is not None:
                        se[label] = se_raw[identifier]

                else:  # Offset + Intercept
                    partial_effect = self.partial_effect(
                        target,
                        term,
                        data,
                        compute_se=compute_se,
                    )
                    fit[label] = partial_effect.fit

                    if compute_se:
                        assert se is not None
                        se[label] = partial_effect.se

            results[target] = PartialEffectsResult(
                fit=pd.DataFrame(fit),
                se=None if not compute_se else pd.DataFrame(se),
            )
        return results


@dataclass(init=False)
class GAM(AbstractGAM):
    """Standard GAM Model."""

    predictors: dict[str, list[TermLike]]
    family_predictors: dict[str, list[TermLike]]
    family: str
    fit_state: FitState | None

    def fit(
        self,
        data: pd.DataFrame,
        *,
        method: GAMFitMethods = "GCV.Cp",
        weights: str | np.ndarray | pd.Series | None = None,
        optimizer: str | tuple[str, str] = ("outer", "newton"),
        scale: Literal["unknown"] | float | int | None = None,
        select: bool = False,
        gamma: float | int = 1,
        n_threads: int = 1,
    ) -> Self:
        """Fit the GAM.

        Args:
            data: DataFrame containing all variables referenced in the specification.
                Variable names must match those used in the model terms.
            method: Method for smoothing parameter estimation, matching the mgcv,
                options.
            weights: Observation weights. Either a string, matching a column name,
                or a array/series with length equal to the number of observations.
            optimizer: An string or length 2 tuple, specifying the numerical
                optimization method to use to optimize the smoothing parameter
                estimation criterion (given by method). "outer" for the direct nested
                optimization approach. "outer" can use several alternative optimizers,
                specified in the second element: "newton" (default), "bfgs", "optim" or
                "nlm". "efs" for the extended Fellner Schall method of Wood and Fasiolo
                (2017).
            scale: If a number is provided, it is treated as a known scale parameter.
                If left to None, the scale parameter is 1 for Poisson and binomial and
                unknown otherwise. Note that (RE)ML methods can only work with scale
                parameter 1 for the Poisson and binomial cases.
            select: If set to True then gam can add an extra penalty to each term so
                that it can be penalized to zero. This means that the smoothing
                parameter estimation during fitting can completely remove terms
                from the model. If the corresponding smoothing parameter is estimated as
                zero then the extra penalty has no effect. Use gamma to increase level
                of penalization.
            gamma: Increase this beyond 1 to produce smoother models. gamma multiplies
                the effective degrees of freedom in the GCV or UBRE/AIC. gamma can be
                viewed as an effective sample size in the GCV score, and this also
                enables it to be used with REML/ML. Ignored with P-RE/ML or the efs
                optimizer.
            n_threads: Number of threads to use for fitting the GAM.
        """
        # TODO some missing options: control, sp, knots, min.sp etc
        data = data.copy()
        weights = data[weights] if isinstance(weights, str) else weights  # type: ignore

        self._check_valid_data(data)
        rgam = mgcv.gam(
            self._to_r_formulae(),
            data=data_to_rdf(data),
            family=ro.rl(self.family),
            method=method,
            weights=ro.NULL if weights is None else np.asarray(weights),
            optimizer=to_rpy(np.array(optimizer)),
            scale=0 if scale is None else (-1 if scale == "unknown" else scale),
            select=select,
            gamma=gamma,
            nthreads=n_threads,
        )
        self.fit_state = FitState(
            rgam=rgam,
            data=data,
        )
        return self

    def predict(
        self,
        data: pd.DataFrame | None = None,
        *,
        compute_se: bool = False,
        block_size: int | None = None,
    ) -> dict[str, PredictionResult]:
        """Compute model predictions with uncertainty estimates.

        Makes predictions for new data using the fitted GAM model. Predictions
        are returned on the link scale (linear predictor scale), not the response
        scale. For response scale predictions, apply the appropriate inverse link
        function to the results.

        Args:
            data: DataFrame containing predictor variables. Must include all
                variables referenced in the original model specification.
            compute_se: Whether to compute standard errors for predictions.
            block_size: Number of rows to process at a time.  If None then block size
                is 1000 if data supplied, and the number of rows in the model frame
                otherwise.

        Returns:
            A dictionary mapping the target variable names to a pandas DataFrame
            containing the predictions and standard errors if `se` is True.
        """
        if data is not None:
            self._check_valid_data(data)

        if self.fit_state is None:
            raise RuntimeError("Cannot call predict before fitting.")

        predictions = rstats.predict(
            self.fit_state.rgam,
            ri.MissingArg if data is None else data_to_rdf(data),
            se=compute_se,
            block_size=ri.MissingArg if block_size is None else block_size,
        )
        return self._format_predictions(
            predictions,
            compute_se=compute_se,
        )

    def partial_effects(
        self,
        data: pd.DataFrame | None = None,
        *,
        compute_se: bool = False,
        block_size: int | None = None,
    ) -> dict[str, PartialEffectsResult]:
        """Compute partial effects for all model terms.

        Calculates the contribution of each model term to the overall prediction on the
        link scale. The sum of all fit columns equals the total prediction (link scale).

        Args:
            data: DataFrame containing predictor variables for evaluation. Defaults to
                using the data for fitting.
            compute_se: Whether to compute and return standard errors.
            block_size: Number of rows to process at a time.  If None then block size
                is 1000 if data supplied, and the number of rows in the model frame
                otherwise.
        """
        if data is not None:
            self._check_valid_data(data)

        if self.fit_state is None:
            raise RuntimeError("Cannot call partial_effects before fitting.")

        predictions = rstats.predict(
            self.fit_state.rgam,
            ri.MissingArg if data is None else data_to_rdf(data),
            se=compute_se,
            type="terms",
            newdata_gauranteed=True,
            block_size=ri.MissingArg if block_size is None else block_size,
        )
        return self._format_partial_effects(
            predictions,
            data if data is not None else self.fit_state.data,
            compute_se=compute_se,
        )


@dataclass(init=False)  # use AbstractGAM init
class BAM(AbstractGAM):
    """A big-data GAM (BAM) model."""

    predictors: dict[str, list[TermLike]]
    family_predictors: dict[str, list[TermLike]]
    family: str
    fit_state: FitState | None

    def fit(
        self,
        data: pd.DataFrame,
        *,
        method: BAMFitMethods = "fREML",
        weights: str | np.ndarray | pd.Series | None = None,
        scale: Literal["unknown"] | float | int | None = None,
        select: bool = False,
        gamma: float | int = 1,
        chunk_size: int = 10000,
        discrete: bool = False,
        samfrac: float | int = 1,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> Self:
        """Fit the GAM.

        Args:
            data: DataFrame containing all variables referenced in the specification.
                Variable names must match those used in the model terms.
            method: Method for smoothing parameter estimation, matching the mgcv,
                options.
            weights: Observation weights. Either a string, matching a column name,
                or a array/series with length equal to the number of observations.
            scale: If a number is provided, it is treated as a known scale parameter.
                If left to None, the scale parameter is 1 for Poisson and binomial and
                unknown otherwise. Note that (RE)ML methods can only work with scale
                parameter 1 for the Poisson and binomial cases.
            select: If set to True then gam can add an extra penalty to each term so
                that it can be penalized to zero. This means that the smoothing
                parameter estimation during fitting can completely remove terms
                from the model. If the corresponding smoothing parameter is estimated as
                zero then the extra penalty has no effect. Use gamma to increase level
                of penalization.
            gamma: Increase this beyond 1 to produce smoother models. gamma multiplies
                the effective degrees of freedom in the GCV or UBRE/AIC. gamma can be
                viewed as an effective sample size in the GCV score, and this also
                enables it to be used with REML/ML. Ignored with P-RE/ML or the efs
                optimizer.
            chunk_size: The model matrix is created in chunks of this size, rather than
                ever being formed whole. Reset to 4*p if chunk.size < 4*p where p is the
                number of coefficients.
            discrete: if True and using method="fREML", discretizes covariates for
                storage and efficiency reasons.
            samfrac: If ``0<samfrac<1``, performs a fast preliminary fitting step using
                a subsample of the data to improve convergence speed.
            gc_level: 0 uses R's garbage collector, 1 and 2 use progressively
                more frequent garbage collection, which takes time but reduces
                memory requirements.
        """
        # TODO some missing options: control, sp, knots, min.sp, nthreads
        data = data.copy()
        weights = data[weights] if isinstance(weights, str) else weights  # type: ignore

        self._check_valid_data(data)
        self.fit_state = FitState(
            rgam=mgcv.bam(
                self._to_r_formulae(),
                data=data_to_rdf(data),
                family=ro.rl(self.family),
                method=method,
                weights=ro.NULL if weights is None else np.asarray(weights),
                scale=0 if scale is None else (-1 if scale == "unknown" else scale),
                select=select,
                gamma=gamma,
                chunk_size=chunk_size,
                discrete=discrete,
                samfrac=samfrac,
                gc_level=gc_level,
            ),
            data=data,
        )
        return self

    def predict(
        self,
        data: pd.DataFrame | None = None,
        *,
        compute_se: bool = False,
        block_size: int = 50000,
        discrete: bool = True,
        n_threads: int = 1,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> dict[str, PredictionResult]:
        """Compute model predictions with uncertainty estimates.

        Makes predictions for new data using the fitted GAM model. Predictions
        are returned on the link scale (linear predictor scale), not the response
        scale. For response scale predictions, apply the appropriate inverse link
        function to the results.

        Args:
            data: DataFrame containing predictor variables. Must include all
                variables referenced in the original model specification.
            compute_se: Whether to compute and return standard errors.
            block_size: Number of rows to process at a time.
            n_threads: Number of threads to use for computation.
            discrete: If True and the model was fitted with discrete=True, then
                uses discrete prediction methods in which covariates are
                discretized for efficiency for storage and efficiency reasons.
            gc_level: 0 uses R's garbage collector, 1 and 2 use progressively
                more frequent garbage collection, which takes time but reduces
                memory requirements.
        """
        if data is not None:
            self._check_valid_data(data)

        if self.fit_state is None:
            raise RuntimeError("Cannot call predict before fitting.")

        predictions = rstats.predict(
            self.fit_state.rgam,
            ri.MissingArg if data is None else data_to_rdf(data),
            se=compute_se,
            block_size=ro.NULL if block_size is None else block_size,
            discrete=discrete,
            n_threads=n_threads,
            gc_level=gc_level,
        )

        return self._format_predictions(
            predictions,
            compute_se=compute_se,
        )

    def partial_effects(
        self,
        data: pd.DataFrame | None = None,
        *,
        compute_se: bool = False,
        block_size: int = 50000,
        n_threads: int = 1,
        discrete: bool = True,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> dict[str, PartialEffectsResult]:
        """Compute partial effects for all model terms.

        Calculates the contribution of each model term to the overall prediction.
        This decomposition is useful for understanding which terms contribute most
        to predictions and for creating partial effect plots. The sum of all fit columns
        equals the total prediction.

        Args:
            data: DataFrame containing predictor variables for evaluation.
            compute_se: Whether to compute and return standard errors.
            block_size: Number of rows to process at a time. Higher is faster
                but more memory intensive.
            n_threads: Number of threads to use for computation.
            discrete: If True and the model was fitted with discrete=True, then
                uses discrete prediction methods in which covariates are
                discretized for efficiency for storage and efficiency reasons.
            gc_level: 0 uses R's garbage collector, 1 and 2 use progressively
                more frequent garbage collection, which takes time but reduces
                memory requirements.
        """
        if data is not None:
            self._check_valid_data(data)

        if self.fit_state is None:
            raise RuntimeError("Cannot call partial_effects before fitting.")

        predictions = rstats.predict(
            self.fit_state.rgam,
            ri.MissingArg if data is None else data_to_rdf(data),
            se=compute_se,
            type="terms",
            newdata_gauranteed=True,
            block_size=ri.MissingArg if block_size is None else block_size,
            n_threads=n_threads,
            discrete=discrete,
            gc_level=gc_level,
        )
        return self._format_partial_effects(
            predictions,
            data=self.fit_state.data if data is None else data,
            compute_se=compute_se,
        )
