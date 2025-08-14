"""Core GAM fitting and model specification functionality."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import pandas as pd
import rpy2.rinterface as ri
import rpy2.robjects as ro
from pandas.api.types import is_numeric_dtype
from rpy2.robjects.packages import importr

from pymgcv.custom_types import PartialEffectsResult, PredictionResult
from pymgcv.families import AbstractFamily, Gaussian
from pymgcv.rpy_utils import data_to_rdf, to_py, to_rpy
from pymgcv.terms import Intercept, TermLike
from pymgcv.utils import data_len

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
    data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series]


@dataclass
class AbstractGAM(ABC):
    """Abstract base class for GAM models.

    This class cannot be initialized but provides a common interface for fitting and
    predicting GAM models.
    """

    predictors: dict[str, list[TermLike]]
    family_predictors: dict[str, list[TermLike]]
    family: AbstractFamily
    fit_state: FitState | None

    def __init__(
        self,
        predictors: Mapping[str, Iterable[TermLike] | TermLike],
        family_predictors: Mapping[str, Iterable[TermLike] | TermLike] | None = None,
        *,
        family: AbstractFamily | None = None,
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
            family: Family for the error distribution. See [Families](./families.md)
                for available options.
            add_intercepts: If True, adds an intercept term to each formula.
                If false, we assume that any [`Intercept`][pymgcv.terms.Intercept]
                terms desired are manually added to the formulae.
        """
        predictors, family_predictors = deepcopy((predictors, family_predictors))
        family_predictors = {} if family_predictors is None else family_predictors
        family = Gaussian() if family is None else family

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
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
        *args,
        **kwargs,
    ) -> Self:
        """Fit the GAM model to the given data."""
        pass

    @abstractmethod
    def predict(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *args,
        type: Literal["response", "link"] = "link",
        compute_se: bool = False,
        **kwargs,
    ) -> dict[str, PredictionResult]:
        """Predict the response variable(s) (link scale) for the given data."""
        pass

    @abstractmethod
    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
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
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
    ) -> None:
        """Validate that data contains all variables required by the model.

        Performs comprehensive validation including:
        - Checking that all term variables exist in the data
        - Validating 'by' variables are present
        - Checking for reserved variable names that conflict with mgcv

        Args:
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.

        Raises:
            ValueError: If required variables are missing from data
            TypeError: If categorical 'by' variables are detected (unsupported)
        """
        all_terms: list[TermLike] = []
        for terms in (self.all_predictors).values():
            all_terms.extend(terms)

        for term in all_terms:
            for varname in term.varnames:
                if varname not in data:
                    raise ValueError(f"Variable {varname} not found in data.")

            if term.by is not None:
                if term.by not in data:
                    raise ValueError(f"Variable {term.by} not found in data.")

            disallowed = ["Intercept", "s(", "te(", "ti(", "t2(", ":", "*"]

            for var in data:
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
        term: TermLike,
        target: str | None = None,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *,
        compute_se: bool = False,
    ) -> PredictionResult:
        """Compute the partial effect for a single model term.

        This method efficiently computes the contribution of one specific term
        to the model predictions.

        Args:
            term: The specific term to evaluate (must match a term used in the
                original model specification)
            target: Name of the target variable (response variable or family
                parameter name from the model specification). If set to None, an error
                is raised when multiple predictors are present; otherwise, the sole
                available target is used.
            data: DataFrame or dictionary containing the variables needed
                to compute the partial effect for the term.
            compute_se: Whether to compute and return standard errors
        """
        if self.fit_state is None:
            raise ValueError(
                "Cannot compute partial effect before fitting the model.",
            )

        if target is None:
            if len(self.all_predictors) > 1:
                raise ValueError(
                    "Target must be specified when multiple predictors are present.",
                )
            target = list(self.all_predictors.keys())[0]

        data = data if data is not None else self.fit_state.data

        formula_idx = list(self.all_predictors.keys()).index(target)
        return term._partial_effect(
            data=data,
            rgam=self.fit_state.rgam,
            formula_idx=formula_idx,
            compute_se=compute_se,
        )

    def edf(self) -> pd.Series:
        """Compute the effective degrees of freedom (EDF) for the model coefficients.

        Returns:
            A series of EDF values, with the mgcv-style coefficient names as the index.
        """
        if self.fit_state is None:
            raise ValueError("Model must be fit before computing the EDF.")
        edf = self.fit_state.rgam.rx2["edf"]
        return pd.Series(to_py(edf), index=to_py(edf.names))

    def penalty_edf(self):
        """Computed the effective degrees of freedom (EDF) associated with each penalty.

        Returns:
            A series of EDF values, with the index being the mgcv-style name of the
            penalty.
        """
        if self.fit_state is None:
            raise ValueError("Model must be fit before computing penalty EDFs.")
        edf = mgcv.pen_edf(self.fit_state.rgam)
        return pd.Series(to_py(edf), index=to_py(edf.names))

    def partial_residuals(
        self,
        term: TermLike,
        target: str | None = None,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *,
        avoid_scaling: bool = False,
    ) -> np.ndarray:
        """Compute partial residuals for model diagnostic plots.

        Partial residuals combine the fitted values from a specific term with
        the overall model residuals. They're useful for assessing whether the
        chosen smooth function adequately captures the relationship, or if a
        different functional form might be more appropriate.

        Args:
            term: The model term to compute partial residuals for.
            target: Name of the target variable (response variable or family
                parameter name from the model specification). If set to None, an error
                is raised when multiple predictors are present; otherwise, the sole
                available target is used.
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.
            avoid_scaling: If True, and the term has a numeric by variable,
                the scaling by the by variable is not included in the term effect.
                This facilitates plotting the residuals, as the plots
                only show the smooth component (unscaled by the by variable).

        Returns:
            Series containing the partial residuals for the specified term
        """
        if self.fit_state is None:
            raise ValueError(
                "Cannot compute partial residuals before fitting the model.",
            )

        if target is None:
            if len(self.all_predictors) > 1:
                raise ValueError(
                    "Target must be specified when multiple predictors are present.",
                )
            target = list(self.all_predictors.keys())[0]

        data = data if data is not None else self.fit_state.data
        data = deepcopy(data)
        link_fit = self.predict(data)[target].fit

        if (
            term.by is not None
            and is_numeric_dtype(data[term.by].dtype)
            and avoid_scaling
        ):
            data = dict(data) if isinstance(data, Mapping) else data
            data[term.by] = np.full(data_len(data), 1)

        term_effect = self.partial_effect(term=term, target=target, data=data).fit

        response_residual = data[target] - self.family.inverse_link(link_fit)

        # We want to transform residuals to link scale.
        # link(response) - link(response_fit) not sensible: poisson + log link -> log(0)
        # Instead use first order taylor expansion of link function around the fit
        d_mu_d_eta = self.family.dmu_deta(link_fit)
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
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
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
                    levels = data[term.by].cat.categories.to_list()  # type: ignore
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
                        term,
                        target,
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

    def aic(self, k: float = 2) -> float:
        """Calculate Akaike's Information Criterion for fitted GAM models.

        Where possible (fitting [`GAM`][pymgcv.gam.GAM]/[`BAM`][pymgcv.gam.BAM]
        models with "ML" or "REML"), this uses the approach of Wood, Pya & Saefken 2016,
        which accounts for smoothing parameter uncertainty, without favouring
        overly simple models.

        Args:
            k: Penalty per parameter (default 2 for classical AIC).
        """
        if self.fit_state is None:
            raise ValueError("Cannot compute AIC before fitting.")
        res = rstats.AIC(self.fit_state.rgam, k=k)
        return res[0]

    def residuals(
        self,
        type: Literal[
            "deviance",
            "pearson",
            "scaled.pearson",
            "working",
            "response",
        ] = "deviance",
    ):
        r"""Compute the residuals for a fitted model.

        Args:
            type: Type of residuals to compute, one of:

                - **response**: Raw residuals $y - \mu$, where $y$ is the observed data
                    and $\mu$ is the model fitted value.
                - **pearson**: Pearson residuals — raw residuals divided by the square
                    root of the model's mean-variance relationship.
                    $$
                    \frac{y - \mu}{\sqrt{V(\mu)}}
                    $$
                - **scaled.pearson**: Raw residuals divided by the standard deviation of
                    the data according to the model mean variance relationship and
                    estimated scale parameter.
                - **deviance**: Deviance residuals as defined by the model’s family.
                - **working**: Working residuals are the residuals returned from
                    model fitting at convergence.
        """
        if self.fit_state is None:
            raise ValueError("Cannot compute residuals before fitting.")
        return to_py(rstats.residuals(self.fit_state.rgam, type=type))

    def residuals_from_y_and_fit(
        self,
        *,
        y: np.ndarray,
        fit: np.ndarray,
        weights: np.ndarray | None = None,
        type: Literal[
            "deviance",
            "pearson",
            "scaled.pearson",
            "working",
            "response",
        ] = "deviance",
    ):
        """Compute the residuals, from y, fit and optionally prior weights."""
        # For now just overwrite attributes and use residuals(gam)
        # From scanning gam.residuals code this looks like it should be reasonable?
        gam = deepcopy(self)
        if gam.fit_state is None:
            raise ValueError("Cannot compute residuals before fitting the model.")

        if weights is None:
            weights = np.ones_like(y)
        gam.fit_state.rgam.rx2["y"] = to_rpy(y)
        gam.fit_state.rgam.rx2["fitted.values"] = to_rpy(fit)
        gam.fit_state.rgam.rx2["prior.weights"] = to_rpy(weights)
        return gam.residuals(type)


@dataclass(init=False)
class GAM(AbstractGAM):
    """Standard GAM Model."""

    predictors: dict[str, list[TermLike]]
    family_predictors: dict[str, list[TermLike]]
    family: AbstractFamily
    fit_state: FitState | None

    def fit(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
        *,
        method: GAMFitMethods = "REML",
        weights: str | np.ndarray | pd.Series | None = None,
        optimizer: str | tuple[str, str] = ("outer", "newton"),
        scale: Literal["unknown"] | float | int | None = None,
        select: bool = False,
        gamma: float | int = 1,
        knots: dict[str, np.ndarray] | None = None,
        n_threads: int = 1,
    ) -> Self:
        # TODO if we do this, we need to support it everywhere we pass data?
        """Fit the GAM.

        Args:
            data: DataFrame or dictionary containing all variables referenced in the
                model. Note, using a dictionary is required when passing matrix-valued
                variables.
            method: Method for smoothing parameter estimation, matching the mgcv
                options.
            weights: Observation weights. Either a string, matching a column name,
                or an array/series with length equal to the number of observations.
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
            knots: Dictionary mapping covariate names to knot locations.
                For most bases, the length of the knot locations should match with a
                user supplied `k` value. E.g. for `S("x", k=64)`, you could
                pass `knots={"x": np.linspace(0, 1, 64)}`. For multidimensional
                smooths, e.g. `S("x", "z", k=64)`, you could create a grid of
                coordinates:

                !!! example

                    ```python
                        import numpy as np
                        coords = np.linspace(0, 1, num=8)
                        X, Z = np.meshgrid(coords, coords)
                        knots = {"x": X.ravel(), "z": Z.ravel()}
                    ```
                Note if using
                [`ThinPlateSpline`][pymgcv.basis_functions.ThinPlateSpline], this will
                avoid the eigen-decomposition used to find the basis, which although
                fast often leads to worse results. Different terms can use different
                numbers of knots, unless they share covariates.
            n_threads: Number of threads to use for fitting the GAM.
        """
        # TODO some missing options: control, sp, min.sp etc
        data = deepcopy(data)
        weights = data[weights] if isinstance(weights, str) else weights  # type: ignore
        r_knots = (
            ro.NULL
            if knots is None
            else ro.ListVector({k: to_rpy(pos) for k, pos in knots.items()})
        )
        self._check_valid_data(data)
        rgam = mgcv.gam(
            self._to_r_formulae(),
            data=data_to_rdf(data),
            family=self.family.rfamily,
            method=method,
            weights=ro.NULL if weights is None else np.asarray(weights),
            optimizer=to_rpy(np.array(optimizer)),
            scale=0 if scale is None else (-1 if scale == "unknown" else scale),
            select=select,
            gamma=gamma,
            knots=r_knots,
            nthreads=n_threads,
        )
        self.fit_state = FitState(
            rgam=rgam,
            data=data,
        )
        return self

    def predict(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *,
        type: Literal["response", "link"] = "link",
        compute_se: bool = False,
        block_size: int | None = None,
    ) -> dict[str, PredictionResult]:
        """Compute model predictions with (optionally) uncertainty estimates.

        Makes predictions for new data using the fitted GAM model. Predictions
        are returned on the link scale (linear predictor scale), not the response
        scale. For response scale predictions, apply the appropriate inverse link
        function to the results.

        Args:
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.
            compute_se: Whether to compute standard errors for predictions.
            type: Type of prediction to compute. Either "link" for linear predictor
                scale or "response" for response scale.
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
            type=type,
            block_size=ri.MissingArg if block_size is None else block_size,
        )
        return self._format_predictions(
            predictions,
            compute_se=compute_se,
        )

    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *,
        compute_se: bool = False,
        block_size: int | None = None,
    ) -> dict[str, PartialEffectsResult]:
        """Compute partial effects for all model terms.

        Calculates the contribution of each model term to the overall prediction on the
        link scale. The sum of all fit columns equals the total prediction (link scale).

        Args:
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.
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
    family: AbstractFamily
    fit_state: FitState | None

    def fit(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
        *,
        method: BAMFitMethods = "fREML",
        weights: str | np.ndarray | pd.Series | None = None,
        scale: Literal["unknown"] | float | int | None = None,
        select: bool = False,
        gamma: float | int = 1,
        knots: dict[str, np.ndarray] | None = None,
        chunk_size: int = 10000,
        discrete: bool = False,
        samfrac: float | int = 1,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> Self:
        """Fit the GAM.

        Args:
            data: DataFrame or dictionary containing all variables referenced in the
                model. Note, using a dictionary is required when passing matrix-valued
                variables.
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
            knots: Dictionary mapping covariate names to knot locations.
                For most bases, the length of the knot locations should match with a
                user supplied `k` value. E.g. for `S("x", k=64)`, you could
                pass `knots={"x": np.linspace(0, 1, 64)}`. For multidimensional
                smooths, e.g. `S("x", "z", k=64)`, you could create a grid of
                coordinates:

                !!! example

                    ```python
                        import numpy as np
                        coords = np.linspace(0, 1, num=8)
                        X, Z = np.meshgrid(coords, coords)
                        knots = {"x": X.ravel(), "z": Z.ravel()}
                    ```
                Note if using
                [`ThinPlateSpline`][pymgcv.basis_functions.ThinPlateSpline], this will
                avoid the eigen-decomposition used to find the basis, which although
                fast often leads to worse results. Different terms can use different
                numbers of knots, unless they share covariates.
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
        data = deepcopy(data)
        weights = data[weights] if isinstance(weights, str) else weights  # type: ignore
        r_knots = (
            ro.NULL
            if knots is None
            else ro.ListVector({k: to_rpy(pos) for k, pos in knots.items()})
        )
        self._check_valid_data(data)
        self.fit_state = FitState(
            rgam=mgcv.bam(
                self._to_r_formulae(),
                data=data_to_rdf(data),
                family=self.family.rfamily,
                method=method,
                weights=ro.NULL if weights is None else np.asarray(weights),
                scale=0 if scale is None else (-1 if scale == "unknown" else scale),
                select=select,
                gamma=gamma,
                knots=r_knots,
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
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *,
        compute_se: bool = False,
        type: Literal["link", "response"] = "link",
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
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.
            compute_se: Whether to compute and return standard errors.
            type: Type of prediction to compute. Either "link" for linear predictor
                scale or "response" for response scale.
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
            type=type,
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
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
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
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.
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
