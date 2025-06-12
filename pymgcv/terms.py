"""The available terms for constructing GAM models.

This module provides various term types that can be used to construct GAM models:
- Linear terms for parametric effects
- Smooth terms for non-linear effects with various basis functions
- Tensor smooth terms for scale-invariant multi-dimensional smoothing
- Interaction terms for parametric interactions
- Offset terms for known relationships

All terms implement the TermLike protocol and can be combined flexibly in
ModelSpecification objects to build complex GAM models.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.bases import BasisLike
from pymgcv.converters import data_to_rdf, to_py

if TYPE_CHECKING:
    pass

# TODO from pymgcv.gam import FittedGAM causes circular import

mgcv = importr("mgcv")
rbase = importr("base")
rstats = importr("stats")
# TODO: Not supporting 'sp' or 'pc' basis types.
# TODO: 'xt' is not needed, as basis-related configuration will handle it.

# For now, we only support terms that do not implicitly expand into multiple terms in mgcv.
# This simplifies the interface:
# - Each term in the equation maps directly to one term in the mgcv model.
# - Plotting becomes straightforward, as we can refer to the original formula terms
#   without tracking internal expansions.
#
# We will support these expanding terms in some form, either through utilities to
#     simplify the definition, or by supporting them directly. But for now:
# - Only numeric 'by' variables are supported.
# - Factor 'by' variables and factor interactions must be manually specified.

# We can provide utility functions to help define these more complex terms when needed.


@runtime_checkable
class TermLike(Protocol):
    """Protocol defining the interface for GAM model terms.

    All term types in pymgcv must implement this protocol. It defines the basic
    interface for model terms including variable references, string representations,
    and the ability to compute partial effects.

    Attributes:
        varnames: Tuple of variable names used by this term. For univariate terms,
            this contains a single variable name. For multivariate terms (like
            tensor smooths), this contains multiple variable names.
        by: Optional name of a 'by' variable that scales this term. When present,
            the term's effect is multiplied by the values of this variable.

    Methods that must be implemented:
        __str__: Convert term to mgcv formula syntax
        simple_string: Generate simplified term identifier
        _partial_effect: Compute partial effects for the term
    """

    varnames: tuple[str, ...]
    by: str | None

    def __str__(self) -> str:
        """Convert the term to mgcv formula syntax.

        Returns:
            String representation that can be used in R mgcv formula,
            e.g., "s(x1,x2,k=10)" or "x1:x2"
        """
        ...

    def simple_string(self, formula_idx: int = 0) -> str:
        """Generate a simplified identifier for the term.

        This representation is used internally for term identification and
        matches the format used by mgcv when predicting separate terms.
        For multi-formula models, mgcv adds an index suffix to distinguish
        terms from different formulae.

        Args:
            formula_idx: Index of the formula in multi-formula models (0-based)

        Returns:
            Simplified string identifier, e.g., "s(x1,x2)" or "s.1(x1,x2)"
        """
        ...

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam: Any,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute partial effects and standard errors for this term.

        This method is used internally to compute the contribution of this
        specific term to the overall model prediction.

        Args:
            data: DataFrame containing predictor variables
            gam: Fitted GAM model object
            target: Name of the target variable (response or family parameter)

        Returns:
            Tuple of (effects, standard_errors) as numpy arrays
        """


@dataclass
class Linear(TermLike):
    """Linear (parametric) term with no basis expansion.

    If the variable is a categorical variable, the term will be expanded (one-hot
    encoded into a set of dummy variables. Otherwise, the variable is included
    as is and the effect computed by a single coefficient multiplied by the variable
    value.

    Args:
        name: Name of the variable to include as a linear term. Must be present
            in the data used for model fitting.
    """

    varnames: tuple[str]
    by: str | None

    def __init__(self, name: str):
        """Initialize a linear term.

        Args:
            name: Variable name for the linear effect
        """
        self.varnames = (name,)
        self.by = None

    def __str__(self) -> str:
        """Return variable name for mgcv formula."""
        return self.varnames[0]

    def simple_string(self, formula_idx: int = 0) -> str:
        """Return term identifier with optional formula index."""
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        return self.varnames[0] + idx

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam: Any,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute linear term partial effects.

        For linear terms, the partial effect is simply the coefficient
        multiplied by the variable values. Standard errors are computed
        using the coefficient variance.
        """
        formula_idx = list(gam.model_specification.all_formulae.keys()).index(target)
        mgcv_label = self.simple_string(formula_idx)
        coef = rstats.coef(gam.rgam)
        slope = to_py(coef.rx2[mgcv_label]).item()
        data_array = data[self.varnames[0]].to_numpy()
        param_idx = rbase.which(coef.names.ro == mgcv_label)
        assert len(param_idx) == 1
        variance = to_py(
            gam.rgam.rx2["Vp"].rx(param_idx, param_idx),
        ).item()  # TODO not sure this is actually correct?
        se = np.abs(data_array) * np.sqrt(variance)
        return data_array * slope, se


# TODO can above logic be comined with interaction?


@dataclass
class Interaction(TermLike):
    """Parametric interaction term between multiple variables.

    Represents multiplicative interactions between variables. The interaction
    is computed as the product of the variable values, creating a single
    parametric term with one coefficient.

    Important: This creates only the specified interaction term. It does NOT
    automatically include main effects or lower-order interactions. These must
    be added explicitly if desired.

    Use interaction terms when:
    - You have theoretical reasons to expect multiplicative effects
    - You want to model how the effect of one variable changes with another
    - You need interpretable parametric interactions
    - You have categorical variables that interact

    Args:
        *varnames: Variable names to include in the interaction. Can be any
            number of variables (2 or more recommended).

    Examples:
        ```python
        # Two-way interaction
        age_income = Interaction('age', 'income')

        # Three-way interaction
        complex_int = Interaction('x1', 'x2', 'x3')

        # Generate all pairwise interactions
        from itertools import combinations
        varnames = ['x1', 'x2', 'x3', 'x4']
        pairwise = [Interaction(*pair) for pair in combinations(varnames, 2)]

        # Complete model with main effects and interactions
        spec = ModelSpecification(
            response_predictors={'y': [
                Linear('x1'), Linear('x2'),  # Main effects
                Interaction('x1', 'x2')      # Interaction
            ]}
        )
        ```

    Note:
        For smooth interactions between continuous variables, consider using
        TensorSmooth instead, which can capture non-linear interaction surfaces.
    """

    varnames: tuple[str, ...]
    by: str | None

    def __init__(self, *varnames: str):
        """Initialize an interaction term.

        Args:
            *varnames: Names of variables to include in the interaction.
                Must be 2 or more variables.
        """
        if len(varnames) < 2:
            raise ValueError("Interaction terms require at least 2 variables")
        self.varnames = tuple(varnames)
        self.by = None

    def __str__(self) -> str:
        """Return interaction in mgcv formula syntax (colon-separated)."""
        return ":".join(self.varnames)

    def simple_string(self, formula_idx: int = 0) -> str:
        """Return interaction identifier with optional formula index."""
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        return ":".join(self.varnames) + idx

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam: Any,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute interaction term partial effects.

        Creates the design matrix for the interaction and computes effects
        using the fitted coefficients and covariance matrix.
        """
        data = data[list(self.varnames)]
        formula_idx = list(gam.model_specification.all_formulae.keys()).index(target)
        predict_mat = rstats.model_matrix(
            ro.Formula(f"~{str(self)}-1"),
            data=data_to_rdf(data),
        )
        coef_names = rbase.colnames(predict_mat)
        post_fix = "" if formula_idx == 0 else f".{formula_idx}"
        all_coefs = rstats.coef(gam.rgam)
        coef_names = rbase.paste0(rbase.colnames(predict_mat), post_fix)
        coefs = all_coefs.rx(coef_names)
        fit = predict_mat @ coefs
        cov = gam.rgam.rx2["Vp"]
        cov.rownames = all_coefs.names
        cov.colnames = all_coefs.names
        subcov = cov.rx(coef_names, coef_names)
        se = rbase.sqrt(rbase.rowSums((predict_mat @ subcov).ro * predict_mat))
        return to_py(fit).squeeze(axis=-1), to_py(se)


@dataclass
class Smooth(TermLike):
    """Smooth term using spline basis functions.

    Note:
        For multiple variables, this creates an isotropic smooth, meaning all
        variables are treated on the same scale. If variables have very different
        scales or units, consider using [`TensorSmooth`][pymgcv.terms.TensorSmooth]
        for scale-invariant smoothing.

    Args:
        *varnames: Names of variables to smooth over. For single variables,
            creates a univariate smooth. For multiple variables, creates an
            isotropic multi-dimensional smooth (use with caution for different scales).
        k: Maximum number of basis functions (dimension of basis). If None,
            mgcv chooses automatically. Higher values allow more complexity but
            risk overfitting.
        bs: Basis function type. If None, uses thin plate splines. See
            pymgcv.bases for available options (CubicSpline, BSpline, etc.).
        m: Order of penalty (affects smoothness). If None, uses basis default.
            Higher values create smoother functions.
        by: Variable name for 'by' variable that scales the smooth. The smooth
            effect is multiplied by this variable's values.
        id: Identifier for grouping smooths with shared penalties. Use when
            you want multiple smooths to have the same smoothing parameter.
        fx: If True, creates a fixed-effect smooth (no smoothing parameter
            estimation). Useful when you want a specific amount of smoothing.
    """

    varnames: tuple[str, ...]
    k: int | None
    bs: BasisLike | None
    m: int | None
    by: str | None
    id: str | None
    fx: bool

    def __init__(
        self,
        *varnames: str,
        k: int | None = None,
        bs: BasisLike | None = None,
        m: int | None = None,
        by: str | None = None,
        id: str | None = None,
        fx: bool = False,
    ):
        if len(varnames) == 0:
            raise ValueError("Smooth terms require at least one variable")
        self.varnames = varnames
        self.k = k
        self.bs = bs
        self.m = m
        self.by = by
        self.id = id
        self.fx = fx

    def __str__(self) -> str:
        """Convert smooth term to mgcv formula syntax."""
        fx_value = None if not self.fx else self.fx  # Map False to None
        kwargs = {
            "k": self.k,
            "bs": self.bs,
            "m": self.m,
            "by": self.by,
            "id": self.id,
            "fx": fx_value,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        map_to_str = {
            "bs": lambda bs: f"'{bs}'",
            "id": lambda id: f"'{id}'",
            "fx": lambda fx: str(fx).upper(),
        }

        smooth_string = f"s({','.join(self.varnames)}"
        for key, val in kwargs.items():
            smooth_string += f",{key}={map_to_str.get(key, str)(val)}"
        return smooth_string + ")"

    def simple_string(self, formula_idx: int = 0) -> str:
        """Generate simplified smooth identifier."""
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        simple_string = f"s{idx}({','.join(self.varnames)})"

        if self.by is not None:
            simple_string += f":{self.by}"

        return simple_string

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam: Any,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict (partial effect) and standard error for a smooth term."""
        data = data.copy()
        by = None if self.by is None else data.pop(self.by)
        if self.by is not None:
            by = data.pop(self.by)
        formula_idx = list(gam.model_specification.all_formulae.keys()).index(target)
        smooth_name = self.simple_string(formula_idx)
        smooths = list(gam.rgam.rx2["smooth"])
        labels = [smooth.rx2["label"][0] for smooth in smooths]
        smooth = smooths[labels.index(smooth_name)]

        required_cols = list(self.varnames)
        if self.by is not None:
            required_cols.append(self.by)

        data = data[required_cols]

        predict_mat = mgcv.PredictMat(smooth, data_to_rdf(data))
        first = round(smooth.rx2["first.para"][0])
        last = round(smooth.rx2["last.para"][0])
        coefs = rstats.coef(gam.rgam)[(first - 1) : last]
        pred = rbase.drop(predict_mat @ coefs)
        # TODO factor by?
        pred = pred if by is None else pred * by

        # Compute SEs
        covariance = rbase.as_matrix(
            gam.rgam.rx2["Vp"].rx(rbase.seq(first, last), rbase.seq(first, last)),
        )
        se = rbase.sqrt(
            rbase.rowSums(
                (predict_mat @ covariance).ro * predict_mat,
            ),
        )
        se = rbase.pmax(0, se)
        return to_py(pred), to_py(se)


def _sequence_to_rvec_str(seq: Sequence, converter: Callable[[Any], str] = str) -> str:
    """Convert Python sequence to R vector string representation.

    Args:
        seq: Python sequence to convert
        converter: Function to convert individual elements to strings

    Returns:
        R vector syntax string, e.g., "c(1,2,3)" or "c('a','b','c')"
    """
    return f"c({','.join(converter(x) for x in seq)})"


# TODO, tensnor smooth d=c(2,1), common, in which case all the sequences should be length 2.
# Worth testing this case. A more intuitive interface would be to specify a list of smooths.
@dataclass
class TensorSmooth(TermLike):
    """Tensor product smooth for scale-invariant multi-dimensional smoothing.

    Tensor smooths create smooth functions of multiple variables that are
    scale-invariant, meaning the smoothing is appropriate even when variables
    have different units or scales.

    Args:
        *varnames: Names of variables for the tensor smooth. Usually 2-4 variables.
        k: Sequence of basis dimensions for each variable. If None, uses mgcv defaults.
            Length must match number of variables.
        bs: Sequence of basis types for each variable. If None, uses defaults.
            Can mix different basis types for different variables.
        d: Sequence specifying the dimension of each variable's smooth.
            Rarely needed - defaults are usually appropriate.
        m: Sequence of penalty orders for each variable. Controls smoothness.
        by: Variable name for 'by' variable scaling the tensor smooth.
        id: Identifier for sharing penalties across multiple tensor smooths.
        fx: If True, fix smoothing parameters (no automatic estimation).
        np: If False, use a single penalty for the tensor product.
            If True (default), use separate penalties for each marginal.
        interaction_only: If True, creates ti() instead of te() - interaction only,
            excluding main effects of individual variables.
    """

    varnames: tuple[str, ...]
    k: tuple[int, ...] | None
    bs: tuple[BasisLike, ...] | None
    d: tuple[int, ...] | None
    m: tuple[int, ...] | None
    by: str | None
    id: str | None
    fx: bool | None
    np: bool | None
    interaction_only: bool

    def __init__(
        self,
        *varnames: str,
        k: Sequence[int] | None = None,
        bs: Sequence[BasisLike] | None = None,
        d: Sequence[int] | None = None,
        m: Sequence[int] | None = None,
        by: str | None = None,
        id: str | None = None,
        fx: bool = False,
        np: bool = True,
        interaction_only: bool = False,
    ):
        """Initialize a tensor smooth term.

        Args:
            *varnames: Variable names for the tensor smooth (2+ recommended).
            k: Basis dimensions for each variable (None for automatic).
            bs: Basis types for each variable (None for defaults).
            d: Dimensions for each variable's smooth (rarely needed).
            m: Penalty orders for each variable (None for defaults).
            by: By variable name (None for no by variable).
            id: Identifier for shared penalties (None for unique).
            fx: Fix smoothing parameters (False for automatic estimation).
            np: Use separate penalties for marginals (True recommended).
            interaction_only: Create ti() instead of te() (False for full tensor).
        """
        if len(varnames) < 2:
            raise ValueError("Tensor smooths require at least 2 variables")

        self.varnames = varnames
        self.k = tuple(k) if k is not None else None
        self.bs = tuple(bs) if bs is not None else None
        self.d = tuple(d) if d is not None else None
        self.m = tuple(m) if m is not None else None
        self.by = by
        self.id = id
        self.fx = None if not fx else fx
        self.np = None if not np else np
        self.interaction_only = interaction_only

    def __str__(self) -> str:
        """Convert tensor smooth to mgcv formula syntax.

        Returns:
            String in mgcv te() or ti() syntax, e.g., "te(x1,x2,k=c(10,15))"
        """
        kwargs = {
            "k": self.k,
            "bs": self.bs,
            "d": self.d,
            "m": self.m,
            "by": self.by,
            "id": self.id,
            "fx": self.fx,
            "np": self.np,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        map_to_str = {
            "k": _sequence_to_rvec_str,
            "bs": lambda bs: _sequence_to_rvec_str(bs, converter=lambda b: f"'{b}'"),
            "d": _sequence_to_rvec_str,
            "m": _sequence_to_rvec_str,
            "id": lambda id: f"'{id}'",
            "fx": lambda fx: str(fx).upper(),
            "np": lambda np: str(np).upper(),
        }

        prefix = "ti" if self.interaction_only else "te"
        smooth_string = f"{prefix}({','.join(self.varnames)}"
        for key, val in kwargs.items():
            smooth_string += f",{key}={map_to_str.get(key, str)(val)}"
        return smooth_string + ")"

    def simple_string(self, formula_idx: int = 0) -> str:
        """Generate simplified tensor smooth identifier.

        Returns:
            Simplified identifier like "te(x1,x2)" or "ti.1(x1,x2):by_var"
        """
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        prefix = "ti" if self.interaction_only else "te"
        simple_string = f"{prefix}{idx}({','.join(self.varnames)})"
        if self.by is not None:
            simple_string += ":" + self.by
        return simple_string

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam: Any,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict (partial effect) and standard error for a smooth term."""
        data = data.copy()
        by = None if self.by is None else data.pop(self.by)
        if self.by is not None:
            by = data.pop(self.by)
        formula_idx = list(gam.model_specification.all_formulae.keys()).index(target)
        smooth_name = self.simple_string(formula_idx)
        smooths = list(gam.rgam.rx2["smooth"])
        labels = [smooth.rx2["label"][0] for smooth in smooths]
        smooth = smooths[labels.index(smooth_name)]

        required_cols = list(self.varnames)
        if self.by is not None:
            required_cols.append(self.by)

        data = data[required_cols]

        predict_mat = mgcv.PredictMat(smooth, data_to_rdf(data))
        first = round(smooth.rx2["first.para"][0])
        last = round(smooth.rx2["last.para"][0])
        coefs = rstats.coef(gam.rgam)[(first - 1) : last]
        pred = rbase.drop(predict_mat @ coefs)
        # TODO factor by?
        pred = pred if by is None else pred * by

        # Compute SEs
        covariance = rbase.as_matrix(
            gam.rgam.rx2["Vp"].rx(rbase.seq(first, last), rbase.seq(first, last)),
        )
        se = rbase.sqrt(
            rbase.rowSums(
                (predict_mat @ covariance).ro * predict_mat,
            ),
        )
        se = rbase.pmax(0, se)
        return to_py(pred), to_py(se)


@dataclass
class Offset(TermLike):
    """Offset term, added to the linear predictor as is.

    This means:
    - For log-link models: offset induces a multiplicative effect on the response scale
    - For identity-link models: an offset induces an additive effect on the response
        scale

    Args:
        name: Name of the variable to use as an offset. Must be present in the modeling
        data.
    """

    varnames: tuple[str]
    by: str | None

    def __init__(self, name: str):
        self.varnames = (name,)
        self.by = None

    def __str__(self) -> str:
        """Return offset in mgcv formula syntax."""
        return f"offset({self.varnames[0]})"

    def simple_string(self, formula_idx: int = 0) -> str:
        """Return offset identifier.

        Note: mgcv doesn't include offsets as parametric terms in predictions,
        so formula_idx is not used and offsets are handled specially.
        """
        return f"offset({self.varnames[0]})"

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam: Any,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute offset partial effects.

        For offset terms, the partial effect is simply the offset variable
        values, with zero standard errors.
        """
        effect = data[self.varnames[0]].to_numpy()
        return effect, np.zeros_like(effect)
