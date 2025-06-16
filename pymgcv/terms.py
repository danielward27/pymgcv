"""The available terms for constructing GAM models."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.basis_functions import BasisLike, CubicSpline, ThinPlateSpline
from pymgcv.converters import data_to_rdf, to_py

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
        by: Optional name of a 'by' variable that scales this term.
    """

    varnames: tuple[str, ...]
    by: str | None

    def __str__(self) -> str:
        """Convert the term to mgcv formula syntax."""
        ...

    def simple_string(self, formula_idx: int = 0) -> str:
        """Generate a simplified identifier for the term.

        This representation is used internally for term identification and
        matches the format used by mgcv when predicting separate terms.
        For multi-formula models, mgcv adds an index suffix to distinguish
        terms from different formulae.

        Args:
            formula_idx: Index of the formula in multi-formula models.
        """
        ...

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam: Any,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute partial effects and standard errors for this term.

        Args:
            data: DataFrame containing predictor variables.
            gam: Fitted GAM model object.
            target: Name of the target variable (response or family parameter).

        Returns:
            Tuple of (effects, standard_errors) as numpy array.
        """


@dataclass
class Linear(TermLike):
    """Linear (parametric) term with no basis expansion.

    If the variable is a categorical variable, the term will be expanded (one-hot
    encoded into a set of dummy variables. Otherwise, the variable is included
    as is and the effect computed by a single coefficient multiplied by the variable
    value.

    Args:
        name: Name of the variable to include as a linear term.
    """

    varnames: tuple[str]
    by: str | None

    def __init__(self, name: str):
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

    Any categorical variables involved in an interaction are expanded into indicator
    variables representing all combinations at the specified interaction order.
    Numeric variables are incorporated by multiplication (i.e. with eachother and
    any indicator variables).

    Note, this does not automatically include main effects or lower order interactions.

    Args:
        *varnames: Variable names to include in the interaction. Can be any
            number of variables.

    Examples:
        ```python
        # Two-way interaction (multiplication)
        age_income = Interaction('age', 'income')

        # Three-way interaction
        varnames = ['group0', 'group1', 'group2']
        three_way = Interaction(*varnames)

        # Generate all pairwise interactions
        from itertools import combinations
        pairwise = [Interaction(*pair) for pair in combinations(varnames, 2)]
        ```
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
        For multiple variables, this creates an isotropic smooth, meaning all variables
        are treated on the same scale. If variables have very different scales or units,
        consider using [`TensorSmooth`][pymgcv.terms.TensorSmooth].

    Args:
        *varnames: Names of variables to smooth over. For single variables,
            creates a univariate smooth. For multiple variables, creates an
            isotropic multi-dimensional smooth.
        k: The dimension of the basis used to represent the smooth term. The
            default depends on the basis and number of variables that the smooth is a
            function of (has placeholder of -1).
        bs: Basis function. For available options see
            [Basis Functions](./api/basis_functions.md). Default to
            [`ThinPlateSpline`][pymgcv.basis_functions.ThinPlateSpline].
        by: variable name used to scale the smooth. If it's a numeric vector, it
            scales the smooth, and the "by" variable should not be included as a
            seperate main effect (as the smooth is usually not centered). If the "by"
            variable is a factor, a separate smooth is created for each factor level.
            These smooths are centered, so the factor typically should be included as a
            main effect.
        id: Identifier for grouping smooths with shared penalties. If using a
            categorical by variable, providing an id will ensure a shared smoothing
            parameter for each level.
        fx: Indicates whether the term is a fixed d.f. regression spline (True) or a
            penalized regression spline (False).
    """

    varnames: tuple[str, ...]
    by: str | None
    k: int
    bs: BasisLike
    id: str | None
    fx: bool

    def __init__(
        self,
        *varnames: str,
        by: str | None = None,
        k: int = -1,
        bs: BasisLike | None = None,
        id: str | None = None,
        fx: bool = False,
    ):
        if len(varnames) == 0:
            raise ValueError("Smooth terms require at least one variable")
        self.varnames = varnames
        self.k = k
        self.bs = bs if bs is not None else ThinPlateSpline()
        self.by = by
        self.id = id
        self.fx = fx

    def __str__(self) -> str:
        """Convert smooth term to mgcv formula syntax."""
        from_basis = self.bs._pass_to_s()
        m = from_basis.get("m", _AsVar("NA"))
        xt = from_basis.get("xt", _AsVar("NULL"))
        kwargs = {
            "k": self.k,
            "fx": self.fx,
            "bs": str(self.bs),
            "m": m,
            "by": _AsVar(self.by if self.by is not None else "NA"),
            "xt": xt,  # TODO xt as var!
            "id": self.id if self.id is not None else _AsVar("NULL"),
        }
        kwarg_strings = [f"{k}={_to_r_literal_string(v)}" for k, v in kwargs.items()]
        return f"s({','.join(self.varnames)},{','.join(kwarg_strings)})"

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
        return _smooth_or_tensorsmooth_partial_effect(
            term=self,
            target=target,
            gam=gam,
            data=data,
        )


# TODO, tensnor smooth d=c(2,1), common, in which case all the sequences should be length 2.
# Worth testing this case. A far more intuitive interface would be to specify a list of smooths.
# But that deviates pretty far from the mgcv way.
@dataclass
class TensorSmooth(TermLike):
    """Tensor product smooth for scale-invariant multi-dimensional smoothing.

    Tensor smooths create smooth functions of multiple variables using marginal
    smooths in order to be robust to variables on different scales.
    """

    varnames: tuple[str, ...]
    by: str | None
    k: tuple[int, ...] | None
    bs: tuple[BasisLike, ...]
    d: tuple[int, ...] | None
    id: str | None
    fx: bool
    np: bool
    interaction_only: bool

    def __init__(
        self,
        *varnames: str,
        by: str | None = None,
        k: Sequence[int] | None = None,
        bs: Sequence[BasisLike] | None = None,
        d: Sequence[int] | None = None,
        id: str | None = None,
        fx: bool = False,
        np: bool = True,
        interaction_only: bool = False,
    ):
        """Initialize a tensor smooth term.

        For the sequence arguments, the length must match the number of variables if
        ``d`` is not provided, else they must match the length of ``d``.

        Args:
            *varnames: Names of variables for the tensor smooth.
            k: Sequence of basis dimensions for each marginal basis.
            bs: Sequence of basis types for each variable. If None, uses defaults.
                Can mix different basis types for different variables.
            d: Sequence specifying the dimension of each variable's smooth. For example,
                (2, 1) would specify to use one two dimensional marginal smooth and one
                1 dimensional marginal smooth. This is useful for space-time smooths (2
                dimensional space and 1 time dimension).
            by: Variable name for 'by' variable scaling the tensor smooth.
            id: Identifier for sharing penalties across multiple tensor smooths.
            fx: indicates whether the term is a fixed d.f. regression spline (True) or
                a penalized regression spline (False). Defaults to False.
            np: If False, use a single penalty for the tensor product.
                If True (default), use separate penalties for each marginal.
            interaction_only: If True, creates ti() instead of te() - interaction only,
                excluding main effects of individual variables.
        """
        if len(varnames) < 2:
            raise ValueError("Tensor smooths require at least 2 variables")

        if d is None:
            d = (1,) * len(varnames)

        if bs is None:
            bs = (CubicSpline(),) * len(d)  # TODO better default for >1d

        self.varnames = varnames
        self.k = tuple(k) if k is not None else None
        self.bs = tuple(bs)
        self.d = tuple(d) if d is not None else None
        self.by = by
        self.id = id
        self.fx = fx
        self.np = np
        self.interaction_only = interaction_only

    def __str__(self) -> str:
        """Convert tensor smooth to mgcv formula syntax.

        Returns:
            String in mgcv te() or ti() syntax, e.g., "te(x1,x2,k=c(10,15))"
        """
        from_bases = [bs._pass_to_s() for bs in self.bs]
        ms = [bs.get("m", _AsVar("NA")) for bs in from_bases]
        xts = [bs.get("xt", _AsVar("NULL")) for bs in from_bases]
        kwargs = {
            "k": self.k if self.k is not None else _AsVar("NA"),
            "bs": self.bs,
            "m": ms,
            "d": self.d,
            "by": _AsVar(self.by if self.by is not None else "NA"),
            "fx": self.fx,
            "np": self.np,
            "xt": xts,
            "id": self.id,
        }
        kwarg_strings = [f"{k}={_to_r_literal_string(v)}" for k, v in kwargs.items()]
        print(f"({','.join(self.varnames)},{','.join(kwarg_strings)})")
        prefix = "ti" if self.interaction_only else "te"
        return f"{prefix}({','.join(self.varnames)},{','.join(kwarg_strings)})"

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
        return _smooth_or_tensorsmooth_partial_effect(
            target=target,
            term=self,
            gam=gam,
            data=data,
        )


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


@singledispatch
def _to_r_literal_string(arg: object) -> str:
    """Attempts to convert simple types into a string representation in R.

    Currently, any (non string) sequence will be converted to a vector
    i.e. c(...). Note, strings are quoted (so cannot be used for variables).
    Wrap in _AsVar if it needs to be passed as a variable.
    """
    return f"'{str(arg)}'"


@dataclass
class _AsVar:
    varname: str


@_to_r_literal_string.register
def _(arg: _AsVar) -> str:
    return arg.varname


@_to_r_literal_string.register
def _(arg: bool) -> str:  # noqa: FBT001
    return str(arg).upper()


@_to_r_literal_string.register
def _(arg: int) -> str:
    return str(arg)


@_to_r_literal_string.register
def _(arg: Sequence) -> str:
    if isinstance(arg, str):
        return f"'{arg}'"
    return f"c({','.join([_to_r_literal_string(item) for item in arg])})"


def _smooth_or_tensorsmooth_partial_effect(
    target: str,
    term: TensorSmooth | Smooth,
    gam: Any,
    data: pd.DataFrame,
):
    """Predict (partial effect) and standard error for a smooth term."""
    data = data.copy()
    by = None if term.by is None else data.pop(term.by)
    if term.by is not None:
        by = data.pop(term.by)
    formula_idx = list(gam.model_specification.all_formulae.keys()).index(target)
    smooth_name = term.simple_string(formula_idx)
    smooths = list(gam.rgam.rx2["smooth"])
    labels = [smooth.rx2["label"][0] for smooth in smooths]
    smooth = smooths[labels.index(smooth_name)]

    required_cols = list(term.varnames)
    if term.by is not None:
        required_cols.append(term.by)

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
