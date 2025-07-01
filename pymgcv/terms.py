"""The available terms for constructing GAM models."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.basis_functions import BasisLike, CubicSpline, ThinPlateSpline
from pymgcv.converters import data_to_rdf, to_py
from pymgcv.formula_utils import _to_r_constructor_string, _Var

mgcv = importr("mgcv")
rbase = importr("base")
rstats = importr("stats")


# TODO: Not supporting 'sp' or 'pc' basis types.


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

    def label(self) -> str:
        """The label used by pymgcv for the term in plotting and columns.

        All labels must be unique in a formula. Labels should be implemented
        such that each unique label must map to a unique mgcv identifier (but
        not necessarily the other way around).
        """
        ...

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Generate the mgcv identifier for the term.

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
        *,
        data: pd.DataFrame,
        rgam: Any,
        formula_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute partial effects and standard errors for this term.

        Args:
            data: DataFrame containing predictor variables.
            rgam: Fitted mgcv gam.
            formula_idx: The formula index of the term.

        Returns:
            Tuple of (effects, standard_errors) as numpy array.
        """
        ...

    def __add__(self, other):
        """Supports adding of terms to create lists of terms."""
        ...

    def __radd__(self, other):
        """Supports adding of terms to create lists of terms."""
        ...


class _AddMixin:
    def __add__(self, other) -> list:
        """Mixin class allowing adding of terms to terms and lists.

        It is important to note this is doing nothing but defining a list of terms.
        It arguably makes formulas more readable though!
        """
        if isinstance(other, list):
            return [self] + other
        if isinstance(other, TermLike):
            return [self, other]
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, list):
            return other + [self]
        if isinstance(other, TermLike):
            return [other, self]
        return NotImplemented


@dataclass
class L(_AddMixin, TermLike):
    """Linear (parametric) term with no basis expansion.

    If the variable is a categorical variable, the term will be expanded (one-hot
    encoded into a set of dummy variables. Otherwise, the variable is included
    as is and the effect computed by a single coefficient multiplied by the variable
    value.

    Args:
        name: Name of the variable to include as a linear term.
    """

    varnames: tuple[str, ...]
    by: str | None

    def __init__(self, name: str):
        self.varnames = (name,)
        self.by = None

    def __str__(self) -> str:
        """Return variable name for mgcv formula."""
        return self.varnames[0]

    def label(self) -> str:
        return f"L({self.varnames[0]})"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Return term identifier with optional formula index."""
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        return self.varnames[0] + idx

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame,
        rgam: Any,
        formula_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _parameteric_partial_effect(
            term=self,
            data=data,
            rgam=rgam,
            formula_idx=formula_idx,
        )


@dataclass
class Interaction(_AddMixin, TermLike):
    """Parametric interaction term between multiple variables.

    Any categorical variables involved in an interaction are expanded into indicator
    variables representing all combinations at the specified interaction order.
    Numeric variables are incorporated by multiplication (i.e. with eachother and
    any indicator variables).

    Note, this does not automatically include main effects or lower order interactions.

    Args:
        *varnames: Variable names to include in the interaction. Can be any
            number of variables.

    !!! example

        ```python
        # Two-way interaction (multiplication if both numeric)
        from pymgcv.terms import Interaction
        age_income = Interaction('age', 'income')

        # Three-way interaction
        varnames = ['group0', 'group1', 'group2']
        three_way = Interaction(*varnames)

        # Generate all pairwise interactions
        from itertools import combinations
        pairs = [Interaction(*pair) for pair in combinations(varnames, 2)]
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

    def label(self):
        return f"Interaction({','.join(self.varnames)})"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Return interaction identifier with optional formula index."""
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        return ":".join(self.varnames) + idx

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame,
        rgam: Any,
        formula_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute interaction term partial effects.

        Creates the design matrix for the interaction and computes effects
        using the fitted coefficients and covariance matrix.
        """
        return _parameteric_partial_effect(
            term=self,
            data=data,
            rgam=rgam,
            formula_idx=formula_idx,
        )


@dataclass
class S(_AddMixin, TermLike):
    """S term using spline basis functions.

    Note:
        For multiple variables, this creates an isotropic smooth, meaning all variables
        are treated on the same scale. If variables have very different scales or units,
        consider using [`T`][pymgcv.terms.T].

    Args:
        *varnames: Names of variables to smooth over. For single variables,
            creates a univariate smooth. For multiple variables, creates an
            isotropic multi-dimensional smooth.
        k: The dimension of the basis used to represent the smooth term. The
            default depends on the basis and number of variables that the smooth is a
            function of (has placeholder of -1).
        bs: Basis function. For available options see
            [Basis Functions](./basis_functions.md). Default to
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
            raise ValueError("S terms require at least one variable")
        self.varnames = varnames
        self.k = k
        self.bs = bs if bs is not None else ThinPlateSpline()
        self.by = by
        self.id = id
        self.fx = fx

    def __str__(self) -> str:
        """Convert smooth term to mgcv formula syntax."""
        from_basis = self.bs._pass_to_s()
        m = from_basis.get("m", ro.NA_Logical)
        xt = from_basis.get("xt", ro.NULL)

        kwargs = {
            "by": _Var(self.by) if self.by is not None else ro.NA_Logical,
            "k": self.k,
            "bs": str(self.bs),
            "m": m,
            "xt": xt,
            "id": self.id if self.id is not None else ro.NULL,
            "fx": self.fx,
        }
        kwarg_strings = [
            f"{k}={_to_r_constructor_string(v)}" for k, v in kwargs.items()
        ]
        defaults = [
            "k=-1L",
            "fx=FALSE",
            'bs="tp"',  # Note quote order matters!
            "m=NA",
            "by=NA",
            "xt=NULL",
            "id=NULL",
        ]
        kwarg_strings = [s for s in kwarg_strings if s not in defaults]
        kwarg_string = "" if len(kwarg_strings) == 0 else f",{','.join(kwarg_strings)}"
        return f"s({','.join(self.varnames)}{kwarg_string})"

    def label(self) -> str:
        by = f",by={self.by}" if self.by else ""
        return f"S({','.join(self.varnames)}{by})"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Generate simplified smooth identifier."""
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        mgcv_identifier = f"s{idx}({','.join(self.varnames)})"

        if self.by is not None:
            mgcv_identifier += f":{self.by}"

        return mgcv_identifier

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame,
        rgam: Any,
        formula_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _smooth_partial_effect(
            data=data,
            term=self,
            rgam=rgam,
            formula_idx=formula_idx,
        )


# A far more intuitive interface would be to specify a list of smooths.
# But that deviates pretty far from the mgcv way.
@dataclass
class T(_AddMixin, TermLike):
    """Tensor product smooth for scale-invariant multi-dimensional smoothing.

    Tensor smooths create smooth functions of multiple variables using marginal
    smooths in order to be robust to variables on different scales.
    """

    varnames: tuple[str, ...]
    by: str | None
    k: int | tuple[int, ...] | None
    bs: BasisLike | tuple[BasisLike, ...]
    d: tuple[int, ...] | None
    id: str | None
    fx: bool
    np: bool
    interaction_only: bool

    def __init__(
        self,
        *varnames: str,
        by: str | None = None,
        k: int | Iterable[int] | None = None,
        bs: BasisLike | Iterable[BasisLike] | None = None,
        d: Iterable[int] | None = None,
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
            k: The basis dimension for each marginal smooth. If an integer, all
                marginal smooths will have the same basis dimension.
            bs: basis type to use, or an iterable of basis types for each marginal
                smooth.
            d: Sequence specifying the dimension of each variable's smooth. For example,
                (2, 1) would specify to use one two dimensional marginal smooth and one
                1 dimensional marginal smooth, where three variables are provided. This
                is useful for space-time smooths (2 dimensional space and 1 time
                dimension).
            by: Variable name for 'by' variable scaling the tensor smooth, or creating
                a smooth for each level of a categorical by variable.
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

        bs = CubicSpline() if bs is None else bs
        self.varnames = varnames
        self.k = k if isinstance(k, int) else (None if k is None else tuple(k))
        self.bs = bs if isinstance(bs, BasisLike) else tuple(bs)
        self.d = tuple(d) if d is not None else d
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
        if isinstance(self.bs, BasisLike):
            from_bases = self.bs._pass_to_s()
            m = from_bases.get("m", ro.NA_Logical)
            xt = from_bases.get("xt", ro.NULL)
        else:
            from_bases = [bs._pass_to_s() for bs in self.bs]
            m = [bs.get("m", ro.NA_Logical) for bs in from_bases]

            if all(x == ro.NA_Logical for x in m):
                m = ro.NA_Logical

            xt = [bs.get("xt", ro.NULL) for bs in from_bases]
            if all(x == ro.NULL for x in xt):
                xt = ro.NULL

        if isinstance(self.bs, BasisLike):
            bs = str(self.bs)
        else:
            bs = ro.StrVector([str(bs) for bs in self.bs])

        kwargs = {
            "by": _Var(self.by) if self.by is not None else ro.NA_Logical,
            "k": self.k if self.k is not None else ro.NA_Logical,
            "bs": bs,
            "m": m,
            "d": ro.NA_Logical if self.d is None else ro.IntVector(self.d),
            "id": self.id if self.id is not None else ro.NULL,
            "fx": self.fx,
            "np": self.np,
            "xt": xt,
        }
        kwarg_strings = [
            f"{k}={_to_r_constructor_string(v)}" for k, v in kwargs.items()
        ]
        defaults = [
            "by=NA",
            "k=NA",
            'bs="cr"',
            "m=NA",
            "d=NA",
            "fx=FALSE",
            "np=TRUE",
            "xt=NULL",
            "id=NULL",
        ]
        kwarg_strings = [s for s in kwarg_strings if s not in defaults]
        kwarg_string = "" if len(kwarg_strings) == 0 else f",{','.join(kwarg_strings)}"
        prefix = "ti" if self.interaction_only else "te"
        return f"{prefix}({','.join(self.varnames)}{kwarg_string})"

    def label(self) -> str:
        by = f",by={self.by}" if self.by else ""
        return f"T({','.join(self.varnames)}{by})"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Generate simplified tensor smooth identifier.

        Returns:
            Simplified identifier like "te(x1,x2)" or "ti.1(x1,x2):by_var"
        """
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        prefix = "ti" if self.interaction_only else "te"
        mgcv_identifier = f"{prefix}{idx}({','.join(self.varnames)})"
        if self.by is not None:
            mgcv_identifier += ":" + self.by
        return mgcv_identifier

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame,
        rgam: Any,
        formula_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _smooth_partial_effect(
            term=self,
            formula_idx=formula_idx,
            rgam=rgam,
            data=data,
        )


@dataclass
class Offset(_AddMixin, TermLike):
    """Offset term, added to the linear predictor as is.

    This means:
    - For log-link models: offset induces a multiplicative effect on the response scale
    - For identity-link models: an offset induces an additive effect on the response
        scale

    Args:
        name: Name of the variable to use as an offset. Must be present in the modeling
            data.
    """

    varnames: tuple[str, ...]
    by: str | None

    def __init__(self, name: str):
        self.varnames = (name,)
        self.by = None

    def __str__(self) -> str:
        """Return offset in mgcv formula syntax."""
        return f"offset({self.varnames[0]})"

    def label(self) -> str:
        return f"Offset({self.varnames[0]})"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Return offset identifier.

        Note: mgcv doesn't include offsets as parametric terms in predictions,
        so formula_idx is not used and offsets are handled specially.
        """
        return f"offset({self.varnames[0]})"

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame,
        rgam: Any,
        formula_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute offset partial effects.

        For offset terms, the partial effect is simply the offset variable
        values, with zero standard errors.
        """
        effect = data[self.varnames[0]].to_numpy()
        return effect, np.zeros_like(effect)


def _mgcv_smooth_prediction_and_se(
    *,
    mgcv_smooth: Any,
    data: pd.DataFrame,
    rgam: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute prediction and standard error for a smooth given data."""
    predict_mat = mgcv.PredictMat(mgcv_smooth, data_to_rdf(data))
    first = round(mgcv_smooth.rx2["first.para"][0])
    last = round(mgcv_smooth.rx2["last.para"][0])
    coefs = rstats.coef(rgam)[(first - 1) : last]
    pred = rbase.drop(predict_mat @ coefs)
    cov = rbase.as_matrix(
        rgam.rx2["Vp"].rx(rbase.seq(first, last), rbase.seq(first, last)),
    )
    se = rbase.sqrt(rbase.rowSums((predict_mat @ cov).ro * predict_mat))
    se = rbase.pmax(0, se)
    return to_py(pred), to_py(se)


class Intercept(_AddMixin, TermLike):
    """Intercept term.

    By default, this is added to all formulas in the model. If you want to control
    the intercept term, then [`GAM`][pymgcv.gam.GAM] should have ``add_intercepts``
    set to `False`, in which case, only intercepts explicitly added will be included
    in the model.
    """

    varnames: tuple[str, ...]
    by: str | None

    def __init__(self):
        self.varnames = ()
        self.by = None

    def __str__(self) -> str:
        """Return variable name for mgcv formula."""
        return "1"

    def label(self) -> str:
        return "Intercept"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Return term identifier with optional formula index."""
        return f"(Intercept){'' if formula_idx == 0 else f'.{formula_idx}'}"

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame,
        rgam: Any,
        formula_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        coef = rstats.coef(rgam)
        intercept_name = self.mgcv_identifier(formula_idx)
        cov = rgam.rx2("Vp")
        cov.rownames, cov.colnames = coef.names, coef.names
        var = cov.rx2(intercept_name, intercept_name)
        intercept = to_py(coef.rx2[intercept_name]).item()
        se = np.sqrt(var).item()
        return np.full(len(data), intercept), np.full(len(data), se)


@dataclass
class _RandomWigglyToByInterface(_AddMixin, TermLike):
    """This wraps a term using a RandomWigglyCurve basis to a term with a by variable.

    This isn't a real usable term, but provides a consistent interface for plotting
    between categorical by variables and random wiggly curve terms.
    """

    random_wiggly_term: S | T
    by: str | None

    def __init__(self, random_wiggly_term: S | T):
        # TODO edge case of by being set and "fs" basis being used!
        self.random_wiggly_term = random_wiggly_term
        self.by = random_wiggly_term.varnames[-1]
        self.varnames = random_wiggly_term.varnames[:-1]

    def __str__(self) -> str:
        """Return variable name for mgcv formula."""
        raise NotImplementedError()

    def label(self) -> str:
        return self.random_wiggly_term.label()

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        raise NotImplementedError()

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame,
        rgam: Any,
        formula_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.random_wiggly_term._partial_effect(
            data=data,
            rgam=rgam,
            formula_idx=formula_idx,
        )


def _parameteric_partial_effect(
    term: L | Interaction,
    data: pd.DataFrame,
    rgam: Any,
    formula_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute partial effects for parametric terms.

    Creates the (naive) design matrix then intersects this with the
    parameters present in mgcv, as mgcv may drop columns of the
    design matrix, e.g. to avoid indeterminacy.
    """
    data = data[list(term.varnames)]
    predict_mat = rstats.model_matrix(
        ro.Formula(f"~{str(term)}-1"),
        data=data_to_rdf(data),
    )
    post_fix = "" if formula_idx == 0 else f".{formula_idx}"
    predict_mat.colnames = rbase.paste0(predict_mat.colnames, post_fix)
    coef_names = rbase.intersect(predict_mat.colnames, rstats.coef(rgam).names)

    predict_mat = rbase.as_matrix(predict_mat.rx(True, coef_names))
    coefs = rstats.coef(rgam).rx(coef_names)

    fitted_vals = predict_mat @ coefs
    cov = rgam.rx2["Vp"]
    cov.rownames = rstats.coef(rgam).names
    cov.colnames = rstats.coef(rgam).names
    subcov = cov.rx(coef_names, coef_names)
    se = rbase.sqrt(rbase.rowSums((predict_mat @ subcov).ro * predict_mat))
    return to_py(fitted_vals).squeeze(axis=-1), to_py(se)


def _smooth_partial_effect(
    *,
    formula_idx: int,
    term: T | S,
    rgam: Any,
    data: pd.DataFrame,
):
    """Predict (partial effect) and standard error for a S or T term."""
    data = data.copy()
    required_cols = list(term.varnames)

    if term.by is not None:
        required_cols.append(term.by)

    data = data[required_cols]

    smooth_name = term.mgcv_identifier(formula_idx)
    smooths = {s.rx2["label"][0]: s for s in rgam.rx2["smooth"]}

    if term.by is not None and data[term.by].dtype == "category":
        levels = data[term.by].cat.categories

        fit_vals = np.empty(len(data))
        se = np.empty(len(data))

        for lev in levels:
            is_lev = data[term.by] == lev
            data_lev = data[is_lev]
            if len(data_lev) == 0:
                continue
            smooth = smooths[smooth_name + lev]
            fit_lev, se_lev = _mgcv_smooth_prediction_and_se(
                mgcv_smooth=smooth,
                data=data_lev,
                rgam=rgam,
            )
            fit_vals[is_lev] = fit_lev
            se[is_lev] = se_lev
    else:
        fit_vals, se = _mgcv_smooth_prediction_and_se(
            mgcv_smooth=smooths[smooth_name],
            data=data,
            rgam=rgam,
        )

    return fit_vals, se
