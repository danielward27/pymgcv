"""Defines the different types of terms available in pymgcv."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.bases import BasisLike
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
    """Protocol defining interface for terms of a GAM model.

    Attributes:
        varnames: The name or names of the variables in the term.
        by: The name of a by variable or None if not present.
    """

    varnames: tuple[str, ...]
    by: str | None

    def __str__(self) -> str:
        """The string representation of the term which will be passed to mgcv."""
        ...

    def simple_string(self, formula_idx: int = 0) -> str:
        """A simplified representation of the term.

        This should match the form used by mgcv when predicting the seperate terms of
        the model. For multiformula models, mgcv adds an index to distinguish which
        formula the term corresponds to. This is supported through ``formula_idx``.
        """
        ...

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict (partial effects) and standard errors for the term given data."""


@dataclass
class Linear(TermLike):
    """A linear term (factor or numeric term) i.e. no basis expansion.

    Args:
        name: The name of the variable to include as a linear term.

    """

    varnames: tuple[str]
    by: str | None

    def __init__(self, name: str):
        self.varnames = (name,)
        self.by = None

    def __str__(self) -> str:
        return self.varnames[0]

    def simple_string(self, formula_idx: int = 0) -> str:
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        return self.varnames[0] + idx

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
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
    """An interaction term in the model.

    This defines the specific interaction term specified only. Remember to
    include any linear or lower order interactions explicitly, if desired.
    If there are many lower order interactions, it may be convenient to
    construct them using ``itertools``, e.g. to get all pairwise interactions:
    ```
    from itertools import combinations
    from pymgcv.terms import Interaction
    varnames = ['x1', 'x2', 'x3']
    interactions = [Interaction(*pair) for pair in combinations(varnames, 2)]
    ```
    """

    varnames: tuple[str, ...]
    by: str | None

    def __init__(self, *varnames: str):
        """Initialize an interaction term.

        Args:
            *varnames: The names of the variables to include in the interaction.
        """
        self.varnames = tuple(varnames)
        self.by = None

    def __str__(self) -> str:
        return ":".join(self.varnames)

    def simple_string(self, formula_idx: int = 0) -> str:
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        return ":".join(self.varnames) + idx

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
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
    """Create a smooth term.

    Note that terms like ``Smooth("x1", "x2")`` are isotropic. This means for
    example variables should be of similar scales/units. For scale invariant smoothing
    of multiple variables, see ``TensorSmooth``.
    """  # TODO properly reference TensorSmooth

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
        self.varnames = varnames
        self.k = k
        self.bs = bs
        self.m = m
        self.by = by
        self.id = id
        self.fx = fx

    def __str__(self) -> str:
        """Returns the mgcv smooth term as a string."""
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
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        simple_string = f"s{idx}({','.join(self.varnames)})"

        if self.by is not None:
            simple_string += f":{self.by}"

        return simple_string

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam,
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


def _sequence_to_rvec_str(seq: Sequence, converter: Callable[[Any], str] = str):
    return f"c({','.join(converter(x) for x in seq)})"


# TODO, tensnor smooth d=c(2,1), common, in which case all the sequences should be length 2.
# Worth testing this case. A more intuitive interface would be to specify a list of smooths.
@dataclass
class TensorSmooth(TermLike):
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
        """A tensor smooth between two or more variables.

        Unlike ``Smooth``, this (by default) is scale invariant, so is useful
        for modelling interactions for variables on different scales.

        Args:
            *varnames: The variable names to smooth.
            k: _description_. Defaults to None.
            bs: _description_. Defaults to None.
            d: _description_. Defaults to None.
            m: _description_. Defaults to None.
            by:  _description_. Defaults to None.
            id:  _description_. Defaults to None.
            fx:  Defaults to False.
            np:  Defaults to True.
            interaction_only: _description_. Defaults to False.
        """
        self.varnames = varnames
        self.k = tuple(k) if k is not None else None
        self.bs = tuple(bs) if bs is not None else None
        self.d = tuple(d) if d is not None else None
        self.m = tuple(m) if m is not None else None
        self.by = by
        self.id = id
        self.fx = None if not fx else fx
        self.np = None if np else np
        self.interaction_only = interaction_only

    def __str__(self) -> str:
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
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        prefix = "ti" if self.interaction_only else "te"
        simple_string = f"{prefix}{idx}({','.join(self.varnames)})"
        if self.by is not None:
            simple_string += ":" + self.by
        return simple_string

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam,
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
    """A constant (zero parameter) offset term.

    Args:
        name: The name of the variable to use as an offset.
    """

    varnames: tuple[str]
    by: str | None

    def __init__(self, name: str):
        self.varnames = (name,)
        self.by = None

    def __str__(self) -> str:
        return f"offset({self.varnames[0]})"

    def simple_string(self, formula_idx: int = 0) -> str:
        # mgcv doesn't include it as a parametric term e.g. in predict terms so we have
        # to special case it. Notice formula_idx not used because of this.
        return f"offset({self.varnames[0]})"

    def _partial_effect(
        self,
        data: pd.DataFrame,
        gam,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        effect = data[self.varnames[0]].to_numpy()
        return effect, np.zeros_like(effect)
