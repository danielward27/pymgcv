"""Defines the different types of terms available in pymgcv."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pymgcv.bases import BasisLike

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
        simple_string: A simplified representation of the term. This should
            match the form of the term after print in the summary of a GAM model.
            We use this for plot labels and for excluding terms from the model
            in mgcv.
        by: The name of a by variable or None if not present.
    """

    varnames: tuple[str, ...]
    simple_string: str
    by: str | None

    def __str__(self) -> str:
        """The string representation of the term which will be passed to mgcv."""
        ...


@dataclass
class Linear(TermLike):
    """A linear term (factor or numeric term) i.e. no basis expansion.

    Args:
        name: The name of the variable to include as a linear term.

    """

    varnames: tuple[str]
    simple_string: str
    by: str | None

    def __init__(self, name: str):
        self.varnames = (name,)
        self.simple_string = self.varnames[0]
        self.by = None

    def __str__(self) -> str:
        return self.simple_string


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
    simple_string: str
    by: str | None

    def __init__(self, *varnames: str):
        """Initialize an interaction term.

        Args:
            *varnames: The names of the variables to include in the interaction.
        """
        self.varnames = tuple(varnames)
        self.simple_string = ":".join(self.varnames)
        self.by = None

    def __str__(self) -> str:
        return self.simple_string


@dataclass
class Smooth(TermLike):
    """Create a smooth term.

    Note that terms like ``Smooth("x1", "x2")`` are isotropic. This means for
    example variables should be of similar scales. For scale invariant smoothing
    of multiple variables, see ``TensorSmooth``.
    """  # TODO properly reference TensorSmooth

    varnames: tuple[str, ...]
    k: int | None
    bs: BasisLike | None
    m: int | None
    by: str | None
    id: str | None
    fx: bool
    simple_string: str

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
        simple_string = f"s({','.join(self.varnames)})"

        if self.by is not None:
            simple_string += f":{self.by}"

        self.simple_string = simple_string

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


def _sequence_to_rvec_str(seq: Sequence, converter: Callable[[Any], str] = str):
    return f"c({','.join(converter(x) for x in seq)})"


# TODO, tensnor smooth d=c(2,1), common, in which case all the sequences should be length 2.
# Worth testing this case.
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
    simple_string: str

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

        simple_string = "ti(" if self.interaction_only else "te("
        simple_string += ",".join(self.varnames) + ")"
        if self.by is not None:
            simple_string += ":" + self.by
        self.simple_string = simple_string

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
