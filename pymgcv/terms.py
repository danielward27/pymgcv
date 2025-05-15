"""Defines the different types of terms available in pymgcv."""

from collections.abc import Callable, Sequence

# TODO not supported sp, pc.
# xt not needed as will be handled with basis.
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pymgcv.bases import BasisProtocol


@runtime_checkable
class TermLike(Protocol):
    """Protocol defining interface for terms of a GAM model.

    Attributes:
        varnames: The name or names of the variables in the term.
        simple_string: A simplified representation of the term. This should match the form of the term after printing
            the summary of a GAM model.


    """
    varnames: tuple[str, ...]
    simple_string: str

    def __str__(self) -> str:
        """The string representation of the term which will be passed to mgcv."""
        ...


@dataclass
class Linear(TermLike):  # TODO do we need a seperate class for a variable factor?
    """A linear term in the model (i.e. no basis expansion).
    
    Args:
        name: The name of the variable to include as a linear term.

    """
    varnames: tuple[str]
    simple_string: str

    def __init__(self, name: str, /):
        self.varnames = (name, )
        self.simple_string = self.varnames[0]

    def __str__(self) -> str:
        return self.simple_string
    

@dataclass
class Interaction(TermLike):
    """An interaction term in the model."""
    varnames: tuple[str, ...]
    simple_string: str

    def __init__(self, a: str, b: str):
        self.varnames = (a, b)
        self.simple_string = f"{a}:{b}"

    def __str__(self) -> str:
        return self.simple_string


@dataclass
class Smooth(TermLike):
    varnames: tuple[str, ...]
    k: int | None
    bs: BasisProtocol | None
    m: int | None
    by: str | None
    id: str | None
    fx: bool
    simple_string: str

    def __init__(
        self,
        *varnames: str,
        k: int | None = None,
        bs: BasisProtocol | None = None,
        m: int | None = None,
        by: str | None = None,
        id: str | None = None,
        fx: bool = False,
    ):
        self.varnames = varnames
        self.k=k
        self.bs =bs
        self.m =m
        self.by =by
        self.id =id
        self.fx =fx
        self.simple_string = f"s({','.join(self.varnames)})"

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


def _sequence_to_rvec_str(seq: Sequence, converter: Callable[[Any], str]=str):
    return f"c({','.join(converter(x) for x in seq)})"


@dataclass
class TensorSmooth(TermLike):
    varnames: tuple[str, ...]
    k: tuple[int, ...] | None
    bs: tuple[BasisProtocol, ...] | None
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
        bs: Sequence[BasisProtocol] | None = None,
        d: Sequence[int] | None = None,
        m: Sequence[int] | None = None,
        by: str | None = None,
        id: str | None = None,
        fx: bool = False,
        np: bool = True,
        interaction_only: bool = False,
    ):
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

        prefix = "ti" if self.interaction_only else "te"
        self.simple_string = f"{prefix}({','.join(self.varnames)})"

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
