# %%

from collections.abc import Callable, Sequence
from functools import partial

from pymgcv.bases import BasisProtocol

# TODO not supported sp, pc.
# xt not needed as will be handled with basis.

from dataclasses import dataclass
from typing import Tuple, Protocol

from typing import Protocol

class SmoothProtocol(Protocol):  # TODO Now we can no longer pass raw strings
    varnames: Tuple[str, ...]

    def __str__(self) -> str: ...


class Linear(Protocol):  # TODO name?
    varnames: Tuple[str, ...]

    def __init__(self, name: str, /):
        self.varnames = (name, )

    def __str__(self) -> str:
        return ','.join(self.varnames)


@dataclass
class Smooth:
    varnames: Tuple[str, ...]
    k: int | None = None
    bs: BasisProtocol | None = None
    m: int | None = None
    by: str | None = None
    id: str | None = None
    fx: bool = False

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


def _sequence_to_rvec_str(seq: Sequence, converter=str):
    return f"c({','.join(converter(x) for x in seq)})"


@dataclass(init=False)
class TensorSmooth:
    varnames: tuple[str, ...]
    k: tuple[int] | None = None
    bs: tuple[BasisProtocol] = None
    d: tuple[int] | None = None
    m: tuple[int] | None = None
    by: str | None = None
    id: str | None = None
    fx: bool | None = False
    np: bool | None = True
    interaction_only: bool = False

    def __init__(
        self,
        *varnames: str,
        k: Sequence[int] | None = None,
        bs: Sequence[BasisProtocol] | None = None,
        d: Sequence[int] | None = None,
        m: Sequence[int | None] = None,
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
            "bs": partial(_sequence_to_rvec_str, converter=lambda bs: f"'{bs}'"),
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
