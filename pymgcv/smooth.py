from dataclasses import dataclass

from pymgcv.smoothing_bases import AbstractBasis, CubicSpline, ThinPlateSpline

# TODO not supported sp, pc.
# xt not needed as handled with basis.
# TODO Does this need to be a class?


@dataclass
class Smooth:
    """Smooth - similar to ``s`` in mgcv.

    bs now replaced with abstract smooth basis.
    """

    varnames: tuple[str, ...]
    bs: AbstractBasis
    m: int | None
    by: str | None
    id: str | None
    fx: bool = False

    def __init__(
        self,
        *varnames: str,
        bs: AbstractBasis | None = None,
        m: int | None = None,
        by: str | None = None,
        id: str | None = None,
        fx: bool = False,
    ):
        self.varnames = varnames
        self.bs = ThinPlateSpline() if bs is None else bs
        self.m = m
        self.by = by
        self.id = id
        self.fx = fx

    def __str__(self):
        """Returns the mgcv smooth term as a string."""
        provided = {
            "k": -1 if self.bs.k is None else self.bs.k,
            "bs": f"'{self.bs.bs_str}'",
            "m": "NA" if self.m is None else self.m,
            "by": "NA" if self.by is None else self.by,
            "id": "NULL" if self.id is None else f"'{self.id}'",
            "fx": str(self.fx).upper(),
        }

        defaults = {
            "k": -1,
            "bs": "'tp'",
            "m": "NA",
            "by": "NA",
            "id": "NULL",
            "fx": "FALSE",
        }
        keep = {k: v for k, v in provided.items() if defaults[k] != v}
        smooth_string = f"s({",".join(self.varnames)}"
        for key, val in keep.items():
            smooth_string += f",{key}={val}"
        return smooth_string + ")"


# TODO some options not supported
# Passing lists to bs?
@dataclass
class TensorSmooth:
    """Tensor smooths (te and ti in mgcv)."""

    varnames: tuple[str, ...]
    bs: AbstractBasis
    d: int | None
    m: int | None
    by: str | None
    id: str | None
    fx: bool
    np: bool
    interaction_only: bool

    def __init__(
        self,
        *varnames: str,
        bs: AbstractBasis | None = None,
        d: int | None = None,
        m: int | None = None,
        by: str | None = None,
        id: str | None = None,
        fx: bool = False,
        np: bool = True,
        interaction_only: bool = False,
    ):
        self.varnames = varnames
        self.bs = CubicSpline() if bs is None else bs
        self.d = d
        self.m = m
        self.by = by
        self.id = id
        self.fx = fx
        self.np = np
        self.interaction_only = interaction_only

    def __str__(self):
        """Returns the mgcv smooth term as a string."""
        provided = {
            "k": -1 if self.bs.k is None else self.bs.k,
            "bs": f"'{self.bs.bs_str}'",
            "d": "NA" if self.d is None else self.d,
            "m": "NA" if self.m is None else self.m,
            "by": "NA" if self.by is None else self.by,
            "id": "NULL" if self.id is None else f"'{self.id}'",
            "fx": str(self.fx).upper(),
            "np": str(self.np).upper(),
        }

        defaults = {
            "k": -1,
            "bs": "'cr'",
            "d": "NA",
            "m": "NA",
            "by": "NA",
            "id": "NULL",
            "fx": "FALSE",
            "np": "TRUE",
        }
        keep = {k: v for k, v in provided.items() if defaults[k] != v}

        bs = "ti" if self.interaction_only else "te"
        smooth_string = f"{bs}({",".join(self.varnames)}"
        for key, val in keep.items():
            smooth_string += f",{key}={val}"
        return smooth_string + ")"
