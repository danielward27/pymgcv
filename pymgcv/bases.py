"""The bases options."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from rpy2.robjects import ListVector


@runtime_checkable
class BasisLike(Protocol):
    """Basis protocol class, defining the interface for configuring basis functions."""

    def __str__(self) -> str:
        """The mgcv string representation of the basis."""
        ...

    def get_xt(self) -> ListVector | None:
        """Get an robject to pass as xt."""
        ...


class RandomEffect(BasisLike):
    """Random effect."""

    def __str__(self) -> str:
        """The mgcv string representation of the basis."""
        return "re"

    def get_xt(self) -> ListVector | None:
        return None


@dataclass(kw_only=True)
class ThinPlateSpline(BasisLike):
    """Thin plate regression spline.

    Args:
        shrinkage: Whether to use a modified smoothing penalty to penalise the null
            space.
    """

    shrinkage: bool | None = False

    def __str__(self) -> str:
        """The mgcv string representation of the basis."""
        return "ts" if self.shrinkage else "tp"

    def get_xt(self):
        return None


@dataclass(kw_only=True)
class CubicSpline(BasisLike):
    """Cubic regression regression spline.

    Args:
        cyclic: Whether the spline should be cyclic (ends matching up to second
            derivative). Defaults to False.
        shrinkage: Whether to use a modified smoothing penalty to penalize the null
            space. Defaults to False.
    """

    shrinkage: bool = False
    cyclic: bool = False

    def __post_init__(self):
        """Checks valid initialization."""
        if self.cyclic and self.shrinkage:
            raise ValueError("Cannot use both cyclic and shrinkage simultaneously.")

    def __str__(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "cs" if self.shrinkage else "cc" if self.cyclic else "cr"

    def get_xt(self):
        return None


@dataclass(kw_only=True)
class DuchonSpline(BasisLike):  # TODO support passing m?
    """A generalization of thin plate splines."""

    def __str__(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "ds"

    def get_xt(self):
        return None


@dataclass(kw_only=True)
class SplineOnSphere(BasisLike):
    """Two dimensional thin plate spline analogues on a sphere.

    For use with two variables denoting latitude and longitude in degrees.
    """

    def __str__(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "sos"

    def get_xt(self):
        return None


@dataclass(kw_only=True)
class BSpline(BasisLike):
    """B-spline basis with integrated squared derivative penalties."""

    def __str__(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "bs"

    def get_xt(self):
        return None


@dataclass(kw_only=True)
class PSpline(BasisLike):
    """These are P-splines as proposed by Eilers and Marx (1996)."""

    cyclic: bool = False

    def __str__(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "cp" if self.cyclic else "ps"

    def get_xt(self):
        return None


@dataclass(kw_only=True)
class MarkovRandomField(BasisLike):
    polys: list[np.ndarray]
    # TODO support xt

    def __str__(self) -> str:
        return "mrf"

    def get_xt(self):
        return NotImplementedError()
