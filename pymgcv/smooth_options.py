from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AbstractSmooth(ABC):

    @property
    @abstractmethod
    def r_bs_str(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        pass


@dataclass(kw_only=True)
class ThinPlateSpline(AbstractSmooth):
    """Thin plate regression spline.

    Args:
        shrinkage: Whether to use a modified smoothing penalty to penalise the null
    """

    shrinkage: bool = False

    @property
    def r_bs_str(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "ts" if self.shrinkage else "tp"


@dataclass
class DuchonSpline(AbstractSmooth):  # TODO support passing m?
    """A generalization of thin plate splines."""

    @property
    def r_bs_str(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "ds"


@dataclass(kw_only=True)
class CubicSpline(AbstractSmooth):
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
        if self.cyclic and self.shrinkage:
            raise ValueError("Cannot use both cyclic and shrinkage simultaneously.")

    @property
    def r_bs_str(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "cs" if self.shrinkage else "cc" if self.cyclic else "cr"


@dataclass
class SplineOnSphere(AbstractSmooth):
    """Two dimensional thin plate spline analogues on a sphere.

    For use with two variables denoting latitude and longitude in degrees.
    """

    @property
    def r_bs_str(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "sos"


@dataclass
class BSpline(AbstractSmooth):
    """B-spline basis with integrated squared derivative penalties."""

    @property
    def r_bs_str(self):
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "bs"


@dataclass(kw_only=True)
class PSpline(AbstractSmooth):
    cyclic: bool = False

    def __init__(self, *, cyclic: bool = False):
        """These are P-splines as proposed by Eilers and Marx (1996)."""
        pass

    @property
    def r_bs_str(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "cp" if self.cyclic else "ps"
