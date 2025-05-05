"""The bases options."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AbstractBasis(ABC):
    """Abstract basis - controlling both the ``bs`` choice and ``xt``."""

    k: int | None

    @abstractmethod
    def __str__(self):
        """The mgcv string representation of the basis."""
        pass


@dataclass(kw_only=True)
class ThinPlateSpline(AbstractBasis):
    """Thin plate regression spline.

    Args:
        shrinkage: Whether to use a modified smoothing penalty to penalise the null
            space.
    """

    k: int | None = None
    shrinkage: bool | None = False

    def __str__(self) -> str:
        """The mgcv string representation of the basis."""
        return "ts" if self.shrinkage else "tp"


@dataclass(kw_only=True)
class CubicSpline(AbstractBasis):
    """Cubic regression regression spline.

    Args:
        cyclic: Whether the spline should be cyclic (ends matching up to second
            derivative). Defaults to False.
        shrinkage: Whether to use a modified smoothing penalty to penalize the null
            space. Defaults to False.
    """

    k: int | None = None
    shrinkage: bool = False
    cyclic: bool = False

    def __post_init__(self):
        if self.cyclic and self.shrinkage:
            raise ValueError("Cannot use both cyclic and shrinkage simultaneously.")

    def __str__(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "cs" if self.shrinkage else "cc" if self.cyclic else "cr"


@dataclass(kw_only=True)
class DuchonSpline(AbstractBasis):  # TODO support passing m?
    """A generalization of thin plate splines."""

    k: int | None = None

    def __str__(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "ds"


@dataclass(kw_only=True)
class SplineOnSphere(AbstractBasis):
    """Two dimensional thin plate spline analogues on a sphere.

    For use with two variables denoting latitude and longitude in degrees.
    """

    k: int | None = None

    def __str__(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "sos"


@dataclass(kw_only=True)
class BSpline(AbstractBasis):
    """B-spline basis with integrated squared derivative penalties."""

    k: int | None = None

    def __str__(self):
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "bs"


@dataclass(kw_only=True)
class PSpline(AbstractBasis):
    """These are P-splines as proposed by Eilers and Marx (1996)."""

    k: int | None = None
    cyclic: bool = False

    def __str__(self) -> str:
        """The 2 letter string passed to the bs argument of mgcv.gam."""
        return "cp" if self.cyclic else "ps"
