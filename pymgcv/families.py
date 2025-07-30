"""Families supported by pymgcv."""

from typing import Literal, Protocol

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.rpy_utils import is_null, to_py, to_rpy

rbase = importr("base")
rstats = importr("stats")
rmgcv = importr("mgcv")

# TODO Some families have multiple links (e.g. gaulss)


# TODO switch to absract base class
class FamilyLike(Protocol):
    rfamily: ro.ListVector

    def link(self, x: np.ndarray) -> np.ndarray:
        """Compute the link function."""
        ...

    def inverse_link(self, x: np.ndarray) -> np.ndarray:
        """Compute the inverse link function."""
        ...

    def dmu_deta(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative dmu/deta of the link function."""
        ...

    def quantile(
        self,
        *,
        probs: float | np.ndarray,
        mu: int | float | np.ndarray,
        wt: int | float | np.ndarray,
        scale: int | float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the quantile function."""
        ...

    def sample(
        self,
        mu: int | float | np.ndarray,
        wt: int | float | np.ndarray,
        scale: int | float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample the family distribution (R family rd method)."""
        ...


class FamilyMixin:
    """Forwards methods from R standard families."""

    rfamily: ro.ListVector

    def link(self, x: np.ndarray) -> np.ndarray:
        result = to_py(self.rfamily.rx2["linkfun"](to_rpy(x)))
        assert isinstance(result, np.ndarray)
        return result

    def inverse_link(self, x: np.ndarray) -> np.ndarray:
        result = to_py(self.rfamily.rx2["linkinv"](to_rpy(x)))
        assert isinstance(result, np.ndarray)
        return result

    def dmu_deta(self, x: np.ndarray) -> np.ndarray:
        result = to_py(self.rfamily.rx2["mu.eta"](to_rpy(x)))
        assert isinstance(result, np.ndarray)
        return result

    def quantile(
        self,
        *,
        probs: float | np.ndarray,
        mu: int | float | np.ndarray,
        wt: int | float | np.ndarray | None = None,
        scale: int | float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the quantile function."""
        q_fun = rmgcv.fix_family_qf(self.rfamily).rx2["qf"]

        if is_null(q_fun):
            raise NotImplementedError(
                f"Quantile function not available for family {self.__class__.__name__}.",
            )
        kwargs = {"mu": mu, "wt": wt, "scale": scale}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        arrays = np.broadcast_arrays(*kwargs.values())
        kwargs = {k: to_rpy(v) for k, v in zip(kwargs.keys(), arrays, strict=True)}
        result = to_py(q_fun(to_rpy(probs), **kwargs))
        assert isinstance(result, np.ndarray)
        return result

    def sample(
        self,
        mu: int | float | np.ndarray,
        wt: int | float | np.ndarray | None = None,
        scale: int | float | np.ndarray | None = None,
    ):
        """Sample the family distributions (R family rd method)."""
        sample_fn = rmgcv.fix_family_rd(self.rfamily).rx2["rd"]
        if is_null(sample_fn):
            raise NotImplementedError(
                f"Sample function not available for family {self.__name__}.",
            )

        kwargs = {"mu": mu, "wt": wt, "scale": scale}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        arrays = np.broadcast_arrays(*kwargs.values())
        kwargs = {k: to_rpy(v) for k, v in zip(kwargs.keys(), arrays, strict=True)}
        return to_py(sample_fn(**kwargs))


class Gaussian(FamilyMixin, FamilyLike):
    def __init__(self, link: Literal["identity", "log", "inverse"] = "identity"):
        self.rfamily = rstats.gaussian(link=link)


class Binomial(FamilyMixin, FamilyLike):
    def __init__(
        self,
        link: Literal["logit", "probit", "cauchit", "log", "cloglog"] = "logit",
    ):
        """Binomial family with specified link function.

        Args:
            link: The link function for the binomial family.
        """
        self.rfamily = rstats.binomial(link=link)


class Gamma(FamilyMixin, FamilyLike):
    def __init__(self, link: Literal["inverse", "identity", "log"] = "inverse"):
        """Gamma family with specified link function.

        Args:
            link: The link function for the Gamma family.
        """
        self.rfamily = rstats.Gamma(link=link)


class InverseGaussian(FamilyMixin, FamilyLike):
    def __init__(
        self,
        link: Literal["1/mu^2", "inverse", "identity", "log"] = "1/mu^2",
    ):
        """Inverse Gaussian family with specified link function.

        Args:
            link: The link function for the inverse Gaussian family.
        """
        self.rfamily = rstats.inverse_gaussian(link=link)


class Poisson(FamilyMixin, FamilyLike):
    def __init__(self, link: Literal["log", "identity", "sqrt"] = "log"):
        """Poisson family with specified link function.

        Args:
            link: The link function for the Poisson family.
        """
        self.rfamily = rstats.poisson(link=link)


class Quasi(FamilyMixin, FamilyLike):
    def __init__(
        self,
        link: Literal[
            "logit",
            "probit",
            "cloglog",
            "identity",
            "inverse",
            "log",
            "1/mu^2",
            "sqrt",
        ] = "identity",
        variance: Literal["constant", "mu(1-mu)", "mu", "mu^2", "mu^3"] = "constant",
    ):
        """Quasi family with specified link and variance functions.

        Args:
            link: The link function for the quasi family. Valid options are:
            variance: The variance function for the quasi family.
        """
        self.rfamily = rstats.quasi(link=link, variance=variance)


class QuasiBinomial(FamilyMixin, FamilyLike):
    def __init__(
        self,
        link: Literal["logit", "probit", "cauchit", "log", "cloglog"] = "logit",
    ):
        """Quasi-binomial family with specified link function.

        Args:
            link: The link function for the quasi-binomial family.
        """
        self.rfamily = rstats.quasibinomial(link=link)


class QuasiPoisson(FamilyMixin, FamilyLike):
    def __init__(self, link: Literal["log", "identity", "sqrt"] = "log"):
        """Quasi-Poisson family with specified link function.

        Args:
            link: The link function for the quasi-Poisson family.
        """
        self.rfamily = rstats.quasipoisson(link=link)


class Tweedie(FamilyMixin, FamilyLike):
    def __init__(
        self,
        p: float | int = 1,
        link: Literal["log", "identity", "inverse", "sqrt"] | int | float = 0,
    ):
        r"""Tweedie family.

        Args:
            p: The variance of an observation is proportional to its mean to the power p. p must
                be greater than 1 and less than or equal to 2. 1 would be Poisson, 2 is gamma.
            link: If a float/int, treated as $ \lambda $ in a link function based on
                $ \eta = \mu^ \lambda $, meaning 0 gives the log link and 1 gives the
                identity link (i.e. R stats package `power`). Can also be one of "log",
                "identity", "inverse", "sqrt".
        """
        if isinstance(link, int | float):
            link = rstats.power(link)
        self.rfamily = rmgcv.Tweedie(p, link)


class NegBin(FamilyMixin, FamilyLike):
    def __init__(self):
        pass


class Betar:
    def __init__(self):
        pass


class CNorm:
    def __init__(self):
        pass


class CLog:
    def __init__(self):
        pass


class CPois:
    def __init__(self):
        pass


class NB:
    def __init__(self):
        pass


class OCat:
    def __init__(self):
        pass


class Scat(FamilyMixin, FamilyLike):
    def __init__(self):
        self.rfamily = rmgcv.scat(link="identity")


class Tw:
    def __init__(self):
        pass


class ZIP:
    def __init__(self):
        pass


class CoxPH:
    def __init__(self):
        pass


class GammaLS:
    def __init__(self):
        pass


class GauLSS(FamilyMixin, FamilyLike):  # TODO check options
    rfamily: ro.ListVector

    def __init__(
        self,
        link: Literal["identity", "inverse", "log", "sqrt"] = "identity",
    ):
        self.rfamily = rmgcv.gaulss(link=ro.StrVector([link, "logb"]))


class GevLSS:
    def __init__(self):
        pass


class GumbLS:
    def __init__(self):
        pass


class Multinom:
    def __init__(self):
        pass


class MVN(FamilyMixin, FamilyLike):  # TODO probably needs special casing
    rfamily: ro.ListVector

    def __init__(self, d: int):
        self.rfamily = rmgcv.mvn(d=d)


class Shash:
    def __init__(self):
        pass


class TwLSS:
    def __init__(self):
        pass


class ZipLSS:
    def __init__(self):
        pass
