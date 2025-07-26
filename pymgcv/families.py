"""Families supported by pymgcv."""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv.converters import to_py, to_rpy

rstats = importr("stats")
rmgcv = importr("mgcv")

# TODO Some families have multiple links (e.g. gaulss)


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

    def quantile(self, x: np.ndarray) -> np.ndarray:
        """Compute the quantile function."""
        ...

    def deviance_residuals(self, *, y: np.ndarray, fit: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute the deviance residuals."""
        ...



class RStandardFamilyMixin:
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

    def quantile(self, x: np.ndarray) -> np.ndarray:
        """Compute the quantile function."""
        result = rmgcv.fix_family_qf(self.rfamily).rx2["qf"](to_rpy(x))
        assert isinstance(result, np.ndarray)
        return result

    def deviance_residuals(self, *, y: np.ndarray, fit: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute the deviance residuals."""
        d_resid = to_py(self.rfamily.rx2["dev.resids"](to_rpy(y), to_rpy(fit), to_rpy(weights)))
        assert isinstance(d_resid, np.ndarray)
        return d_resid


class Gaussian(RStandardFamilyMixin, FamilyLike):
    def __init__(self, link: Literal["identity", "log", "inverse"] = "identity"):
        self.rfamily = rstats.gaussian(link=link)


class Binomial(RStandardFamilyMixin, FamilyLike):
    def __init__(
        self,
        link: Literal["logit", "probit", "cauchit", "log", "cloglog"] = "logit",
    ):
        """Binomial family with specified link function.

        Args:
            link: The link function for the binomial family.
        """
        self.rfamily = rstats.binomial(link=link)


class Gamma(RStandardFamilyMixin, FamilyLike):
    def __init__(self, link: Literal["inverse", "identity", "log"] = "inverse"):
        """Gamma family with specified link function.

        Args:
            link: The link function for the Gamma family.
        """
        self.rfamily = rstats.Gamma(link=link)


class InverseGaussian(RStandardFamilyMixin, FamilyLike):
    def __init__(
        self,
        link: Literal["1/mu^2", "inverse", "identity", "log"] = "1/mu^2",
    ):
        """Inverse Gaussian family with specified link function.

        Args:
            link: The link function for the inverse Gaussian family.
        """
        self.rfamily = rstats.inverse_gaussian(link=link)


class Poisson(RStandardFamilyMixin, FamilyLike):
    def __init__(self, link: Literal["log", "identity", "sqrt"] = "log"):
        """Poisson family with specified link function.

        Args:
            link: The link function for the Poisson family.
        """
        self.rfamily = rstats.poisson(link=link)


class Quasi(RStandardFamilyMixin, FamilyLike):
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


class QuasiBinomial(RStandardFamilyMixin, FamilyLike):
    def __init__(
        self,
        link: Literal["logit", "probit", "cauchit", "log", "cloglog"] = "logit",
    ):
        """Quasi-binomial family with specified link function.

        Args:
            link: The link function for the quasi-binomial family.
        """
        self.rfamily = rstats.quasibinomial(link=link)


class QuasiPoisson(RStandardFamilyMixin, FamilyLike):
    def __init__(self, link: Literal["log", "identity", "sqrt"] = "log"):
        """Quasi-Poisson family with specified link function.

        Args:
            link: The link function for the quasi-Poisson family.
        """
        self.rfamily = rstats.quasipoisson(link=link)

# TODO implememt!
@dataclass
class MVN(FamilyLike): # TODO check options
    rfamily: ro.ListVector

    def __init__(self, d: int):
        self.rfamily = rmgcv.mvn(d=d)

    def link(self, x: np.ndarray) -> np.ndarray:
         raise NotImplementedError()

    def inverse_link(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def dmu_deta(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def quantile(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def deviance_residuals(self, *, y: np.ndarray, fit: np.ndarray, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

# TODO implememt!
@dataclass
class GauLSS(FamilyLike): # TODO check options
    rfamily: ro.ListVector

    def __init__(self):  # TODO link options
        self.rfamily = rmgcv.gaulss()

    def link(self, x: np.ndarray) -> np.ndarray:
         raise NotImplementedError()

    def inverse_link(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def dmu_deta(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def quantile(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def deviance_residuals(self, *, y: np.ndarray, fit: np.ndarray, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
