"""Families supported by pymgcv."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import rpy2.robjects as ro

from pymgcv.rlibs import rbase, rmgcv, rstats
from pymgcv.rpy_utils import is_null, to_py, to_rpy

# For qq_simulate we need sample, and for qq_cdf we need cdf implemented.


class AbstractFamily(ABC):
    """Provides default implmentations for distribution methods.

    This applies mgcv `fix.family.qf` for the quantile function, and
    `fix.family.rd` for the sampling function.
    """

    rfamily: ro.ListVector

    def link(self, x: np.ndarray) -> np.ndarray:
        """Compute the link function."""
        result = to_py(self.rfamily.rx2["linkfun"](to_rpy(x)))
        assert isinstance(result, np.ndarray)
        return result

    def inverse_link(self, x: np.ndarray) -> np.ndarray:
        """Compute the inverse link function."""
        result = to_py(self.rfamily.rx2["linkinv"](to_rpy(x)))
        assert isinstance(result, np.ndarray)
        return result

    def dmu_deta(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative dmu/deta of the link function."""
        result = to_py(self.rfamily.rx2["mu.eta"](to_rpy(x)))
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
        # R won't broadcast length 1 arrays
        kwargs = {
            k: to_rpy(v) if v.size != 1 else v.item()
            for k, v in kwargs.items()
            if v is not None
        }
        return to_py(sample_fn(**kwargs))


class SupportsCDF(ABC):
    """Mixin for families supporting cumulative distribution functions."""

    @abstractmethod
    def cdf(
        self,
        x: np.ndarray,
        *,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        """Cumulative distribution function."""
        pass


@dataclass
class Gaussian(AbstractFamily, SupportsCDF):
    """Gaussian family with specified link function.

    Args:
        link: The link function.
    """

    def __init__(self, link: Literal["identity", "log", "inverse"] = "identity"):
        self.rfamily = rstats.gaussian(link=link)

    def cdf(
        self,
        x: np.ndarray,
        *,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        """Gaussian CDF."""
        sd = np.sqrt(scale / wt)
        x, mu, sd = (to_rpy(arr) for arr in np.broadcast_arrays(x, mu, sd))
        return to_py(rstats.pnorm(x, mean=mu, sd=sd))


# TODO another case where matrix inputs need to be supported.
# TODO document response form
@dataclass
class Binomial(AbstractFamily, SupportsCDF):
    """Binomial family with specified link function.

    Args:
        link: The link function. "logit", "probit" and "cauchit", correspond to
            logistic, normal and Cauchy CDFs respectively. "cloglog" is the
            complementary log-log.
    """

    def __init__(
        self,
        link: Literal["logit", "probit", "cauchit", "log", "cloglog"] = "logit",
    ):
        self.rfamily = rstats.binomial(link=link)

    def cdf(
        self,
        x: np.ndarray,
        *,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: np.ndarray,
    ):
        """Binomial CDF, scale is ignored."""
        x, mu, wt = (to_rpy(arr) for arr in np.broadcast_arrays(x, mu, wt))
        return to_py(rstats.pbinom(x * (wt + rbase.as_numeric(wt == 0)), wt, mu))


class Gamma(AbstractFamily, SupportsCDF):
    """Gamma family with specified link function.

    Args:
        link: The link function for the Gamma family.
    """

    def __init__(self, link: Literal["inverse", "identity", "log"] = "inverse"):
        self.rfamily = rstats.Gamma(link=link)

    def cdf(
        self,
        x: np.ndarray,
        *,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: np.ndarray,
    ):
        """Gamma CDF, wt is ignored."""
        x, mu, scale = [to_rpy(arr) for arr in np.broadcast_arrays(x, mu, scale)]
        return to_py(
            rstats.pgamma(to_rpy(x), shape=to_rpy(1 / scale), scale=to_rpy(mu * scale)),
        )


class InverseGaussian(AbstractFamily):
    """Inverse Gaussian family with specified link function.

    Args:
        link: The link function for the inverse Gaussian family.
    """

    def __init__(
        self,
        link: Literal["1/mu^2", "inverse", "identity", "log"] = "1/mu^2",
    ):
        self.rfamily = rstats.inverse_gaussian(link=link)


class Poisson(AbstractFamily, SupportsCDF):
    """Poisson family with specified link function.

    Args:
        link: The link function for the Poisson family.
    """

    def __init__(self, link: Literal["log", "identity", "sqrt"] = "log"):
        self.rfamily = rstats.poisson(link=link)

    def cdf(
        self,
        x: np.ndarray,
        *,
        mu: np.ndarray,
        wt: np.ndarray | None = None,
        scale: np.ndarray | None = None,
    ):
        """Cumulative distribution function."""
        return to_py(rstats.ppois(to_rpy(x), to_rpy(mu)))


# TODO These won't support sampling
class Quasi(AbstractFamily):
    """Quasi family with specified link and variance functions.

    Args:
        link: The link function for the quasi family. Valid options are:
        variance: The variance function for the quasi family.
    """

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
        self.rfamily = rstats.quasi(link=link, variance=variance)


class QuasiBinomial(AbstractFamily):
    """Quasi-binomial family with specified link function.

    Args:
        link: The link function for the quasi-binomial family.
    """

    def __init__(
        self,
        link: Literal["logit", "probit", "cauchit", "log", "cloglog"] = "logit",
    ):
        self.rfamily = rstats.quasibinomial(link=link)


class QuasiPoisson(AbstractFamily):
    """Quasi-Poisson family with specified link function.

    Args:
        link: The link function for the quasi-Poisson family.
    """

    def __init__(self, link: Literal["log", "identity", "sqrt"] = "log"):
        self.rfamily = rstats.quasipoisson(link=link)


class Betar(AbstractFamily):
    def __init__(self):
        raise NotImplementedError()


class CNorm(AbstractFamily):
    def __init__(self):
        raise NotImplementedError()


class CLog(AbstractFamily):
    def __init__(self):
        raise NotImplementedError()


class CPois(AbstractFamily):
    def __init__(self):
        raise NotImplementedError()


class NegBin(AbstractFamily):
    r"""Negative binomial family.

    Args:
        theta: The positive parameter such that
            $\text{var}(y) = \mu + \mu^2/\theta$, where $\mu = \mathbb{E}[y]$.
        link: The link function to use.
        theta_fixed: Whether to treat theta as fixed or estimated. If estimated,
            then theta is the starting value.
    """

    def __init__(
        self,
        theta: float | int | None = None,
        link: Literal["log", "identity", "sqrt"] = "log",
        *,
        theta_fixed: bool = False,
    ):
        # For now this just uses nb family (not negbin)
        if theta_fixed and theta is None:
            raise ValueError("Theta must be specified if fixed.")

        if theta is not None:
            theta = theta if theta_fixed else -theta  # mgcv convention

        self.rfamily = rmgcv.nb(theta=ro.NULL if theta is None else theta, link=link)


class OCat(AbstractFamily):
    """Ordered categorical family.

    For performing regression with ordered categorical data.

    Args:
        num_categories: The number of categories.
    """

    def __init__(self, num_categories: int):
        self.rfamily = rmgcv.ocat(R=num_categories)


class Scat(AbstractFamily):
    r"""Scaled t family for heavy tailed data.

    Args:
        link: The link function to use.
        min_df: The minimum degrees of freedom. Must be >2 to avoid infinite
            response variance.
        theta: The parameters to be estimated $\nu = b + \exp(\theta_1)$
            (where $b$ is `min_df`) and $\sigma = \exp(\theta_2)$. If supplied
            and both positive, then taken to be fixed values of $\nu$ and
            $\sigma$. If any negative, then absolute values taken as starting
            values.
        theta_fixed: If theta is provided, controls whether to treat theta as fixed
            or estimated. If estimated, then theta is the starting value.
    """

    def __init__(
        self,
        link: Literal["identity", "log", "inverse"] = "identity",
        min_df: float | int = 3,
        theta: np.ndarray | None = None,
        *,
        theta_fixed: bool = False,
    ):
        if theta is not None and not theta_fixed:
            theta = -theta  # mgcv convention.
        self.rfamily = rmgcv.scat(link=link, min_df=min_df)


class Tweedie(AbstractFamily):
    r"""Tweedie family with fixed power.

    Args:
        p: The variance of an observation is proportional to its mean to the power p. p must
            be greater than 1 and less than or equal to 2. 1 would be Poisson, 2 is gamma.
        link: If a float/int, treated as $\lambda$ in a link function based on
            $\eta = \mu^ \lambda$, meaning 0 gives the log link and 1 gives the
            identity link (i.e. R stats package `power`). Can also be one of "log",
            "identity", "inverse", "sqrt".
    """

    def __init__(
        self,
        p: float | int = 1,
        link: Literal["log", "identity", "inverse", "sqrt"] | int | float = 0,
    ):
        if isinstance(link, int | float):
            link = rstats.power(link)
        self.rfamily = rmgcv.Tweedie(p, link)


class Tw(AbstractFamily):
    r"""Tweedie family with estimated power.

    Restricted to variance function powers between 1 and 2.

    Args:
        link: The link function to use.
        a: The lower bound of the power parameter for optimization.
        b: The upper bound of the power parameter for optimization.
        theta: Related to the Tweedie power parameter by
            $p=(a+b \exp(\theta))/(1+\exp(\theta))$. If this is supplied as a positive
            value then it is taken as the fixed value for p. If it is a negative values
            then its absolute value is taken as the initial value for p.
        theta_fixed: If theta is provided, controls whether to treat theta as fixed
            or estimated. If estimated, then theta is the starting value.
    """

    def __init__(
        self,
        link: Literal["log", "identity", "inverse", "sqrt"] = "log",
        a: float = 1.01,
        b: float = 1.99,
        theta: float | int | None = None,
        *,
        theta_fixed: bool = False,
    ):
        if theta_fixed and theta is None:
            raise ValueError("Theta must be specified if fixed.")

        if theta is not None and not theta_fixed:
            theta = -theta  # mgcv convention.

        self.rfamily = rmgcv.tw(
            theta=ro.NULL if theta is None else theta,
            link=link,
            a=a,
            b=b,
        )


class ZIP(AbstractFamily):
    r"""Zero-inflated Poisson family.

    The probability of a zero count is given by $1-p$, whereas the probability of
    count $y>0$ is given by the truncated Poisson probability function
    $p\mu^y/((\exp(\mu)-1)y!)$. The linear predictor gives $\log \mu$, while
    $\eta = \log(-\log(1-p))$ and $\eta = \theta_1 + \{b+\exp(\theta_2)\} \log \mu$.
    The theta parameters are estimated alongside the smoothing parameters. Increasing
    the b parameter from zero can greatly reduce identifiability problems, particularly
    when there are very few non-zero data.

    The fitted values for this model are the log of the Poisson parameter. Use the
    predict function with type=="response" to get the predicted expected response. Note
    that the theta parameters reported in model summaries are
    $\theta_1 and b + \exp(\theta_2)$.

    !!! warning

        These models should be subject to very careful checking, especially if fitting
        has not converged. It is quite easy to set up models with identifiability
        problems, particularly if the data are not really zero inflated, but simply have
        many zeroes because the mean is very low in some parts of the covariate space.

    Args:
        b: A non-negative constant, specifying the minimum dependence of the zero
            inflation rate on the linear predictor.
        theta: The 2 parameters controlling the slope and intercept of the linear
            transform of the mean controlling the zero inflation rate. If supplied then
            treated as fixed parameters (\theta_1 and \theta_2), otherwise estimated.
    """

    def __init__(
        self,
        b: int | float = 0,
        theta: tuple[int | float, int | float] | None = None,
    ):
        if theta is not None:
            theta = np.asarray(theta)  # type: ignore
        self.rfamily = rmgcv.ziP(theta=ro.NULL if theta is None else theta, b=b)


# TODO support stratification? There is a lot of small details missing in the docs.
# TODO cox.pht
class CoxPH(AbstractFamily):
    """Additive Cox Proportional Hazard Model.

    Cox Proportional Hazards model with Peto's correction for ties, optional
    stratification, and estimation by penalized partial likelihood maximization, for use
    with [`GAM`][pymgcv.gam.GAM]. In the model formula, event time is the response.

    Under stratification the response has two columns: time and a numeric index for
    stratum. The weights vector provides the censoring information (0 for censoring, 1
    for event). CoxPH deals with the case in which each subject has one event/censoring
    time and one row of covariate values.
    """

    def __init__(self):
        self.rfamily = rmgcv.cov_ph()


class GammaLS(AbstractFamily):
    r"""Gamma location-scale model family.

    The log of the mean, $\mu$, and the log of the scale parameter, $\phi$ can depend on
    additive smooth predictors (i.e. using two formulae).

    Args:
        min_log_scale: The minimum value for the log scales parameter.
    """

    def __init__(
        self,
        min_log_scale: float | int = -7,
    ):
        self.family = rmgcv.gammals(b=min_log_scale)


# TODO when e.g. qq plotting finalized check works correctly with GauLSS
class GauLSS(AbstractFamily):
    r"""Gaussian location-scale model family for GAMs.

    Models both the mean $\mu$ and standard deviation $\sigma$ of a Gaussian
    response. The standard deviation uses a "logb" link, i.e.
    $\eta = \log(\sigma - b)$ to avoid singularities near zero.

    Only compatible with [`GAM`][pymgcv.gam.GAM], to which two predictors
    must be specified, for the response variable and the scale respectively.

    - Predictions with `type="response"` returns columns `[mu, 1/sigma]`
    - Predictions with `type="link"` returns columns `[eta_mu, log(sigma - b)]`
    - Plots use the `log(sigma - b)` scale.

    Args:
        link: The link function to use for $\mu$.
        min_std: Minimum standard deviation $b$, for the "logb" link.
    """

    rfamily: ro.ListVector

    def __init__(
        self,
        link: Literal["identity", "inverse", "log", "sqrt"] = "identity",
        min_std: float = 0.01,
    ):
        self.rfamily = rmgcv.gaulss(link=ro.StrVector([link, "logb"]), b=min_std)


class GevLSS(AbstractFamily):
    r"""Generalized extreme value location, scale and shape family.

    Requires three predictors, one for the location, log scale and the shape.

    Uses the p.d.f. $t(y)^{\xi+1} e^{-t(y)} / \sigma$, where:
    $t(x) = [1 + \xi(y-\mu)/\sigma]^{-1/\xi}$ if $\xi \neq 0$
    and $\exp[-(y-\mu)/\sigma]$ otherwise.
    """

    def __init__(self, link: Literal["identity", "log"]):
        self.rfamily = rmgcv.gevlss(link=link)


class GumbLS(AbstractFamily):
    def __init__(self):
        raise NotImplementedError()


class Multinom(AbstractFamily):
    def __init__(self):
        raise NotImplementedError()


class MVN(AbstractFamily):
    """Multivariate normal family.

    Args:
        d: The dimension of the distribution.
    """

    rfamily: ro.ListVector

    def __init__(self, d: int):
        self.rfamily = rmgcv.mvn(d=d)


class Shash(AbstractFamily):
    def __init__(self):
        raise NotImplementedError()


class TwLSS(AbstractFamily):
    def __init__(self):
        raise NotImplementedError()


class ZipLSS(AbstractFamily):
    def __init__(self):
        raise NotImplementedError()
