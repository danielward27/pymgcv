from dataclasses import dataclass
from typing import Literal

import numpy as np
from rpy2.robjects.packages import importr

from pymgcv.gam import AbstractGAM
from pymgcv.rpy_utils import is_null, to_py

rbase = importr("base")
rstats = importr("stats")
rmgcv = importr("mgcv")


def qq_uniform(
    gam: AbstractGAM,
    *,
    n: int = 10,
    type: Literal["deviance", "response", "pearson"] = "deviance",
):
    if gam.fit_state is None:
        raise ValueError("GAM has not been fit")

    rgam = gam.fit_state.rgam
    resids = gam.residuals(type)
    fit = rstats.fitted(rgam)
    weights = rstats.weights(rgam, type="prior")
    sigma2 = rgam.rx2["sig2"]

    # TODO we could easily support e.g. gaulss

    if is_null(sigma2):
        sigma2 = rbase.summary(rgam, re_test=False).rx2["dispersion"]

    fit, weights, sigma2 = to_py(fit), to_py(weights), to_py(sigma2)

    if fit.ndim > 1:
        raise NotImplementedError(
            "Families producing matrix outputs are not yet supported for qq_uniform.",
        )
    n_resids = len(resids)
    rng = np.random.default_rng()

    sims = []
    for _ in range(n):
        unif = rng.uniform(size=n_resids)

        qq = gam.family.quantile(
            probs=unif,
            mu=fit,
            wt=weights,
            scale=sigma2,
        )

        res = gam.residuals_from_y_and_fit(
            y=qq,
            fit=fit,
            weights=weights,
            type=type,
        )
        sims.append(np.sort(res))

    sims = np.stack(sims, axis=1)
    theoretical = sims.mean(axis=1)
    resids = np.sort(resids)

    return QQResult(
        theoretical=theoretical,
        residuals=resids,
    )


def qq_simulate(
    gam: AbstractGAM,
    *,
    n=50,
    type: Literal["deviance", "response", "pearson"] = "deviance",
):  # TODO add alpha?
    if gam.fit_state is None:
        raise ValueError("GAM must be fitted before simulating quantiles.")
    if n < 2:
        raise ValueError("n must be at least 2.")

    model = gam.fit_state.rgam
    fit = rstats.fitted(model)

    weights = rstats.weights(model, type="prior")
    sigma2 = model.rx2["sig2"]

    if is_null(sigma2):
        sigma2 = rbase.summary(model, re_test=False).rx2["dispersion"]

    fit, weights, sigma2 = to_py(fit), to_py(weights), to_py(sigma2)

    if fit.ndim > 1:
        raise NotImplementedError(
            "Families producing matrix outputs are not yet supported for qq_simulate.",
        )

    sims = []
    for _ in range(n):
        ysim = gam.family.sample(mu=fit, wt=weights, scale=sigma2)
        res = gam.residuals_from_y_and_fit(
            y=ysim,
            fit=fit,
            weights=weights,
            type=type,
        )
        sim = np.sort(res)
        sims.append(sim)

    sims = np.stack(sims, axis=1)
    fit = to_py(fit)
    n_obs = len(fit)
    theoretical = np.quantile(
        sims,
        q=(np.arange(n_obs) + 0.5) / n_obs,
    )  # TODO implicitly flattens, I think this is correct?
    interval = np.quantile(sims, q=(0.05, 1 - 0.05), axis=1)
    residuals = gam.residuals(type=type)
    if residuals.ndim > 1:
        raise ValueError(
            "Cannot use qq simulate for models with multivariate residuals.",
        )
    residuals = np.sort(residuals)

    return QQResult(
        theoretical=theoretical,
        residuals=residuals,
        interval=(interval[0], interval[1]),
    )


@dataclass
class QQResult:
    theoretical: np.ndarray
    residuals: np.ndarray
    interval: tuple[np.ndarray, np.ndarray] | None = None  # lower, upper
