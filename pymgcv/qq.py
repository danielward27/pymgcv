import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr

from pymgcv.converters import to_py, to_rpy
from pymgcv.gam import AbstractGAM

rbase = importr("base")
rstats = importr("stats")
rmgcv = importr("mgcv")


def _get_family_functions(rgam):
    """Extract and validate family functions needed for Q-Q plot."""
    family = rstats.family(rgam)
    family = rmgcv.fix_family_qf(family)

    dev_resid_fun = family.rx2["residuals"]
    # If NULL likely one of the standard families so
    # use the the dev.resids object from the family
    if rbase.is_null(dev_resid_fun)[0]:
        dev_resid_fun = family.rx2["dev.resids"]

    q_fun = family.rx2["qf"]
    if rbase.is_null(q_fun)[0]:
        family_name = str(family.rx2["family"][0])
        raise NotImplementedError(f"Quantile function not available for family '{family_name}'")

    return dev_resid_fun, q_fun


def qq_uniform(
    gam: AbstractGAM,
    *,
    n: int = 10,
    detrend: bool = False,
):
    if gam.fit_state is None:
        raise ValueError("GAM has not been fit")

    rgam = gam.fit_state.rgam
    dev_resid_fun, q_fun = _get_family_functions(rgam)

    resids = gam.residuals()
    fit = rstats.fitted(rgam)
    weights = rstats.weights(rgam, type="prior")
    sigma2 = rgam.rx2["sig2"]

    if rbase.is_null(sigma2)[0]:
        sigma2 = rbase.summary(rgam, re_test=False).rx2["dispersion"]

    n_resids = len(resids)

    unif = to_rpy((np.arange(1, n_resids + 1) - 0.5) / n_resids)
    sims = np.empty((n_resids, n))
    for i in range(n):
        unif = rbase.sample(unif, n_resids)
        sims[:, i] = qq_uniform_quantiles(
            quantiles=unif,
            quantile_fn=q_fun,
            fit=fit,
            weights=weights,
            sigma2=sigma2,
            dev_resid_fun=dev_resid_fun,
            model=rgam,
        )
    theoretical = sims.mean(axis=1)
    resids = np.sort(resids)

    if detrend:
        resids = resids - theoretical

    return pd.DataFrame(
        {
            "theoretical": theoretical,
            "residuals": resids,
        },
    )


def qq_uniform_quantiles(
    quantiles,
    quantile_fn,
    fit,
    weights,
    sigma2,
    dev_resid_fun,
    model,
):
    ## generate quantiles for uniforms from q_fun
    qq = quantile_fn(quantiles, fit, weights, sigma2)

    res = deviance_residuals(
        qq,
        fit=fit,
        weights=weights,
        dev_resid_fun=dev_resid_fun,
        model=model,
    )
    return np.sort(res)


# TODO do we need to handle NAs?


def deviance_residuals(y, fit, weights, dev_resid_fun, model):
    if "object" in to_py(
        rbase.names(rbase.formals(dev_resid_fun)),
    ):
        # TODO test with cox.ph
        # have to handle families that provide a residuals function, which
        # takes the fitted model as input
        model.rx2["y"] = y
        model.rx2["fitted.values"] = fit
        model.rx2["prior.weights"] = weights
        return dev_resid_fun(model, type="deviance")

    d_resid = dev_resid_fun(y, fit, weights)
    posneg = rbase.attr(d_resid, "sign")  # Not defined for all families
    if rbase.is_null(posneg)[0]:
        posneg = to_rpy(np.sign(to_py(y) - to_py(fit)))
    d_resid = to_py(d_resid)
    posneg = to_py(posneg)
    return np.sqrt(np.maximum(d_resid, 0)) * posneg
