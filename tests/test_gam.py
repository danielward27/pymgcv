import numpy as np
import rpy2.robjects as ro

from pymgcv import smooth, tensor_smooth
from pymgcv.converters import to_py
from pymgcv.gam import gam, variables_to_formula

mgcv = ro.packages.importr("mgcv")  # type: ignore


# TODO a way to check result against R equivilent model?


def test_variables_to_formula():
    assert (
        variables_to_formula(
            dependent="y",
            independent=("x",),
        )
        == "y~x"
    )
    assert (
        variables_to_formula(
            dependent="y",
            independent=("x0", smooth("x1")),
        )
        == "y~x0+s(x1)"
    )


# TODO accept string in smooth?


def test_gam():
    data = to_py(mgcv.gamSim(5, n=200, scale=2))
    test_data = to_py(mgcv.gamSim(5, n=50, scale=2))
    test_data = {k: v.to_numpy() for k, v in test_data.items()}

    g = gam(
        dependent="y",
        independent=("x0", smooth("x1"), smooth("x2"), smooth("x3")),
        data=data,
    )
    assert isinstance(g.summary(), str)

    pred = g.predict(test_data)

    assert pred["fit"].ndim == 1
    assert pred["se_fit"].ndim == 1

    pred = g.predict(test_data, type="terms")
    assert pred["fit"].ndim == 2
    assert pred["se_fit"].ndim == 2


def test_tensor_gam():
    rng = np.random.default_rng(seed=42)
    n = 200
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1) + rng.normal(0, 0.2, n)
    data = {"x0": x0, "x1": x1, "y": y}
    g = gam(
        dependent="y",
        independent=(tensor_smooth("x0", "x1"),),
        data=data,
    )
    pred = g.predict(data)  # TODO ALlow default in data?
    assert pred["fit"].ndim == 1


def test_gam_with_xt():
    # e.g. mrf
    raise NotImplementedError()


def test_multivariate_gam():
    raise NotImplementedError()


def test_gam_with_list_formulas():
    # i.e. gam(list(y0~s(x0)+s(x1),y1~s(x2)+s(x3)),family=mvn(d=2),data=dat)
    raise NotImplementedError()


def test_gam_with_offset():
    raise NotImplementedError()
