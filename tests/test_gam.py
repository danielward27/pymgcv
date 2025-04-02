import rpy2.robjects as ro

from pymgcv.converters import to_py
from pymgcv.gam import gam, variables_to_formula
from pymgcv.smooths import Smooth

mgcv = ro.packages.importr("mgcv")


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
            independent=("x0", Smooth("x1")),
        )
        == "y~x0+s(x1,k=-1,bs='tp')"
    )


def test_gam():

    data = to_py(mgcv.gamSim(5, n=200, scale=2))

    g = gam(
        dependent="y",
        independent=("x0", Smooth("x1"), Smooth("x2"), Smooth("x3")),
        data=data,
    )
    assert g is not None  # TODO update when gam result finished.
