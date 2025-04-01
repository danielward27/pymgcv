"First module."

from rpy2 import robjects
from rpy2.robjects.packages import importr


def first_func() -> bool:
    return True


def basic_gam() -> robjects.ListVector:
    mgcv = importr("mgcv")
    data = mgcv.gamSim(5, n=200, scale=2)
    return mgcv.gam(robjects.Formula("y ~ x0 + s(x1) + s(x2) + s(x3)"), data=data)
