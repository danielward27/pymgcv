"""We centralise the imports here to avoid overhead from repeated rpy2 imports."""

from rpy2.robjects.packages import importr

rbase = importr("base")
rmgcv = importr("mgcv")
rstats = importr("stats")
rutils = importr("utils")
