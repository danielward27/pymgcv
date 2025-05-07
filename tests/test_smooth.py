import pytest

from pymgcv import smooth, tensor_smooth
from pymgcv.bases import CubicSpline, SplineOnSphere

test_cases = [
    (smooth, dict(vars=["a"]), "s(a)"),
    (smooth, dict(vars=["a", "b"], m=3), "s(a,b,m=3)"),
    (
        smooth,
        dict(
            vars=["a", "b"],
            k=10,
            bs=CubicSpline(cyclic=True),
            m=5,
            by="var",
            id="2",
            fx=True,
        ),
        "s(a,b,k=10,bs='cc',m=5,by=var,id='2',fx=TRUE)",
    ),
    (tensor_smooth, dict(vars=["a", "b"]), "te(a,b)"),
    (tensor_smooth, dict(vars=["a", "b"], interaction_only=True), "ti(a,b)"),
    (
        tensor_smooth,
        dict(
            vars=["long", "lat"],
            bs=[SplineOnSphere(), SplineOnSphere()],
            m=[2, 3],
            d=[5, 3],
            by="var",
            np=False,
            id="my_id",
            fx=True,
            interaction_only=True,
        ),
        "ti(long,lat,bs=c('sos','sos'),d=c(5,3),m=c(2,3),by=var,id='my_id',fx=TRUE,np=FALSE)",
    ),
    (
        tensor_smooth,
        dict(vars=["x", "y"], bs=[CubicSpline(), CubicSpline()]),
        "te(x,y,bs=c('cr','cr'))",
    ),
]


@pytest.mark.parametrize(("smooth_type", "kwargs", "expected"), test_cases)
def test_smooth_to_str(smooth_type, kwargs, expected):
    smooth_str = smooth_type(*kwargs.pop("vars"), **kwargs)
    assert smooth_str == expected
