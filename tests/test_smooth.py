# def test_smooth_str_conversion():
#     s = Smooth("x", "y")
#     assert str(s) == "s(x,y,k=-1,bs='tp')"
import pytest

from pymgcv import Smooth, TensorSmooth
from pymgcv.smoothing_bases import CubicSpline, SplineOnSphere

test_cases = [
    (Smooth(("a",)), "s(a)"),
    (Smooth(("a", "b"), m=3), "s(a,b,m=3)"),
    (
        Smooth(
            ("a", "b"),
            bs=CubicSpline(cyclic=True),
            m=5,
            by="var",
            id="2",
            fx=True,
        ),
        "s(a,b,bs='cc',m=5,by=var,id='2',fx=TRUE)",
    ),
    (TensorSmooth(("a", "b")), "te(a,b)"),
    (TensorSmooth(("a", "b"), interaction_only=True), "ti(a,b)"),
    (
        TensorSmooth(
            ("long", "lat"),
            bs=SplineOnSphere(),
            m=3,
            d=5,
            by="var",
            np=False,
            id="my_id",
            fx=True,
            interaction_only=True,
        ),
        "ti(long,lat,bs='sos',d=5,m=3,by=var,id='my_id',fx=TRUE,np=FALSE)",
    ),
]


@pytest.mark.parametrize(("smooth", "string"), test_cases)
def test_Smooth_to_str(smooth, string):
    assert str(smooth) == string
