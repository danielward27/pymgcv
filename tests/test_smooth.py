from pymgcv.smooths import Smooth


def test_smooth_str_conversion():
    s = Smooth("x", "y")
    assert str(s) == "s(x,y,k=-1,bs='tp')"
