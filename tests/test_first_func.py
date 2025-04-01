import pymgcv

# Simple examples, such that we can set up CI and check for environment issues.


def test_first_func():
    assert pymgcv.first_func.first_func()


def test_basic_gam():
    assert pymgcv.first_func.basic_gam() is not None
