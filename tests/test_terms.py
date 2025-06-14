from dataclasses import dataclass

import pytest

from pymgcv.bases import CubicSpline, SplineOnSphere
from pymgcv.terms import Interaction, Linear, Smooth, TensorSmooth, TermLike


@dataclass
class TermTestCase:
    term: TermLike
    expected_str: str
    expected_simple: str
    expected_simple_with_idx: str


test_cases = [
    TermTestCase(
        term=Linear("a"),
        expected_str="a",
        expected_simple="a",
        expected_simple_with_idx="a.1",
    ),
    TermTestCase(
        term=Interaction("a", "b", "c"),
        expected_str="a:b:c",
        expected_simple="a:b:c",
        expected_simple_with_idx="a:b:c.1",
    ),
    TermTestCase(
        term=Smooth("a"),
        expected_str="s(a)",
        expected_simple="s(a)",
        expected_simple_with_idx="s.1(a)",
    ),
    TermTestCase(
        term=Smooth("a", "b", m=3),
        expected_str="s(a,b,m=3)",
        expected_simple="s(a,b)",
        expected_simple_with_idx="s.1(a,b)",
    ),
    TermTestCase(
        term=Smooth(
            "a",
            "b",
            k=10,
            bs=CubicSpline(cyclic=True),
            m=5,
            by="var",
            id="2",
            fx=True,
        ),
        expected_str="s(a,b,by=var,k=10,bs='cc',m=5,id='2',fx=TRUE)",
        expected_simple="s(a,b):var",
        expected_simple_with_idx="s.1(a,b):var",
    ),
    TermTestCase(
        term=TensorSmooth("a", "b"),
        expected_str="te(a,b)",
        expected_simple="te(a,b)",
        expected_simple_with_idx="te.1(a,b)",
    ),
    TermTestCase(
        term=TensorSmooth("a", "b", interaction_only=True),
        expected_str="ti(a,b)",
        expected_simple="ti(a,b)",
        expected_simple_with_idx="ti.1(a,b)",
    ),
    TermTestCase(
        term=TensorSmooth(
            "long",
            "lat",
            bs=[SplineOnSphere(), SplineOnSphere()],
            m=[2, 3],
            d=[5, 3],
            by="var",
            np=False,
            id="my_id",
            fx=True,
            interaction_only=True,
        ),
        expected_str="ti(long,lat,by=var,bs=c('sos','sos'),d=c(5,3),m=c(2,3),id='my_id',fx=TRUE,np=FALSE)",
        expected_simple="ti(long,lat):var",
        expected_simple_with_idx="ti.1(long,lat):var",
    ),
    TermTestCase(
        term=TensorSmooth("x", "y", bs=[CubicSpline(), CubicSpline()]),
        expected_str="te(x,y,bs=c('cr','cr'))",
        expected_simple="te(x,y)",
        expected_simple_with_idx="te.1(x,y)",
    ),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_smooth_to_str(test_case: TermTestCase):
    assert str(test_case.term) == test_case.expected_str
    assert test_case.term.simple_string() == test_case.expected_simple
    assert test_case.term.simple_string(1) == test_case.expected_simple_with_idx
