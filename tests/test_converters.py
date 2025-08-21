import numpy as np
import pandas as pd
import rpy2.robjects as ro

from pymgcv.rpy_utils import data_to_rdf


def test_data_to_rdf():
    d = pd.DataFrame({"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])})
    df = data_to_rdf(d, include="all")
    assert df.nrow == 3
    assert df.ncol == 2
    assert list(df.rx2("a")) == [1, 2, 3]
    assert list(df.rx2("b")) == [4, 5, 6]


def test_data_to_rdf_with_matrix():
    d = {"a": np.array([1, 2, 3]), "b": np.ones((3, 4))}
    df = data_to_rdf(d, include="all")
    assert df.nrow == 3
    assert df.ncol == 2
    assert df.rx2["b"].ncol == 4


def test_data_to_rdf_categorical_factors():
    data = pd.DataFrame(
        {
            "y": np.arange(3),
            "x": pd.Categorical(
                ["green", "green", "blue"],
                categories=["red", "green", "blue"],
            ),
        },
    )

    rdf = data_to_rdf(data, include="all")
    factor = rdf.rx2("x")
    assert isinstance(factor, ro.vectors.FactorVector)
    assert factor.nlevels == 3

    rdf = data_to_rdf(pd.DataFrame(data), include="all")
    factor = rdf.rx2("x")
    assert isinstance(factor, ro.vectors.FactorVector)
    assert factor.nlevels == 3
