import numpy as np
import pandas as pd
import rpy2.robjects as ro

from pymgcv.converters import data_to_rdf, to_py


def test_data_to_rdf_basic_dict():
    d = pd.DataFrame({"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])})
    df = data_to_rdf(d)

    assert df.nrow == 3
    assert df.ncol == 2
    assert list(df.rx2("a")) == [1, 2, 3]
    assert list(df.rx2("b")) == [4, 5, 6]


def test_data_to_rdf_with_matrix():
    d = pd.DataFrame({"a": np.array([1, 2, 3]), "b0": np.ones(3), "b1": np.ones(3)})
    df = data_to_rdf(d, as_array_prefixes=("b",))

    assert df.nrow == 3
    assert df.ncol == 2
    assert to_py(df.rx2("a")).shape == (3,)
    assert to_py(df.rx2("b")).shape == (3, 2)


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

    rdf = data_to_rdf(data)
    factor = rdf.rx2("x")
    assert isinstance(factor, ro.vectors.FactorVector)
    assert factor.nlevels == 3

    rdf = data_to_rdf(pd.DataFrame(data))
    factor = rdf.rx2("x")
    assert isinstance(factor, ro.vectors.FactorVector)
    assert factor.nlevels == 3
