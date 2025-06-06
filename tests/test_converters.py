import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import ListVector

from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py


def test_rlistvec_to_dict():
    """Test the list vector to dict conversion."""
    # Test with a simple list vector
    d = {"a": 1, "b": 2}

    reconstructed = rlistvec_to_dict(ListVector(d))

    # Integer results in vector shape (1,) - will permit this rule for now
    assert d["a"] == reconstructed["a"].item()
    assert d["b"] == reconstructed["b"].item()

    # Test errors with a list vector with duplicate names

    x = ListVector([("a", 1), ("b", 2), ("a", 3)])

    with pytest.raises(ValueError, match="duplicate names"):
        rlistvec_to_dict(x)


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
    data = pd.DataFrame({
        "y": np.arange(3),
        "x": pd.Categorical(
            ["green", "green", "blue"],
            categories=["red", "green", "blue"],
        ),
    })

    rdf = data_to_rdf(data)
    factor = rdf.rx2("x")
    assert isinstance(factor, ro.vectors.FactorVector)
    assert factor.nlevels == 3

    rdf = data_to_rdf(pd.DataFrame(data))
    factor = rdf.rx2("x")
    assert isinstance(factor, ro.vectors.FactorVector)
    assert factor.nlevels == 3
