import numpy as np
import pytest
from rpy2.robjects import ListVector

from pymgcv.converters import dict_to_rdf, list_vec_to_dict, to_py


def test_list_vector_to_dict():
    """Test the list vector to dict conversion."""
    # Test with a simple list vector
    d = {"a": 1, "b": 2}

    reconstructed = list_vec_to_dict(ListVector(d))

    # Integer results in vector shape (1,) - will permit this rule for now
    assert d["a"] == reconstructed["a"].item()
    assert d["b"] == reconstructed["b"].item()

    # Test errors with a list vector with duplicate names

    x = ListVector([("a", 1), ("b", 2), ("a", 3)])

    with pytest.raises(ValueError, match="duplicate names"):
        list_vec_to_dict(x)


def test_dict_to_rdf():
    """Test the dict of arrays to rpy dataframe conversion."""
    # Test with a simple dictionary of arrays
    d = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    df = dict_to_rdf(d)

    assert df.nrow == 3
    assert df.ncol == 2

    assert list(df.rx2("a")) == [1, 2, 3]
    assert list(df.rx2("b")) == [4, 5, 6]

    # Test with a matrix component
    d = {"a": np.array([1, 2, 3]), "b": np.ones((3, 2))}
    df = dict_to_rdf(d)

    assert df.nrow == 3
    assert df.ncol == 2

    assert to_py(df.rx2("a")).shape == (3,)
    assert to_py(df.rx2("b")).shape == (3, 2)

    with pytest.raises(ValueError, match="All arrays must match on axis 0."):
        dict_to_rdf({"a": np.ones((3,)), "b": np.ones((4,))})

    with pytest.raises(ValueError, match="All arrays must be 1D or 2D."):
        dict_to_rdf({"a": np.array([1, 2, 3]), "b": np.zeros(())})
