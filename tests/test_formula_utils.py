import numpy as np
import rpy2.robjects as ro

from pymgcv.formula_utils import _to_r_constructor_string


def test_to_r_constructor_string():
    assert _to_r_constructor_string(1) == "1L"
    assert _to_r_constructor_string(True) == "TRUE"
    assert _to_r_constructor_string("a") == '"a"'
    assert _to_r_constructor_string(ro.IntVector([7, 2, 3])) == "c(7L, 2L, 3L)"
    assert _to_r_constructor_string(ro.StrVector(["a", "b", "c"])) == 'c("a", "b", "c")'
    assert _to_r_constructor_string(ro.NA_Logical) == "NA"
    assert _to_r_constructor_string(ro.NULL) == "NULL"

    assert _to_r_constructor_string([1, 2, "a"]) == 'list(1L, 2L, "a")'

    assert (
        _to_r_constructor_string(np.ones((3, 3)))
        == "structure(c(1, 1, 1, 1, 1, 1, 1, 1, 1), dim = c(3L, 3L))"
    )

    # Check it doesn't matter if we use NA_Logical
    assert (
        _to_r_constructor_string(ro.IntVector([7, ro.NA_Logical, 3])) == "c(7L, NA, 3L)"
    )
