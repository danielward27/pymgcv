

from pymgcv.gam import gam, FittedGAM

from pymgcv import terms as t
import numpy as np
import pytest 

def test_fitted_gam_predict_terms():


    n = 500
    rng = np.random.default_rng(42)
    x0 =  rng.normal(0, 1, size=n)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.normal(0, 1, size=n)
    y = 0.2*x0 + np.cos(x1) + 0.2*x2 + 0.2*x3**2 + 3 + rng.normal(0, 1, size=n)

    data = {"x0": x0, "x1": x1, "x2": x2, "x3": x3, "y": y }

    terms = [t.Linear("x0"), t.Smooth("x1","x2"), t.Smooth("x3")]

    fitted_gam = gam(
        dependent="y",
        terms=terms,
        data=data,
    )

    for term in terms:
        pred = fitted_gam.predict_term(term, data=data)
        assert pred["fit"].shape == (500, )
        assert pred["se_fit"].shape == (500, )

        # test minimal data still runs and produces same result
        minimal_data = {k: data[k] for k in term.varnames}
        pred2 = fitted_gam.predict_term(term, data=minimal_data)
        assert pytest.approx(pred["fit"]) == pred2["fit"]
        assert pytest.approx(pred["se_fit"]) == pred2["se_fit"]





    