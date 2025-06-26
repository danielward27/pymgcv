"""Smoke tests for plotting functions, simply check that they run without errors.

To see the plots created during testing, replace plt.close("all") with plt.show()
"""

import matplotlib.pyplot as plt
import pytest

from pymgcv.plot import plot_categorical, plot_continuous_1d, plot_continuous_2d

from . import gam_test_cases as tc


def get_test_cases_1d_continuous():
    test_cases_1d_continuous = [
        (tc.linear_gam, {"target": "y"}),
        (tc.smooth_1d_gam, {"target": "y"}),
        (tc.smooth_1d_by_numeric_gam, {"target": "y", "by_val": 1}),
        (tc.smooth_1d_random_wiggly_curve_gam, {"target": "y", "by_val": "a"}),
        (tc.smooth_1d_by_categorical_gam, {"target": "y", "by_val": "a"}),
    ]
    return {f.__name__: (f(), kwargs) for f, kwargs in test_cases_1d_continuous}


test_cases_1d_continuous = get_test_cases_1d_continuous()


@pytest.mark.parametrize(
    ("test_case", "kwargs"),
    test_cases_1d_continuous.values(),
    ids=test_cases_1d_continuous.keys(),
)
def test_plot_continuous_1d(test_case: tc.GAMTestCase, kwargs: dict):
    gam = test_case.gam_model.fit(test_case.data)
    term = gam.gam.all_formulae[kwargs["target"]][0]  # Assume first term of interest
    plot_continuous_1d(**kwargs, gam=gam, term=term, data=test_case.data)
    plt.close("all")


def get_test_cases_2d_continuous():
    test_cases_1d_continuous = [
        (tc.smooth_2d_gam, {"target": "y"}),
        (tc.tensor_2d_gam, {"target": "y"}),
        (tc.tensor_2d_by_numeric_gam, {"target": "y", "by_val": 1}),
        (tc.tensor_2d_by_categorical_gam, {"target": "y", "by_val": "a"}),
        (tc.tensor_2d_random_wiggly_curve_gam, {"target": "y", "by_val": "a"}),
    ]
    return {f.__name__: (f(), kwargs) for f, kwargs in test_cases_1d_continuous}


test_cases_2d_continuous = get_test_cases_2d_continuous()


@pytest.mark.parametrize(
    ("test_case", "kwargs"),
    test_cases_2d_continuous.values(),
    ids=test_cases_2d_continuous.keys(),
)
def test_plot_continuous_2d(test_case: tc.GAMTestCase, kwargs: dict):
    gam = test_case.gam_model.fit(test_case.data)
    term = gam.gam.all_formulae[kwargs["target"]][0]  # Assume first term of interest
    plot_continuous_2d(**kwargs, gam=gam, term=term, data=test_case.data)
    plt.close("all")


def test_plot_categorical():
    test_case = tc.categorical_linear_gam()
    gam = test_case.gam_model.fit(test_case.data)
    term = gam.gam.predictors["y"][0]
    plot_categorical("y", gam=gam, term=term, data=test_case.data)
    plt.close("all")
