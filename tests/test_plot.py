"""Smoke tests for plotting functions, simply check that they run without errors.

To see the plots created during testing, replace plt.close("all") with plt.show()
"""

import matplotlib.pyplot as plt
import pytest

from pymgcv.plot import (
    plot_categorical,
    plot_continuous_1d,
    plot_continuous_2d,
    plot_gam,
)

from . import gam_test_cases as tc


def get_test_cases_1d_continuous():
    test_cases_1d_continuous = [
        (tc.linear_gam, {"target": "y"}),
        (tc.smooth_1d_gam, {"target": "y", "residuals": True}),
        (tc.smooth_1d_by_numeric_gam, {"target": "y"}),
        (tc.smooth_1d_random_wiggly_curve_gam, {"target": "y", "level": "a"}),
        (tc.smooth_1d_by_categorical_gam, {"target": "y", "level": "a"}),
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
    term = gam.gam.all_predictors[kwargs["target"]][0]  # Assume first term of interest
    plot_continuous_1d(**kwargs, fit=gam, term=term, data=test_case.data)
    plt.close("all")


def get_test_cases_2d_continuous():
    test_cases_1d_continuous = [
        (tc.smooth_2d_gam, {"target": "y"}),
        (tc.tensor_2d_gam, {"target": "y"}),
        (tc.tensor_2d_by_numeric_gam, {"target": "y"}),
        (tc.tensor_2d_by_categorical_gam, {"target": "y", "level": "a"}),
        (tc.tensor_2d_random_wiggly_curve_gam, {"target": "y", "level": "a"}),
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
    term = gam.gam.all_predictors[kwargs["target"]][0]  # Assume first term of interest
    plot_continuous_2d(**kwargs, fit=gam, term=term, data=test_case.data)
    plt.close("all")


def test_plot_categorical():
    test_case = tc.categorical_linear_gam()
    gam = test_case.gam_model.fit(test_case.data)
    term = gam.gam.predictors["y"][0]
    plot_categorical(target="y", fit=gam, term=term, data=test_case.data)
    plt.close("all")


all_test_cases = (
    test_cases_1d_continuous
    | test_cases_2d_continuous
    | {"categorical_linear": (tc.categorical_linear_gam(), {"target": "y"})}
)

# Exclude test case with categorical_interaction_gam only
all_gam_test_cases = tc.get_test_cases()
all_gam_test_cases.pop("categorical_interaction_gam")


@pytest.mark.parametrize(
    "test_case",
    all_gam_test_cases.values(),
    ids=all_gam_test_cases.keys(),
)
def test_plot_gam(test_case: tc.GAMTestCase):
    fit = test_case.gam_model.fit(test_case.data)
    print(fit.summary())
    plot_gam(fit=fit, ncols=1)
    plt.close("all")
