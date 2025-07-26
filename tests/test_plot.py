"""Smoke tests for plotting functions, simply check that they run without errors.

To see the plots created during testing, replace plt.close("all") with plt.show()
"""

import matplotlib.pyplot as plt
import pytest

from pymgcv.gam import BAM, GAM, AbstractGAM
from pymgcv.plot import (
    plot_categorical,
    plot_continuous_1d,
    plot_continuous_2d,
    plot_gam,
    plot_qq
)

from . import gam_test_cases as tc


def get_cases_1d_continuous(model_type: type[AbstractGAM]):
    cases = [
        (tc.linear_gam, {"target": "y"}),
        (tc.smooth_1d_gam, {"target": "y", "residuals": True}),
        (tc.smooth_1d_by_numeric_gam, {"target": "y"}),
        (tc.smooth_1d_random_wiggly_curve_gam, {"target": "y", "level": "a"}),
        (tc.smooth_1d_by_categorical_gam, {"target": "y", "level": "a"}),
    ]
    return {
        f"{model_type.__name__} - {f.__name__}": (f(model_type), kwargs)
        for f, kwargs in cases
    }


cases_1d_continuous = get_cases_1d_continuous(GAM) | get_cases_1d_continuous(BAM)


@pytest.mark.parametrize(
    ("test_case", "kwargs"),
    cases_1d_continuous.values(),
    ids=cases_1d_continuous.keys(),
)
def test_plot_continuous_1d(test_case: tc.GAMTestCase, kwargs: dict):
    gam = test_case.gam_model.fit(test_case.data)
    term = gam.all_predictors[kwargs["target"]][0]  # Assume first term of interest
    plot_continuous_1d(**kwargs, gam=gam, term=term, data=test_case.data)
    plt.close("all")


def get_cases_2d_continuous(model_type: type[AbstractGAM]):
    test_cases_1d_continuous = [
        (tc.smooth_2d_gam, {"target": "y"}),
        (tc.tensor_2d_gam, {"target": "y"}),
        (tc.tensor_2d_by_numeric_gam, {"target": "y"}),
        (tc.tensor_2d_by_categorical_gam, {"target": "y", "level": "a"}),
        (tc.tensor_2d_random_wiggly_curve_gam, {"target": "y", "level": "a"}),
    ]
    return {
        f"{model_type.__name__} - {f.__name__}": (f(model_type), kwargs)
        for f, kwargs in test_cases_1d_continuous
    }


cases_2d_continuous = get_cases_2d_continuous(GAM) | get_cases_2d_continuous(BAM)


@pytest.mark.parametrize(
    ("test_case", "kwargs"),
    cases_2d_continuous.values(),
    ids=cases_2d_continuous.keys(),
)
def test_plot_continuous_2d(test_case: tc.GAMTestCase, kwargs: dict):
    gam = test_case.gam_model.fit(test_case.data)
    term = gam.all_predictors[kwargs["target"]][0]  # Assume first term of interest
    plot_continuous_2d(**kwargs, gam=gam, term=term, data=test_case.data)
    plt.close("all")


@pytest.mark.parametrize(
    "model_type",
    [GAM, BAM],
)
def test_plot_categorical(model_type: type[AbstractGAM]):
    test_case = tc.categorical_linear_gam(model_type)
    gam = test_case.gam_model.fit(test_case.data)
    term = gam.predictors["y"][0]
    plot_categorical(target="y", gam=gam, term=term, data=test_case.data)
    plt.close("all")


all_test_cases = (
    cases_1d_continuous
    | cases_2d_continuous
    | {"categorical_linear": (tc.categorical_linear_gam(GAM), {"target": "y"})}
)

all_gam_test_cases = tc.get_test_cases()


@pytest.mark.parametrize(
    "test_case",
    all_gam_test_cases.values(),
    ids=all_gam_test_cases.keys(),
)
def test_plot_gam(test_case: tc.GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data)
    try:
        plot_gam(gam=gam, ncols=1)
    except ValueError as e:
        if "plot any" in str(e):
            pass
        else:
            raise

    plt.close("all")


@pytest.mark.parametrize(
    "test_case",
    all_gam_test_cases.values(),
    ids=all_gam_test_cases.keys(),
)
def test_qq_plot(test_case: tc.GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data)
    try:
        plot_qq(gam=gam)
    except NotImplementedError as e:
        if "Quantile" in str(e):  # e.g. mvn and gaulss
            pass
        else:
            raise
    plt.close("all")
