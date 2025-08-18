import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

from pymgcv.rpy_utils import to_py

from .gam_test_cases import GAMTestCase, get_test_cases

mgcv = ro.packages.importr("mgcv")  # type: ignore


test_cases = get_test_cases()


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_pymgcv_mgcv_equivilance(test_case: GAMTestCase):
    pymgcv_gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    mgcv_gam = test_case.mgcv_gam(test_case.data)
    assert (
        pytest.approx(
            expected=mgcv_gam.rx2["coefficients"],
        )
        == pymgcv_gam.coefficients().to_numpy()
    )


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_predict_terms_structure(test_case: GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    all_terms_with_se = gam.partial_effects(test_case.data, compute_se=True)
    expected = test_case.expected_predict_terms_structure

    assert sorted(all_terms_with_se.keys()) == sorted(expected.keys())

    for term_name, fit_and_se in all_terms_with_se.items():
        for fit_or_se in (fit_and_se.fit, fit_and_se.se):
            actual = fit_or_se.columns.values.tolist()
            assert sorted(expected[term_name]) == sorted(actual)


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_partial_effects_colsum_matches_predict(test_case: GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    predictions = gam.predict(test_case.data)
    term_predictions = gam.partial_effects(test_case.data, compute_se=True)

    for target, pred in predictions.items():
        term_fit = term_predictions[target].fit
        assert pytest.approx(pred.fit) == term_fit.sum(axis=1)


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_partial_effect_against_partial_effects(test_case: GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)

    partial_effects = gam.partial_effects(test_case.data, compute_se=True)

    all_predictors = gam.all_predictors
    for target, terms in all_predictors.items():
        for term in terms:
            try:
                effect = gam.partial_effect(
                    term,
                    target,
                    test_case.data,
                    compute_se=True,
                )
            except NotImplementedError as e:
                if str(e) != "":
                    raise e
                continue

            name = term.label()
            expected_fit = pytest.approx(partial_effects[target].fit[name], abs=1e-6)
            expected_se = pytest.approx(partial_effects[target].se[name], abs=1e-6)

            assert expected_fit == effect.fit
            assert expected_se == effect.se


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_with_se_matches_without(test_case: GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)

    partial_effects_with_se = gam.partial_effects(compute_se=True)
    partial_effects_without = gam.partial_effects(compute_se=False)

    for target in gam.all_predictors.keys():
        assert (
            pytest.approx(partial_effects_with_se[target].fit)
            == partial_effects_without[target].fit
        )


abstract_method_test_cases = [
    "GAM - smooth_1d_gam",
    "GAM - multivariate_normal_gam",
    "GAM - gaulss_gam",
    "BAM - smooth_1d_random_wiggly_curve_gam",
]


# check it gives something reasonable in both uni/multivariate, and bam case
@pytest.mark.parametrize(
    "test_case",
    [test_cases[k] for k in abstract_method_test_cases],
    ids=abstract_method_test_cases,
)
def test_abstract_methods(test_case: GAMTestCase):
    fit = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    coef = fit.coefficients()
    cov = fit.covariance()
    assert cov.shape[0] == cov.shape[1]
    assert cov.shape[0] == coef.shape[0]
    assert np.all(coef.index == cov.index)
    assert isinstance(fit.aic(), float)

    residuals = fit.residuals()
    assert residuals.shape[0] == test_case.data.shape[0]

    resid_from_y_and_fit = fit.residuals_from_y_and_fit(
        y=to_py(fit.fit_state.rgam.rx2["y"]),
        fit=to_py(fit.fit_state.rgam.rx2["fitted.values"]),
        weights=to_py(fit.fit_state.rgam.rx2["prior.weights"]),
    )
    assert np.all(residuals == resid_from_y_and_fit)
    assert isinstance(fit.edf(), pd.Series)
    assert isinstance(fit.penalty_edf(), pd.Series)
