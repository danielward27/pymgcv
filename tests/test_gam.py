import numpy as np
import pytest
import rpy2.robjects as ro

from pymgcv.converters import rlistvec_to_dict

mgcv = ro.packages.importr("mgcv")  # type: ignore


from .gam_test_cases import GAMTestCase, get_test_cases

test_cases = get_test_cases()


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_pymgcv_mgcv_equivilance(test_case: GAMTestCase):
    pymgcv_gam = test_case.gam_model.fit(test_case.data)
    mgcv_gam = test_case.mgcv_gam(test_case.data)
    assert (
        pytest.approx(
            expected=rlistvec_to_dict(mgcv_gam)["coefficients"],
        )
        == pymgcv_gam.coefficients().to_numpy()
    )


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_predict_terms_structure(test_case: GAMTestCase):
    fit = test_case.gam_model.fit(test_case.data)
    all_terms = fit.partial_effects(test_case.data)
    expected = test_case.expected_predict_terms_structure

    assert sorted(all_terms.keys()) == sorted(expected.keys())

    for term_name, fit_and_se in all_terms.items():
        for fit_or_se in ["fit", "se"]:
            actual = fit_and_se[fit_or_se].columns.values.tolist()
            assert sorted(expected[term_name]) == sorted(actual)


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_partial_effects_colsum_matches_predict(test_case: GAMTestCase):
    pymgcv_gam = test_case.gam_model.fit(test_case.data)
    predictions = pymgcv_gam.predict(test_case.data)
    term_predictions = pymgcv_gam.partial_effects(test_case.data)

    for target, pred in predictions.items():
        term_fit = term_predictions[target]["fit"]
        assert pytest.approx(pred["fit"]) == term_fit.sum(axis=1)


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_partial_effect_against_partial_effects(test_case: GAMTestCase):
    fit = test_case.gam_model.fit(test_case.data)

    partial_effects = fit.partial_effects(test_case.data)

    all_predictors = fit.gam.all_predictors
    for target, terms in all_predictors.items():
        for term in terms:
            try:
                effect = fit.partial_effect(target, term, test_case.data)
            except NotImplementedError as e:
                if str(e) != "":
                    raise e
                continue

            name = term.label()
            expected_fit = pytest.approx(partial_effects[target]["fit"][name], abs=1e-6)
            expected_se = pytest.approx(partial_effects[target]["se"][name], abs=1e-6)

            assert expected_fit == effect["fit"]
            assert expected_se == effect["se"]


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_coef_and_cov(test_case: GAMTestCase):
    fit = test_case.gam_model.fit(test_case.data)
    coef = fit.coefficients()
    cov = fit.covariance()
    assert cov.shape[0] == cov.shape[1]
    assert cov.shape[0] == coef.shape[0]
    assert np.all(coef.index == cov.index)
