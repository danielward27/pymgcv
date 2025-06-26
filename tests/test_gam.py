from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

from pymgcv import terms
from pymgcv.basis_functions import (
    CubicSpline,
    MarkovRandomField,
    RandomEffect,
    RandomWigglyCurve,
)
from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py
from pymgcv.gam import GAM, FittedGAM
from pymgcv.terms import Linear as L
from pymgcv.terms import Smooth as S
from pymgcv.terms import TensorSmooth as T

mgcv = ro.packages.importr("mgcv")  # type: ignore


class AbstractTestCase(ABC):
    mgcv_call: str
    add_to_r_env: dict[str, ro.RObject]
    expected_predict_terms_structure: dict[str, list[str]]

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """Returns data dictionary."""
        pass

    @abstractmethod
    def pymgcv_gam(
        self,
        data: pd.DataFrame,
    ) -> FittedGAM:
        """Returns pymgcv gam."""
        pass

    def mgcv_gam(self, data: pd.DataFrame):
        with ro.local_context() as env:
            env["data"] = data_to_rdf(data)
            for k, v in self.add_to_r_env.items():
                env[k] = v
            return ro.r(self.mgcv_call)


def get_test_cases():
    class SingleLinearGAM(AbstractTestCase):
        mgcv_call = """
                gam(y~x, data=data)
                """
        add_to_r_env = {}
        expected_predict_terms_structure = {"y": ["Linear(x)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x": rng.uniform(0, 1, 200),
                    "y": rng.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": L("x")})
            return model.fit(data=data)

    class SingleCategoricalLinearGAM(AbstractTestCase):
        mgcv_call = """
                gam(y~group, data=data)
                """
        add_to_r_env = {}
        expected_predict_terms_structure = {"y": ["Linear(group)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "group": pd.Series(
                        rng.choice(["A", "B", "C"], 200),
                        dtype="category",
                    ),
                    "y": rng.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": L("group")})
            return model.fit(data=data)

    class SingleSmoothGAM(AbstractTestCase):
        mgcv_call = """
            gam(y~s(x), data=data)
            """
        add_to_r_env = {}
        expected_predict_terms_structure = {"y": ["Smooth(x)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            x = rng.uniform(0, 1, 200)
            return pd.DataFrame(
                {
                    "x": x,
                    "y": rng.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": S("x")})
            return model.fit(data=data)

    class SingleTensorSmoothGAM(AbstractTestCase):
        mgcv_call = """
        gam(y~te(x0, x1), data=data)
        """
        add_to_r_env = {}
        expected_predict_terms_structure = {"y": ["TensorSmooth(x0,x1)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            n = 200
            x0 = rng.uniform(0, 1, n)
            x1 = rng.uniform(0, 1, n)
            y = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1) + rng.normal(0, 0.2, n)
            return pd.DataFrame({"y": y, "x0": x0, "x1": x1})

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM(
                {"y": [T("x0", "x1")]},
            )
            return model.fit(data=data)

    class SingleRandomEffect(AbstractTestCase):
        mgcv_call = "gam(y~s(x) + s(group, bs='re'), data=data)"
        add_to_r_env = {}
        expected_predict_terms_structure = {
            "y": ["Smooth(x)", "Smooth(group)", "intercept"],
        }

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(42)
            n = 50
            group = pd.Series(
                pd.Categorical(
                    rng.choice(["a", "b", "c"], n),
                    categories=["a", "b", "c"],
                ),
            )
            x = np.linspace(0, 10, n)
            y = np.sin(x) + group.cat.codes + rng.normal(scale=0.1, size=n)
            return pd.DataFrame(
                {
                    "group": group,
                    "x": x,
                    "y": y,
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": S("x") + S("group", bs=RandomEffect())})
            return model.fit(data=data)

    class SingleFactorInteractionGAM(AbstractTestCase):
        mgcv_call = """
            gam(y~a:b, data=data)
            """
        add_to_r_env = {}
        expected_predict_terms_structure = {"y": ["Interaction(a,b)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "a": pd.Categorical(
                        rng.choice(["A", "B"], size=200, replace=True),
                    ),
                    "b": pd.Categorical(
                        rng.choice(["C", "D", "E"], size=200, replace=True),
                    ),
                    "y": rng.uniform(0, 1, size=200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM(
                {"y": [terms.Interaction("a", "b")]},
            )
            return model.fit(data=data)

    class SimpleGAM(AbstractTestCase):
        mgcv_call = """
            gam(y~x0+s(x1)+s(x2), data=data)
            """
        add_to_r_env = {}
        expected_predict_terms_structure = {
            "y": ["Linear(x0)", "Smooth(x1)", "Smooth(x2)", "intercept"],
        }

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x0": rng.uniform(0, 1, 200),
                    "x1": rng.uniform(0, 1, 200),
                    "x2": rng.uniform(0, 1, 200),
                    "y": rng.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": L("x0") + S("x1") + S("x2")})
            return model.fit(data=data)

    class MultivariateMultiFormula(AbstractTestCase):
        mgcv_call = """
            gam(
                list(
                    y0 ~ s(x, k=5),
                    y1 ~ x
                ),
                data=data,
                family=mvn(d=2)
            )
            """
        add_to_r_env = {}
        expected_predict_terms_structure = {
            "y0": ["Smooth(x)", "intercept"],
            "y1": ["Linear(x)", "intercept"],
        }

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x": rng.uniform(0, 1, 200),
                    "y0": rng.normal(0, 0.2, 200),
                    "y1": rng.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM(
                {"y0": S("x", k=5), "y1": L("x")},
                family="mvn(d=2)",
            )
            return model.fit(data=data)

    class LocationScaleMultiFormula(AbstractTestCase):
        mgcv_call = """
            gam(
                list(
                    y ~ s(x0),
                    ~ s(x1)
                ),
                data=data,
                family=gaulss()
            )
            """
        add_to_r_env = {}
        expected_predict_terms_structure = {
            "y": ["Smooth(x0)", "intercept"],
            "log_scale": ["Smooth(x1)", "intercept"],
        }

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x0": rng.uniform(0, 1, 200),
                    "x1": rng.normal(0, 0.2, 200),
                    "y": rng.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM(
                {"y": S("x0")},
                family_predictors={"log_scale": S("x1")},
                family="gaulss()",
            )
            return model.fit(data=data)

    class OffsetGAM(AbstractTestCase):
        mgcv_call = """
            gam(y~s(x) + offset(z), data=data)
            """
        add_to_r_env = {}
        expected_predict_terms_structure = {
            "y": ["Smooth(x)", "Offset(z)", "intercept"],
        }

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x": rng.uniform(0, 1, 200),
                    "z": rng.uniform(0, 1, 200),
                    "y": rng.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": S("x") + terms.Offset("z")})
            return model.fit(data=data)

    class SmoothByCategoricalGAM(AbstractTestCase):
        mgcv_call = """
                gam(y~s(x, by=group), data=data)
                """
        add_to_r_env = {}
        expected_predict_terms_structure = {"y": ["Smooth(x,by=group)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x": rng.standard_normal(100),
                    "group": pd.Categorical(rng.choice(["a", "b", "c"], 100)),
                    "y": rng.standard_normal(100),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": S("x", by="group")})
            return model.fit(data=data)

    class SmoothByNumericGAM(AbstractTestCase):
        mgcv_call = """
                    gam(y~s(x, by=by_var), data=data)
                    """
        add_to_r_env = {}
        expected_predict_terms_structure = {"y": ["Smooth(x,by=by_var)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x": rng.standard_normal(100),
                    "by_var": rng.standard_normal(100),
                    "y": rng.standard_normal(100),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": S("x", by="by_var")})
            return model.fit(data=data)

    class TensorByCategoricalGAM(AbstractTestCase):
        mgcv_call = """
                gam(y~te(x0,x1, by=group), data=data)
                """
        add_to_r_env = {}
        expected_predict_terms_structure = {
            "y": ["TensorSmooth(x0,x1,by=group)", "intercept"],
        }

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x0": rng.standard_normal(100),
                    "x1": rng.standard_normal(100),
                    "group": pd.Categorical(rng.choice(["a", "b", "c"], 100)),
                    "y": rng.standard_normal(100),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": T("x0", "x1", by="group")})
            return model.fit(data=data)

    class RandomWigglyCurveSmoothGAM(AbstractTestCase):
        mgcv_call = """
            gam(y~s(x,group,bs="fs",xt=list(bs="cr")),data=data)
        """
        add_to_r_env = {}
        expected_predict_terms_structure = {
            "y": ["Smooth(x,group)", "intercept"],
        }

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x": rng.standard_normal(100),
                    "group": pd.Categorical(rng.choice(["a", "b", "c"], 100)),
                    "y": rng.standard_normal(100),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            bs = RandomWigglyCurve(CubicSpline())
            model = GAM({"y": S("x", "group", bs=bs)})
            return model.fit(data=data)

    class TensorByNumericGAM(AbstractTestCase):
        mgcv_call = """
                    gam(y~te(x0,x1,by=by_var), data=data)
                    """
        add_to_r_env = {}
        expected_predict_terms_structure = {
            "y": ["TensorSmooth(x0,x1,by=by_var)", "intercept"],
        }

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x0": rng.standard_normal(100),
                    "x1": rng.standard_normal(100),
                    "by_var": rng.standard_normal(100),
                    "y": rng.standard_normal(100),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": T("x0", "x1", by="by_var")})
            return model.fit(data=data)

    class PoissonGAM(AbstractTestCase):
        mgcv_call = """
                    gam(counts~s(x), data=data, family=poisson)
                    """
        add_to_r_env = {}
        expected_predict_terms_structure = {"counts": ["Smooth(x)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                {
                    "x": rng.standard_normal(100),
                    "counts": rng.poisson(size=100),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"counts": S("x")}, family="poisson")
            return model.fit(data=data)

    class MarkovRandomFieldGAM(AbstractTestCase):
        mgcv_call = """
        gam(crime ~ s(district,bs="mrf",xt=list(polys=polys)),data=columb,method="REML")
        """
        add_to_r_env: dict
        expected_predict_terms_structure = {"crime": ["Smooth(district)", "intercept"]}

        def __init__(self):
            polys = ro.packages.data(mgcv).fetch("columb.polys")["columb.polys"]
            self.add_to_r_env = {"polys": polys}

        def get_data(self) -> pd.DataFrame:
            data = ro.packages.data(mgcv).fetch("columb")["columb"]
            data = to_py(data)

            rng = np.random.default_rng(seed=42)
            n = 200
            x0 = rng.uniform(0, 1, n)
            x1 = rng.uniform(0, 1, n)
            y = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1) + rng.normal(0, 0.2, n)
            return pd.DataFrame({"y": y, "x0": x0, "x1": x1})

        def pymgcv_gam(self, data) -> FittedGAM:
            polys = list(rlistvec_to_dict(self.add_to_r_env["polys"]).values())
            model = GAM(
                {"y": S("district", bs=MarkovRandomField(polys=polys))},
            )
            return model.fit(data=data)

    # TODO MRF not included yet!
    # TODO:
    # multivariate gam

    return [
        SingleLinearGAM(),
        SingleCategoricalLinearGAM(),
        SingleSmoothGAM(),
        SingleTensorSmoothGAM(),
        SingleRandomEffect(),
        RandomWigglyCurveSmoothGAM(),
        SingleFactorInteractionGAM(),
        SimpleGAM(),
        MultivariateMultiFormula(),
        LocationScaleMultiFormula(),
        OffsetGAM(),
        SmoothByCategoricalGAM(),
        SmoothByNumericGAM(),
        TensorByCategoricalGAM(),
        TensorByNumericGAM(),
        PoissonGAM(),
    ]


test_cases = get_test_cases()


@pytest.mark.parametrize("test_case", test_cases, ids=[type(t) for t in test_cases])
def test_pymgcv_mgcv_equivilance(test_case: AbstractTestCase):
    data = test_case.get_data()
    pymgcv_gam = test_case.pymgcv_gam(data)
    mgcv_gam = test_case.mgcv_gam(data)
    assert (
        pytest.approx(
            expected=rlistvec_to_dict(mgcv_gam)["coefficients"],
        )
        == pymgcv_gam.coefficients
    )
    # TODO test predictions equal?


@pytest.mark.parametrize("test_case", test_cases, ids=[type(t) for t in test_cases])
def test_predict_terms_structure(test_case: AbstractTestCase):
    data = test_case.get_data()
    fit = test_case.pymgcv_gam(data)
    all_terms = fit.partial_effects(data)  # TODO: For now just testing that it runs.
    expected = test_case.expected_predict_terms_structure
    assert sorted(all_terms.keys()) == sorted(expected.keys())

    for term_name, fit_and_se in all_terms.items():
        for fit_or_se in ["fit", "se"]:
            actual = fit_and_se[fit_or_se].columns.values.tolist()
            assert sorted(expected[term_name]) == sorted(actual)


@pytest.mark.parametrize("test_case", test_cases, ids=[type(t) for t in test_cases])
def test_partial_effects_colsum_matches_predict(test_case: AbstractTestCase):
    data = test_case.get_data()
    pymgcv_gam = test_case.pymgcv_gam(data)
    predictions = pymgcv_gam.predict(
        data,
    )
    term_predictions = pymgcv_gam.partial_effects(data)

    for target, pred in predictions.items():
        term_fit = term_predictions[target]["fit"]
        assert pytest.approx(pred["fit"]) == term_fit.sum(axis=1)


@pytest.mark.parametrize("test_case", test_cases, ids=[type(t) for t in test_cases])
def test_partial_effect_against_partial_effects(test_case: AbstractTestCase):
    data = test_case.get_data()
    fit = test_case.pymgcv_gam(data)

    partial_effects = fit.partial_effects(data)

    all_formulae = fit.gam.all_formulae
    for target, terms in all_formulae.items():
        for term in terms:
            try:
                effect = fit.partial_effect(target, term, data)
            except NotImplementedError as e:
                if str(e) != "":
                    raise e
                continue

            name = term.label()
            expected_fit = pytest.approx(partial_effects[target]["fit"][name], abs=1e-6)
            expected_se = pytest.approx(partial_effects[target]["se"][name], abs=1e-6)

            assert expected_fit == effect["fit"]
            assert expected_se == effect["se"]


@pytest.mark.parametrize("test_case", test_cases, ids=[type(t) for t in test_cases])
def test_coef_and_cov(test_case: AbstractTestCase):
    data = test_case.get_data()
    fit = test_case.pymgcv_gam(data)
    coef = fit.coefficients()
    cov = fit.covariance()
    assert cov.shape[0] == cov.shape[1]
    assert cov.shape[0] == coef.shape[0]
    assert np.all(coef.index == cov.index)
