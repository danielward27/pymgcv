import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

from pymgcv import terms
from pymgcv.basis_functions import MarkovRandomField, RandomEffect
from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py
from pymgcv.gam import GAM, FittedGAM
from pymgcv.rgam_utils import _get_intercepts_and_se
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
        expected_predict_terms_structure = {"y": ["x", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "x": np.random.uniform(0, 1, 200),
                    "y": np.random.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": L("x")})
            return model.fit(data=data)

    class SingleSmoothGAM(AbstractTestCase):
        mgcv_call = """
            gam(y~s(x), data=data)
            """
        add_to_r_env = {}
        expected_predict_terms_structure = {"y": ["s(x)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            x = np.random.uniform(0, 1, 200)
            return pd.DataFrame(
                {
                    "x": x,
                    "y": np.random.normal(0, 0.2, 200),
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
        expected_predict_terms_structure = {"y": ["te(x0,x1)", "intercept"]}

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

    class SingleRandomEffect(AbstractTestCase):  # TODO fix
        mgcv_call = "gam(y~s(x) + s(group, bs='re'), data=data)"
        add_to_r_env = {}
        expected_predict_terms_structure = {"y": ["s(x)", "s(group)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(1)
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
        expected_predict_terms_structure = {"y": ["a:b", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "a": pd.Categorical(
                        np.random.choice(["A", "B"], size=200, replace=True),
                    ),
                    "b": pd.Categorical(
                        np.random.choice(["C", "D", "E"], size=200, replace=True),
                    ),
                    "y": np.random.uniform(0, 1, size=200),
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
        expected_predict_terms_structure = {"y": ["x0", "s(x1)", "s(x2)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "x0": np.random.uniform(0, 1, 200),
                    "x1": np.random.uniform(0, 1, 200),
                    "x2": np.random.uniform(0, 1, 200),
                    "y": np.random.normal(0, 0.2, 200),
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
            "y0": ["s(x)", "intercept"],
            "y1": ["x", "intercept"],
        }

        def get_data(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "x": np.random.uniform(0, 1, 200),
                    "y0": np.random.normal(0, 0.2, 200),
                    "y1": np.random.normal(0, 0.2, 200),
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
            "y": ["s(x0)", "intercept"],
            "log_scale": ["s(x1)", "intercept"],
        }

        def get_data(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "x0": np.random.uniform(0, 1, 200),
                    "x1": np.random.normal(0, 0.2, 200),
                    "y": np.random.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM(
                {"y": S("x0")},
                other_predictors={"log_scale": S("x1")},
                family="gaulss()",
            )
            return model.fit(data=data)

    class OffsetGAM(AbstractTestCase):
        mgcv_call = """
            gam(y~s(x) + offset(z), data=data)
            """
        add_to_r_env = {}
        expected_predict_terms_structure = {"y": ["s(x)", "offset(z)", "intercept"]}

        def get_data(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "x": np.random.uniform(0, 1, 200),
                    "z": np.random.uniform(0, 1, 200),
                    "y": np.random.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            model = GAM({"y": S("x") + terms.Offset("z")})
            return model.fit(data=data)

    # class SmoothByFactorGAM(AbstractTestCase):  # TODO not currently supported in this form.
    #     mgcv_call = "gam(y~s(x, by=group), data=data)"
    #     add_to_r_env = {}

    #     def get_data(self) -> pd.DataFrame:
    #         rng = np.random.default_rng(
    #         n = 50
    #         group = pd.Series(pd.Categorical(rng.choice(["a", "b", "c"], n)))
    #         x = np.linspace(0, 10, n)
    #         y = np.sin(x) + group.cat.codes + rng.normal(scale=0.1, size=n)
    #         return pd.DataFrame(
    #             {
    #                 "group": group,
    #                 "x": x,
    #                 "y": y,
    #             },
    #         )

    # def pymgcv_gam(self, data) -> FittedGAM:
    #     return gam(
    #         dependent="y",
    #         terms=[
    #             S("x", by="group"),
    #         ],  # TODO are strings supported?
    #         data=data,
    #     )

    class MarkovRandomFieldGAM(AbstractTestCase):
        mgcv_call = """
        gam(crime ~ s(district,bs="mrf",xt=list(polys=polys)),data=columb,method="REML")
        """
        add_to_r_env: dict
        expected_predict_terms_structure = {"crime": ["s(district)", "intercept"]}

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

    # TODO MRF and SmoothByFactor not included yet!
    # TODO:
    # multivariate gam
    # list formulas: gam(list(y0~s(x0)+s(x1),y1~s(x2)+s(x3)),family=mvn(d=2),data=dat)
    # Offset gam

    return [
        SingleLinearGAM(),
        SingleSmoothGAM(),
        SingleTensorSmoothGAM(),
        SingleRandomEffect(),
        SingleFactorInteractionGAM(),
        SimpleGAM(),
        MultivariateMultiFormula(),
        LocationScaleMultiFormula(),
        OffsetGAM(),
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
            assert sorted(actual) == sorted(expected[term_name])


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


def test_categorical_by_variables_not_supported():
    data = pd.DataFrame(
        {
            "x": np.random.randn(100),
            "y": np.random.randn(100),
            "z": pd.Categorical(np.random.choice(["a", "b", "c"], 100)),
        },
    )

    model = GAM({"y": S("x", by="z")})
    with pytest.raises(TypeError, match="Categorical by variables not yet supported"):
        model.fit(data=data)


def test_intercept_and_se():
    data = pd.DataFrame(
        {
            "x0": np.random.uniform(0, 1, 200),
            "x1": np.random.normal(0, 0.2, 200),
            "y": np.random.normal(0, 0.2, 200),
        },
    )

    model = GAM(
        {"y": S("x0")},
        other_predictors={"log_scale": S("x1")},
        family="gaulss()",
    )
    fit = model.fit(data=data)
    intercept = _get_intercepts_and_se(fit.rgam)

    # Use regex to check values match model summary
    summary_str = fit.summary()
    pattern = r"^\(Intercept\)\.?\d*\s+([-eE0-9.+]+)\s+([-eE0-9.+]+)"
    matches = re.findall(pattern, summary_str, re.MULTILINE)
    expected = {
        "(Intercept)": {"fit": float(matches[0][0]), "se": float(matches[0][1])},
        "(Intercept).1": {"fit": float(matches[1][0]), "se": float(matches[1][1])},
    }

    for name, expected_values in expected.items():
        got_values = intercept[name]
        for fit_or_se in ["fit", "se"]:
            assert (
                pytest.approx(expected_values[fit_or_se], abs=1e-3)
                == got_values[fit_or_se]
            )

    # TODO test model with no intercept


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

            for fit_or_se in ["fit", "se"]:
                assert (
                    pytest.approx(
                        partial_effects[target][fit_or_se][term.simple_string()],
                        abs=1e-6,
                    )
                    == effect[fit_or_se]
                )
