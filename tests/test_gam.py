from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

from pymgcv import Smooth, TensorSmooth, terms
from pymgcv.bases import MarkovRandomField, RandomEffect
from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py
from pymgcv.gam import FittedGAM, gam, terms_to_formula
from pymgcv.term_utils import smooth_by_factor
from pymgcv.terms import Smooth

mgcv = ro.packages.importr("mgcv")  # type: ignore


def test_variables_to_formula():
    assert (
        terms_to_formula(
            dependent="y",
            terms=("x",),
        )
        == "y~x"
    )
    assert (
        terms_to_formula(
            dependent="y",
            terms=("x0", Smooth("x1")),
        )
        == "y~x0+s(x1)"
    )


class AbstractTestCase(ABC):
    mgcv_call: str
    add_to_r_env: dict[str, ro.RObject]

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

        def get_data(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "x": np.random.uniform(0, 1, 200),
                    "y": np.random.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            return gam(
                dependent="y",
                terms=(terms.Linear("x"),),
                data=data,
            )

    class SingleSmoothGAM(AbstractTestCase):
        mgcv_call = """
            gam(y~s(x), data=data)
            """
        add_to_r_env = {}

        def get_data(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "x": np.random.uniform(0, 1, 200),
                    "y": np.random.normal(0, 0.2, 200),
                },
            )

        def pymgcv_gam(self, data) -> FittedGAM:
            return gam(
                dependent="y",
                terms=(terms.Smooth("x"),),
                data=data,
            )

    class SingleTensorSmoothGAM(AbstractTestCase):
        mgcv_call = """
        gam(y~te(x0, x1), data=data)
        """
        add_to_r_env = {}

        def get_data(self) -> pd.DataFrame:
            rng = np.random.default_rng(seed=42)
            n = 200
            x0 = rng.uniform(0, 1, n)
            x1 = rng.uniform(0, 1, n)
            y = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1) + rng.normal(0, 0.2, n)
            return pd.DataFrame({"y": y, "x0": x0, "x1": x1})

        def pymgcv_gam(self, data) -> FittedGAM:
            return gam(
                dependent="y",
                terms=(TensorSmooth("x0", "x1"),),
                data=data,
            )

    class SingleRandomEffect(AbstractTestCase):  # TODO fix
        mgcv_call = "gam(y~s(x) + s(group, bs='re'), data=data)"
        add_to_r_env = {}

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
            return gam(
                dependent="y",
                terms=[
                    Smooth("x"),
                    Smooth("group", bs=RandomEffect()),
                ],  # TODO are strings supported?
                data=data,
            )

    class SimpleGAM(AbstractTestCase):
        mgcv_call = """
            gam(y~x0+s(x1)+s(x2), data=data)
            """
        add_to_r_env = {}

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
            return gam(
                dependent="y",
                terms=(
                    terms.Linear("x0"),
                    terms.Smooth("x1"),
                    terms.Smooth("x2"),
                ),
                data=data,
            )

        # class SmoothByFactorGAM(AbstractTestCase):  # TODO not currently supported in this form.
        #     mgcv_call = "gam(y~s(x, by=group), data=data)"
        #     add_to_r_env = {}

        #     def get_data(self) -> pd.DataFrame:
        #         rng = np.random.default_rng(1)
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
        #             Smooth("x", by="group"),
        #         ],  # TODO are strings supported?
        #         data=data,
        #     )

    class MarkovRandomFieldGAM(AbstractTestCase):
        mgcv_call = """
        gam(crime ~ s(district,bs="mrf",xt=list(polys=polys)),data=columb,method="REML")
        """
        add_to_r_env: dict

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

            return gam(
                dependent="y",
                terms=(Smooth("district", bs=MarkovRandomField(polys=polys)),),
                data=data,
            )

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
        SimpleGAM(),
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
def test_predict_term_against_predict_terms(test_case: AbstractTestCase):
    # Test column names are expected
    data = test_case.get_data()
    fit = test_case.pymgcv_gam(data)
    all_terms = fit.predict_terms(data)

    for term in fit.terms:
        term_fit = fit.predict_term(term, data=data)
        for fit_or_se in ["fit", "se"]:
            assert term_fit[fit_or_se].name in all_terms[fit_or_se].columns
            assert (
                pytest.approx(term_fit[fit_or_se])
                == all_terms[fit_or_se][term_fit[fit_or_se].name]
            )


@pytest.mark.parametrize("test_case", test_cases, ids=[type(t) for t in test_cases])
def test_fitted_gam_predict_term(test_case: AbstractTestCase):
    data = test_case.get_data()
    fit = test_case.pymgcv_gam(data)

    for term in fit.terms:
        pred = fit.predict_term(term, data=data)
        assert pred["fit"].shape == (data.shape[0], 1)
        assert pred["se"].shape == (data.shape[0], 1)

        # test minimal data still runs and produces same result
        minimal_data = pd.DataFrame({k: data[k] for k in term.varnames})
        pred2 = fit.predict_term(term, data=minimal_data)
        assert pytest.approx(pred["fit"]) == pred2["fit"]
        assert pytest.approx(pred["se"]) == pred2["se"]


def test_categorical_by_variables_not_supported():
    data = pd.DataFrame(
        {
            "x": np.random.randn(100),
            "y": np.random.randn(100),
            "z": pd.Categorical(np.random.choice(["a", "b", "c"], 100)),
        },
    )
    with pytest.raises(TypeError, match="Categorical by variables not yet supported"):
        gam("y", [Smooth("x", by="z")], data=data)


def test_factor_by():
    np.random.seed(42)
    n = 200

    # Create dataframe
    df = pd.DataFrame(
        {
            "x": np.random.uniform(0, 10, n),
            "group": np.random.choice(["A", "B"], size=n),
        },
    )

    # Create response with different smooth functions for groups A and B
    df["y"] = np.where(
        df["group"] == "A",
        np.sin(df["x"]) + np.random.normal(0, 0.2, n),
        np.cos(df["x"]) + np.random.normal(0, 0.2, n),
    )
    df["group"] = df["group"].astype("category")

    factor = df["group"]
    assert isinstance(factor, pd.Series)
    smooths, indicators = smooth_by_factor("x", smooth_type=Smooth, factor=factor)

    gam(
        "y",
        terms=smooths,
        data=pd.concat([df, indicators], axis=1),
    )  # TODO can we test against MGCV?
