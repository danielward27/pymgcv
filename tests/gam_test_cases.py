"""A collection of GAM test cases."""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import rpy2.robjects as ro

import pymgcv.basis_functions as bs
from pymgcv import terms
from pymgcv.families import MVN, GauLSS, Poisson
from pymgcv.gam import BAM, GAM, AbstractGAM
from pymgcv.rpy_utils import data_to_rdf, to_py
from pymgcv.terms import L, Offset, S, T


def get_method_default(gam_type: type[AbstractGAM]):
    """Returns the pymgcv default fitting method for the gams."""
    sig = inspect.signature(gam_type.fit)
    return sig.parameters["method"].default


@cache
def get_test_data() -> pd.DataFrame:
    """Simple toy test dataset for testing most models.

    Used as default if data is not specified in test cases.
    """
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    x1 = x + rng.normal(loc=2, scale=1.5, size=n)
    group = pd.Categorical(rng.choice(["A", "B", "C"], size=n))
    group1 = pd.Categorical(rng.choice(["E", "F", "G"], size=n))
    group_effect = pd.Series(group).map({"A": 0.0, "B": 1.0, "C": -1.0}).to_numpy()
    y = 2 + 0.5 * x - 0.3 * x1 + group_effect + rng.normal(scale=0.5, size=n)
    y1 = np.sin(x) + 0.1 * x1**2 + group_effect + rng.normal(scale=0.3, size=n)
    pos_int = rng.poisson(lam=np.exp(0.2 * x + 0.1 * x1 + 0.3 * group_effect))
    pos_float = np.exp(
        1 + 0.4 * x - 0.2 * x1 + 0.5 * group_effect + rng.normal(scale=0.2, size=n),
    )
    obj = [("a", "b")] * n
    return pd.DataFrame(
        {
            "y": y,
            "y1": y1,
            "x": x,
            "x1": x1,
            "pos_int": pos_int,
            "pos_float": pos_float,
            "group": group,
            "group1": group1,
            "unused": obj,  # Check object column doesn't prevent conversion to R df
        },
    )


@dataclass(kw_only=True)
class GAMTestCase:  # GAM/BAM test cases
    """Test cases for GAMs.

    Note the method will often need to be specified in `mgcv_args`, as pymgcv changes
    the default to `REML` for `GAM`.
    """

    mgcv_args: str
    gam_model: AbstractGAM
    expected_predict_terms_structure: dict[str, list[str]]
    add_to_r_env: dict[str, ro.RObject] = field(default_factory=dict)
    data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] = field(
        default_factory=get_test_data,
    )
    fit_kwargs: dict[str, Any] = field(default_factory=dict)

    def mgcv_gam(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
    ) -> ro.ListVector:
        """Returns the mgcv gam object."""
        with ro.local_context() as env:
            env["data"] = data_to_rdf(data, include=self.gam_model.referenced_variables)
            for k, v in self.add_to_r_env.items():
                env[k] = v
            result = ro.r(self.mgcv_call)
            assert isinstance(result, ro.ListVector)
            return result

    @property
    def mgcv_call(self):
        """Returns the mgcv gam call as a string."""
        return f"{self.gam_model.__class__.__name__.lower()}({self.mgcv_args})"


# Factory functions for test cases
def linear_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    gam = gam_type({"y": L("x")})
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~x, data=data, method='{method}'",
        gam_model=gam,
        expected_predict_terms_structure={"y": ["L(x)", "Intercept"]},
    )


def categorical_linear_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~group, data=data, method='{method}'",
        gam_model=gam_type({"y": L("group")}),
        expected_predict_terms_structure={"y": ["L(group)", "Intercept"]},
    )


def smooth_1d_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x), data=data, method='{method}'",
        gam_model=gam_type({"y": S("x")}),
        expected_predict_terms_structure={"y": ["S(x)", "Intercept"]},
    )


def smooth_with_specified_knots(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    # We need exact same floating representation
    knots = to_py(ro.r("(0:4)/4"))
    return GAMTestCase(
        mgcv_args=f"y~s(x, k=5), data=data, method='{method}',knots=list(x=(0:4)/4)",
        gam_model=gam_type({"y": S("x", k=5)}),
        expected_predict_terms_structure={"y": ["S(x)", "Intercept"]},
        fit_kwargs={"knots": {"x": knots}},
    )


def smooth_2d_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x, x1), data=data, method='{method}'",
        gam_model=gam_type({"y": S("x", "x1")}),
        expected_predict_terms_structure={"y": ["S(x,x1)", "Intercept"]},
    )


def smooth_2d_gam_pass_to_s(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    basis = bs.ThinPlateSpline(max_knots=3, m=2)
    return GAMTestCase(
        mgcv_args=f"y~s(x,x1,m=2, xt=list(max.knots=3)), data=data, method='{method}'",
        gam_model=gam_type({"y": S("x", "x1", bs=basis)}),
        expected_predict_terms_structure={"y": ["S(x,x1)", "Intercept"]},
    )


def tensor_2d_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~te(x, x1), data=data, method='{method}'",
        gam_model=gam_type({"y": T("x", "x1")}),
        expected_predict_terms_structure={"y": ["T(x,x1)", "Intercept"]},
    )


def tensor_interaction_2d_gam_with_mc(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~ti(x, x1, mc=c(TRUE, FALSE)), data=data, method='{method}'",
        gam_model=gam_type(
            {"y": T("x", "x1", mc=[True, False], interaction_only=True)},
        ),
        expected_predict_terms_structure={"y": ["T(x,x1)", "Intercept"]},
    )


def random_effect_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x) + s(group, bs='re'), data=data, method='{method}'",
        gam_model=gam_type({"y": S("x") + S("group", bs=bs.RandomEffect())}),
        expected_predict_terms_structure={
            "y": ["S(x)", "S(group)", "Intercept"],
        },
    )


def categorical_interaction_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~group:group1, data=data, method='{method}'",
        gam_model=gam_type({"y": terms.Interaction("group", "group1")}),
        expected_predict_terms_structure={
            "y": ["Interaction(group,group1)", "Intercept"],
        },
    )


def multivariate_normal_gam(gam_type: type[AbstractGAM]):
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"list(y~s(x,k=5),y1~x),data=data,family=mvn(d=2),method='{method}'",
        gam_model=gam_type({"y": S("x", k=5), "y1": L("x")}, family=MVN(d=2)),
        expected_predict_terms_structure={
            "y": ["S(x)", "Intercept"],
            "y1": ["L(x)", "Intercept"],
        },
    )


def gaulss_gam(gam_type: type[AbstractGAM]):
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"list(y~s(x),~s(x1)),data=data,family=gaulss(),method='{method}'",
        gam_model=gam_type(
            {"y": S("x")},
            family_predictors={"scale": S("x1")},
            family=GauLSS(),
        ),
        expected_predict_terms_structure={
            "y": ["S(x)", "Intercept"],
            "scale": ["S(x1)", "Intercept"],
        },
    )


def offset_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x) + offset(x1), data=data, method='{method}'",
        gam_model=gam_type({"y": S("x") + Offset("x1")}),
        expected_predict_terms_structure={"y": ["S(x)", "Offset(x1)", "Intercept"]},
    )


def smooth_1d_by_categorical_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x, by=group), data=data, method='{method}'",
        gam_model=gam_type({"y": S("x", by="group")}),
        expected_predict_terms_structure={"y": ["S(x,by=group)", "Intercept"]},
    )


def smooth_1d_by_numeric_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x, by=x1), data=data, method='{method}'",
        gam_model=gam_type({"y": S("x", by="x1")}),
        expected_predict_terms_structure={"y": ["S(x,by=x1)", "Intercept"]},
    )


def tensor_2d_by_categorical_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~te(x,x1, by=group), data=data, method='{method}'",
        gam_model=gam_type({"y": T("x", "x1", by="group")}),
        expected_predict_terms_structure={
            "y": ["T(x,x1,by=group)", "Intercept"],
        },
    )


def tensor_2d_by_numeric_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~te(x,x1,by=pos_int), data=data, method='{method}'",
        gam_model=gam_type({"y": T("x", "x1", by="pos_int")}),
        expected_predict_terms_structure={
            "y": ["T(x,x1,by=pos_int)", "Intercept"],
        },
    )


def smooth_1d_random_wiggly_curve_gam(
    gam_type: type[AbstractGAM] = GAM,
) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x,group,bs='fs',xt=list(bs='cr')),data=data, method='{method}'",
        gam_model=gam_type(
            {"y": S("x", "group", bs=bs.FactorSmooth(bs.CubicSpline()))},
        ),
        expected_predict_terms_structure={"y": ["S(x,group)", "Intercept"]},
    )


def tensor_2d_random_wiggly_curve_gam(
    gam_type: type[AbstractGAM] = GAM,
) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~t(x,x1,group,bs='fs'),data=data, method='{method}'",
        gam_model=gam_type({"y": T("x", "x1", "group", bs=bs.FactorSmooth())}),
        expected_predict_terms_structure={
            "y": ["T(x,x1,group)", "Intercept"],
        },
    )


def poisson_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"pos_int~s(x), data=data, family=poisson, method='{method}'",
        gam_model=gam_type({"pos_int": S("x")}, family=Poisson()),
        expected_predict_terms_structure={"pos_int": ["S(x)", "Intercept"]},
    )


# def markov_random_field_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
#     mgcv = importr("mgcv")
#     polys = ro.packages.data(mgcv).fetch("columb.polys")["columb.polys"]
#     data = ro.packages.data(mgcv).fetch("columb")["columb"]
#     data = to_py(data)
#     polys_list = list([to_py(x) for x in polys.values()])
#     method = get_method_default(gam_type)
#     return GAMTestCase(
#         mgcv_args=(
#             "crime ~ s(district,bs='mrf',xt=list(polys=polys)), "
#             f"data=columb,method='REML', method='{method}'"
#         ),
#         gam_model=gam_type(
#             {"y": S("district", bs=MarkovRandomField(polys=polys_list))},
#         ),
#         data=data,
#         expected_predict_terms_structure={"crime": ["S(district)", "Intercept"]},
#         add_to_r_env={"polys": polys},
#     )


def linear_functional_smooth_1d_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    rng = np.random.default_rng(123)
    n = 200
    n_hours = 24
    hourly_x = rng.lognormal(size=(n, n_hours))
    y = sum(np.sqrt(col) for col in hourly_x.T) + rng.normal(scale=0.1, size=n)
    data = {"y": y, "hourly_x": hourly_x}
    gam = gam_type({"y": S("hourly_x")})

    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y ~ s(hourly_x), data=data, method='{method}'",
        gam_model=gam,
        data=data,
        expected_predict_terms_structure={"y": ["S(hourly_x)", "Intercept"]},
    )


def linear_functional_tensor_2d_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    rng = np.random.default_rng(123)
    n = 200
    n_times = 4
    x0 = rng.lognormal(size=(n, n_times))
    x1 = rng.lognormal(size=(n, n_times))

    def _true_fn(x0, x1):
        return np.sqrt(x0) + np.sqrt(x1)

    y = sum(
        _true_fn(x0_col, x1_col) for x0_col, x1_col in zip(x0.T, x1.T, strict=True)
    ) + rng.normal(scale=0.1, size=n)
    data = {"y": y, "x0": x0, "x1": x1}
    gam = gam_type({"y": T("x0", "x1")})

    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y ~ te(x0, x1), data=data, method='{method}'",
        gam_model=gam,
        data=data,
        expected_predict_terms_structure={"y": ["T(x0,x1)", "Intercept"]},
    )


def spline_test_cases() -> dict[str, GAMTestCase]:  # TODO maybe add other basis types
    bases = {
        "bs='tp'": bs.ThinPlateSpline(),
        "bs='cr'": bs.CubicSpline(),
        "bs='ds', m=c(1,0)": bs.DuchonSpline(m=1),
        "bs='bs'": bs.BSpline(),
        "bs='bs', m=c(3,2,1)": bs.BSpline(degree=3, penalty_orders=[2, 1]),
        "bs='ps', m=c(2,2)": bs.PSpline(degree=3, penalty_order=2),
    }
    test_cases = {}
    for k, v in bases.items():
        test_cases[k] = GAMTestCase(
            mgcv_args=f"y ~ s(x, {k}), data=data, method='REML'",
            gam_model=GAM({"y": S("x", bs=v)}),
            expected_predict_terms_structure={"y": ["S(x)", "Intercept"]},
        )
    return test_cases


def get_test_cases() -> dict[str, GAMTestCase]:
    supported_types_and_cases = [
        (
            (GAM, BAM),
            [
                linear_gam,
                categorical_linear_gam,
                smooth_1d_gam,
                smooth_2d_gam,
                smooth_2d_gam_pass_to_s,
                smooth_with_specified_knots,
                tensor_2d_gam,
                tensor_interaction_2d_gam_with_mc,
                random_effect_gam,
                smooth_1d_random_wiggly_curve_gam,
                categorical_interaction_gam,
                offset_gam,
                smooth_1d_by_categorical_gam,
                smooth_1d_by_numeric_gam,
                tensor_2d_by_categorical_gam,
                tensor_2d_by_numeric_gam,
                poisson_gam,
                linear_functional_smooth_1d_gam,
                linear_functional_tensor_2d_gam,
                # markov_random_field_gam  # TODO: Uncomment when ready
            ],
        ),
        (
            (GAM,),
            [
                multivariate_normal_gam,
                gaulss_gam,
            ],
        ),
    ]

    test_cases = {}
    for gam_types, cases in supported_types_and_cases:
        for gam_type in gam_types:
            for case in cases:
                test_cases[f"{gam_type.__name__} - {case.__name__}"] = case(gam_type)

    return test_cases | spline_test_cases()
