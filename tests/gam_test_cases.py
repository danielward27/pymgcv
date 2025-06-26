"""A collection of GAM test cases."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from pymgcv import terms
from pymgcv.basis_functions import (
    CubicSpline,
    MarkovRandomField,
    RandomEffect,
    RandomWigglyCurve,
)
from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py
from pymgcv.gam import GAM
from pymgcv.terms import Linear as L
from pymgcv.terms import Smooth as S
from pymgcv.terms import TensorSmooth as T


@dataclass
class GAMTestCase:
    mgcv_call: str
    gam_model: GAM
    data: pd.DataFrame
    expected_predict_terms_structure: dict[str, list[str]]
    add_to_r_env: dict[str, ro.RObject] = field(default_factory=dict)

    def mgcv_gam(self, data: pd.DataFrame):
        with ro.local_context() as env:
            env["data"] = data_to_rdf(data)
            for k, v in self.add_to_r_env.items():
                env[k] = v
            return ro.r(self.mgcv_call)


# Factory functions for test cases
def create_single_linear_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame({"x": rng.uniform(0, 1, 200), "y": rng.normal(0, 0.2, 200)})
    return GAMTestCase(
        mgcv_call="gam(y~x, data=data)",
        gam_model=GAM({"y": L("x")}),
        data=data,
        expected_predict_terms_structure={"y": ["Linear(x)", "intercept"]},
    )


def create_single_categorical_linear_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "group": pd.Series(rng.choice(["A", "B", "C"], 200), dtype="category"),
            "y": rng.normal(0, 0.2, 200),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(y~group, data=data)",
        gam_model=GAM({"y": L("group")}),
        data=data,
        expected_predict_terms_structure={"y": ["Linear(group)", "intercept"]},
    )


def create_single_smooth_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame({"x": rng.uniform(0, 1, 200), "y": rng.normal(0, 0.2, 200)})
    return GAMTestCase(
        mgcv_call="gam(y~s(x), data=data)",
        gam_model=GAM({"y": S("x")}),
        data=data,
        expected_predict_terms_structure={"y": ["Smooth(x)", "intercept"]},
    )


def create_single_tensor_smooth_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    n = 200
    x0, x1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1) + rng.normal(0, 0.2, n)
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})
    return GAMTestCase(
        mgcv_call="gam(y~te(x0, x1), data=data)",
        gam_model=GAM({"y": [T("x0", "x1")]}),
        data=data,
        expected_predict_terms_structure={"y": ["TensorSmooth(x0,x1)", "intercept"]},
    )


def create_single_random_effect() -> GAMTestCase:
    rng = np.random.default_rng(42)
    n = 50
    group = pd.Series(rng.choice(["a", "b", "c"], n), dtype="category")
    x = np.linspace(0, 10, n)
    y = np.sin(x) + group.cat.codes + rng.normal(scale=0.1, size=n)
    data = pd.DataFrame({"group": group, "x": x, "y": y})
    return GAMTestCase(
        mgcv_call="gam(y~s(x) + s(group, bs='re'), data=data)",
        gam_model=GAM({"y": S("x") + S("group", bs=RandomEffect())}),
        data=data,
        expected_predict_terms_structure={
            "y": ["Smooth(x)", "Smooth(group)", "intercept"],
        },
    )


def create_single_factor_interaction_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "a": pd.Categorical(rng.choice(["A", "B"], size=200, replace=True)),
            "b": pd.Categorical(rng.choice(["C", "D", "E"], size=200, replace=True)),
            "y": rng.uniform(0, 1, size=200),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(y~a:b, data=data)",
        gam_model=GAM({"y": [terms.Interaction("a", "b")]}),
        data=data,
        expected_predict_terms_structure={"y": ["Interaction(a,b)", "intercept"]},
    )


def create_simple_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "x0": rng.uniform(0, 1, 200),
            "x1": rng.uniform(0, 1, 200),
            "x2": rng.uniform(0, 1, 200),
            "y": rng.normal(0, 0.2, 200),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(y~x0+s(x1)+s(x2), data=data)",
        gam_model=GAM({"y": L("x0") + S("x1") + S("x2")}),
        data=data,
        expected_predict_terms_structure={
            "y": ["Linear(x0)", "Smooth(x1)", "Smooth(x2)", "intercept"],
        },
    )


def create_multivariate_multi_formula():
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "x": rng.uniform(0, 1, 200),
            "y0": rng.normal(0, 0.2, 200),
            "y1": rng.normal(0, 0.2, 200),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(list(y0 ~ s(x, k=5), y1 ~ x), data=data, family=mvn(d=2))",
        gam_model=GAM({"y0": S("x", k=5), "y1": L("x")}, family="mvn(d=2)"),
        data=data,
        expected_predict_terms_structure={
            "y0": ["Smooth(x)", "intercept"],
            "y1": ["Linear(x)", "intercept"],
        },
    )


def create_location_scale_multi_formula():
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "x0": rng.uniform(0, 1, 200),
            "x1": rng.normal(0, 0.2, 200),
            "y": rng.normal(0, 0.2, 200),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(list(y ~ s(x0), ~ s(x1)), data=data, family=gaulss())",
        gam_model=GAM(
            {"y": S("x0")},
            family_predictors={"log_scale": S("x1")},
            family="gaulss()",
        ),
        data=data,
        expected_predict_terms_structure={
            "y": ["Smooth(x0)", "intercept"],
            "log_scale": ["Smooth(x1)", "intercept"],
        },
    )


def create_offset_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "x": rng.uniform(0, 1, 200),
            "z": rng.uniform(0, 1, 200),
            "y": rng.normal(0, 0.2, 200),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(y~s(x) + offset(z), data=data)",
        gam_model=GAM({"y": S("x") + terms.Offset("z")}),
        data=data,
        expected_predict_terms_structure={"y": ["Smooth(x)", "Offset(z)", "intercept"]},
    )


def create_smooth_by_categorical_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "x": rng.standard_normal(100),
            "group": pd.Categorical(rng.choice(["a", "b", "c"], 100)),
            "y": rng.standard_normal(100),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(y~s(x, by=group), data=data)",
        gam_model=GAM({"y": S("x", by="group")}),
        data=data,
        expected_predict_terms_structure={"y": ["Smooth(x,by=group)", "intercept"]},
    )


def create_smooth_by_numeric_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "x": rng.standard_normal(100),
            "by_var": rng.standard_normal(100),
            "y": rng.standard_normal(100),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(y~s(x, by=by_var), data=data)",
        gam_model=GAM({"y": S("x", by="by_var")}),
        data=data,
        expected_predict_terms_structure={"y": ["Smooth(x,by=by_var)", "intercept"]},
    )


def create_tensor_by_categorical_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "x0": rng.standard_normal(100),
            "x1": rng.standard_normal(100),
            "group": pd.Categorical(rng.choice(["a", "b", "c"], 100)),
            "y": rng.standard_normal(100),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(y~te(x0,x1, by=group), data=data)",
        gam_model=GAM({"y": T("x0", "x1", by="group")}),
        data=data,
        expected_predict_terms_structure={
            "y": ["TensorSmooth(x0,x1,by=group)", "intercept"],
        },
    )


def create_random_wiggly_curve_smooth_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "x": rng.standard_normal(100),
            "group": pd.Categorical(rng.choice(["a", "b", "c"], 100)),
            "y": rng.standard_normal(100),
        },
    )
    bs = RandomWigglyCurve(CubicSpline())
    return GAMTestCase(
        mgcv_call="gam(y~s(x,group,bs='fs',xt=list(bs='cr')),data=data)",
        gam_model=GAM({"y": S("x", "group", bs=bs)}),
        data=data,
        expected_predict_terms_structure={"y": ["Smooth(x,group)", "intercept"]},
    )


def create_tensor_by_numeric_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "x0": rng.standard_normal(100),
            "x1": rng.standard_normal(100),
            "by_var": rng.standard_normal(100),
            "y": rng.standard_normal(100),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(y~te(x0,x1,by=by_var), data=data)",
        gam_model=GAM({"y": T("x0", "x1", by="by_var")}),
        data=data,
        expected_predict_terms_structure={
            "y": ["TensorSmooth(x0,x1,by=by_var)", "intercept"],
        },
    )


def create_poisson_gam() -> GAMTestCase:
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame(
        {
            "x": rng.standard_normal(100),
            "counts": rng.poisson(size=100),
        },
    )
    return GAMTestCase(
        mgcv_call="gam(counts~s(x), data=data, family=poisson)",
        gam_model=GAM({"counts": S("x")}, family="poisson"),
        data=data,
        expected_predict_terms_structure={"counts": ["Smooth(x)", "intercept"]},
    )


def create_markov_random_field_gam() -> GAMTestCase:
    mgcv = importr("mgcv")
    polys = ro.packages.data(mgcv).fetch("columb.polys")["columb.polys"]
    data = ro.packages.data(mgcv).fetch("columb")["columb"]
    data = to_py(data)
    polys_list = list(rlistvec_to_dict(polys).values())
    return GAMTestCase(
        mgcv_call="gam(crime ~ s(district,bs='mrf',xt=list(polys=polys)),data=columb,method='REML')",
        gam_model=GAM({"y": S("district", bs=MarkovRandomField(polys=polys_list))}),
        data=data,
        expected_predict_terms_structure={"crime": ["Smooth(district)", "intercept"]},
        add_to_r_env={"polys": polys},
    )


def get_test_cases() -> list[GAMTestCase]:
    return [
        create_single_linear_gam(),
        create_single_categorical_linear_gam(),
        create_single_smooth_gam(),
        create_single_tensor_smooth_gam(),
        create_single_random_effect(),
        create_random_wiggly_curve_smooth_gam(),
        create_single_factor_interaction_gam(),
        create_simple_gam(),
        create_multivariate_multi_formula(),
        create_location_scale_multi_formula(),
        create_offset_gam(),
        create_smooth_by_categorical_gam(),
        create_smooth_by_numeric_gam(),
        create_tensor_by_categorical_gam(),
        create_tensor_by_numeric_gam(),
        create_poisson_gam(),
        # create_markov_random_field_gam()  # TODO: Uncomment when ready
    ]


# """A collection of GAM test cases."""

# from abc import ABC, abstractmethod
# from dataclasses import dataclass, field

# import numpy as np
# import pandas as pd
# import rpy2.robjects as ro

# from pymgcv import terms
# from pymgcv.basis_functions import (
#     CubicSpline,
#     MarkovRandomField,
#     RandomEffect,
#     RandomWigglyCurve,
# )
# from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py
# from pymgcv.gam import GAM, FittedGAM
# from pymgcv.terms import Linear as L
# from pymgcv.terms import Smooth as S
# from pymgcv.terms import TensorSmooth as T


# @dataclass
# class TestCase:
#     mgcv_call: str
#     gam_model: GAM
#     data: pd.DataFrame
#     expected_predict_terms_structure: dict[str, list[str]]
#     add_to_r_env: dict[str, ro.RObject] = field(default_factory=dict)

#     def mgcv_gam(self, data: pd.DataFrame):
#         with ro.local_context() as env:
#             env["data"] = data_to_rdf(data)
#             for k, v in self.add_to_r_env.items():
#                 env[k] = v
#             return ro.r(self.mgcv_call)


# class AbstractTestCase(ABC):
#     mgcv_call: str
#     add_to_r_env: dict[str, ro.RObject]
#     expected_predict_terms_structure: dict[str, list[str]]

#     @abstractmethod
#     def get_data(self) -> pd.DataFrame:
#         """Returns data dictionary."""
#         pass

#     @abstractmethod
#     def pymgcv_gam(self, data) -> FittedGAM:
#         pass


# def get_test_cases():
#     class SingleLinearGAM(AbstractTestCase):
#         mgcv_call = """
#                 gam(y~x, data=data)
#                 """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {"y": ["Linear(x)", "intercept"]}

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x": rng.uniform(0, 1, 200),
#                     "y": rng.normal(0, 0.2, 200),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"y": L("x")})
#             return model.fit(data=data)

#     class SingleCategoricalLinearGAM(AbstractTestCase):
#         mgcv_call = """
#                 gam(y~group, data=data)
#                 """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {"y": ["Linear(group)", "intercept"]}

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "group": pd.Series(
#                         rng.choice(["A", "B", "C"], 200),
#                         dtype="category",
#                     ),
#                     "y": rng.normal(0, 0.2, 200),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"y": L("group")})
#             return model.fit(data=data)

#     class SingleSmoothGAM(AbstractTestCase):
#         mgcv_call = """
#             gam(y~s(x), data=data)
#             """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {"y": ["Smooth(x)", "intercept"]}

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             x = rng.uniform(0, 1, 200)
#             return pd.DataFrame(
#                 {
#                     "x": x,
#                     "y": rng.normal(0, 0.2, 200),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"y": S("x")})
#             return model.fit(data=data)

#     class SingleTensorSmoothGAM(AbstractTestCase):
#         mgcv_call = """
#         gam(y~te(x0, x1), data=data)
#         """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {"y": ["TensorSmooth(x0,x1)", "intercept"]}

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             n = 200
#             x0 = rng.uniform(0, 1, n)
#             x1 = rng.uniform(0, 1, n)
#             y = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1) + rng.normal(0, 0.2, n)
#             return pd.DataFrame({"y": y, "x0": x0, "x1": x1})

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM(
#                 {"y": [T("x0", "x1")]},
#             )
#             return model.fit(data=data)

#     class SingleRandomEffect(AbstractTestCase):
#         mgcv_call = "gam(y~s(x) + s(group, bs='re'), data=data)"
#         add_to_r_env = {}
#         expected_predict_terms_structure = {
#             "y": ["Smooth(x)", "Smooth(group)", "intercept"],
#         }

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(42)
#             n = 50
#             group = pd.Series(
#                 pd.Categorical(
#                     rng.choice(["a", "b", "c"], n),
#                     categories=["a", "b", "c"],
#                 ),
#             )
#             x = np.linspace(0, 10, n)
#             y = np.sin(x) + group.cat.codes + rng.normal(scale=0.1, size=n)
#             return pd.DataFrame(
#                 {
#                     "group": group,
#                     "x": x,
#                     "y": y,
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"y": S("x") + S("group", bs=RandomEffect())})
#             return model.fit(data=data)

#     class SingleFactorInteractionGAM(AbstractTestCase):
#         mgcv_call = """
#             gam(y~a:b, data=data)
#             """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {"y": ["Interaction(a,b)", "intercept"]}

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "a": pd.Categorical(
#                         rng.choice(["A", "B"], size=200, replace=True),
#                     ),
#                     "b": pd.Categorical(
#                         rng.choice(["C", "D", "E"], size=200, replace=True),
#                     ),
#                     "y": rng.uniform(0, 1, size=200),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM(
#                 {"y": [terms.Interaction("a", "b")]},
#             )
#             return model.fit(data=data)

#     class SimpleGAM(AbstractTestCase):
#         mgcv_call = """
#             gam(y~x0+s(x1)+s(x2), data=data)
#             """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {
#             "y": ["Linear(x0)", "Smooth(x1)", "Smooth(x2)", "intercept"],
#         }

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x0": rng.uniform(0, 1, 200),
#                     "x1": rng.uniform(0, 1, 200),
#                     "x2": rng.uniform(0, 1, 200),
#                     "y": rng.normal(0, 0.2, 200),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"y": L("x0") + S("x1") + S("x2")})
#             return model.fit(data=data)

#     class MultivariateMultiFormula(AbstractTestCase):
#         mgcv_call = """
#             gam(
#                 list(
#                     y0 ~ s(x, k=5),
#                     y1 ~ x
#                 ),
#                 data=data,
#                 family=mvn(d=2)
#             )
#             """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {
#             "y0": ["Smooth(x)", "intercept"],
#             "y1": ["Linear(x)", "intercept"],
#         }

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x": rng.uniform(0, 1, 200),
#                     "y0": rng.normal(0, 0.2, 200),
#                     "y1": rng.normal(0, 0.2, 200),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM(
#                 {"y0": S("x", k=5), "y1": L("x")},
#                 family="mvn(d=2)",
#             )
#             return model.fit(data=data)

#     class LocationScaleMultiFormula(AbstractTestCase):
#         mgcv_call = """
#             gam(
#                 list(
#                     y ~ s(x0),
#                     ~ s(x1)
#                 ),
#                 data=data,
#                 family=gaulss()
#             )
#             """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {
#             "y": ["Smooth(x0)", "intercept"],
#             "log_scale": ["Smooth(x1)", "intercept"],
#         }

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x0": rng.uniform(0, 1, 200),
#                     "x1": rng.normal(0, 0.2, 200),
#                     "y": rng.normal(0, 0.2, 200),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM(
#                 {"y": S("x0")},
#                 family_predictors={"log_scale": S("x1")},
#                 family="gaulss()",
#             )
#             return model.fit(data=data)

#     class OffsetGAM(AbstractTestCase):
#         mgcv_call = """
#             gam(y~s(x) + offset(z), data=data)
#             """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {
#             "y": ["Smooth(x)", "Offset(z)", "intercept"],
#         }

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x": rng.uniform(0, 1, 200),
#                     "z": rng.uniform(0, 1, 200),
#                     "y": rng.normal(0, 0.2, 200),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"y": S("x") + terms.Offset("z")})
#             return model.fit(data=data)

#     class SmoothByCategoricalGAM(AbstractTestCase):
#         mgcv_call = """
#                 gam(y~s(x, by=group), data=data)
#                 """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {"y": ["Smooth(x,by=group)", "intercept"]}

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x": rng.standard_normal(100),
#                     "group": pd.Categorical(rng.choice(["a", "b", "c"], 100)),
#                     "y": rng.standard_normal(100),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"y": S("x", by="group")})
#             return model.fit(data=data)

#     class SmoothByNumericGAM(AbstractTestCase):
#         mgcv_call = """
#                     gam(y~s(x, by=by_var), data=data)
#                     """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {"y": ["Smooth(x,by=by_var)", "intercept"]}

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x": rng.standard_normal(100),
#                     "by_var": rng.standard_normal(100),
#                     "y": rng.standard_normal(100),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"y": S("x", by="by_var")})
#             return model.fit(data=data)

#     class TensorByCategoricalGAM(AbstractTestCase):
#         mgcv_call = """
#                 gam(y~te(x0,x1, by=group), data=data)
#                 """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {
#             "y": ["TensorSmooth(x0,x1,by=group)", "intercept"],
#         }

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x0": rng.standard_normal(100),
#                     "x1": rng.standard_normal(100),
#                     "group": pd.Categorical(rng.choice(["a", "b", "c"], 100)),
#                     "y": rng.standard_normal(100),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"y": T("x0", "x1", by="group")})
#             return model.fit(data=data)

#     class RandomWigglyCurveSmoothGAM(AbstractTestCase):
#         mgcv_call = """
#             gam(y~s(x,group,bs="fs",xt=list(bs="cr")),data=data)
#         """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {
#             "y": ["Smooth(x,group)", "intercept"],
#         }

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x": rng.standard_normal(100),
#                     "group": pd.Categorical(rng.choice(["a", "b", "c"], 100)),
#                     "y": rng.standard_normal(100),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             bs = RandomWigglyCurve(CubicSpline())
#             model = GAM({"y": S("x", "group", bs=bs)})
#             return model.fit(data=data)

#     class TensorByNumericGAM(AbstractTestCase):
#         mgcv_call = """
#                     gam(y~te(x0,x1,by=by_var), data=data)
#                     """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {
#             "y": ["TensorSmooth(x0,x1,by=by_var)", "intercept"],
#         }

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x0": rng.standard_normal(100),
#                     "x1": rng.standard_normal(100),
#                     "by_var": rng.standard_normal(100),
#                     "y": rng.standard_normal(100),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"y": T("x0", "x1", by="by_var")})
#             return model.fit(data=data)

#     class PoissonGAM(AbstractTestCase):
#         mgcv_call = """
#                     gam(counts~s(x), data=data, family=poisson)
#                     """
#         add_to_r_env = {}
#         expected_predict_terms_structure = {"counts": ["Smooth(x)", "intercept"]}

#         def get_data(self) -> pd.DataFrame:
#             rng = np.random.default_rng(seed=42)
#             return pd.DataFrame(
#                 {
#                     "x": rng.standard_normal(100),
#                     "counts": rng.poisson(size=100),
#                 },
#             )

#         def pymgcv_gam(self, data) -> FittedGAM:
#             model = GAM({"counts": S("x")}, family="poisson")
#             return model.fit(data=data)

#     class MarkovRandomFieldGAM(AbstractTestCase):
#         mgcv_call = """
#         gam(crime ~ s(district,bs="mrf",xt=list(polys=polys)),data=columb,method="REML")
#         """
#         add_to_r_env: dict
#         expected_predict_terms_structure = {"crime": ["Smooth(district)", "intercept"]}

#         def __init__(self):
#             polys = ro.packages.data(mgcv).fetch("columb.polys")["columb.polys"]
#             self.add_to_r_env = {"polys": polys}

#         def get_data(self) -> pd.DataFrame:
#             data = ro.packages.data(mgcv).fetch("columb")["columb"]
#             data = to_py(data)

#             rng = np.random.default_rng(seed=42)
#             n = 200
#             x0 = rng.uniform(0, 1, n)
#             x1 = rng.uniform(0, 1, n)
#             y = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1) + rng.normal(0, 0.2, n)
#             return pd.DataFrame({"y": y, "x0": x0, "x1": x1})

#         def pymgcv_gam(self, data) -> FittedGAM:
#             polys = list(rlistvec_to_dict(self.add_to_r_env["polys"]).values())
#             model = GAM(
#                 {"y": S("district", bs=MarkovRandomField(polys=polys))},
#             )
#             return model.fit(data=data)

#     # TODO MRF not included yet!
#     # TODO:
#     # multivariate gam

#     return [
#         SingleLinearGAM(),
#         SingleCategoricalLinearGAM(),
#         SingleSmoothGAM(),
#         SingleTensorSmoothGAM(),
#         SingleRandomEffect(),
#         RandomWigglyCurveSmoothGAM(),
#         SingleFactorInteractionGAM(),
#         SimpleGAM(),
#         MultivariateMultiFormula(),
#         LocationScaleMultiFormula(),
#         OffsetGAM(),
#         SmoothByCategoricalGAM(),
#         SmoothByNumericGAM(),
#         TensorByCategoricalGAM(),
#         TensorByNumericGAM(),
#         PoissonGAM(),
#     ]
