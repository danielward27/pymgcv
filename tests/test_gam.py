from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

from pymgcv import Smooth, TensorSmooth
from pymgcv.bases import MarkovRandomField, RandomEffect
from pymgcv.converters import data_to_rdf, rlistvec_to_dict, to_py
from pymgcv.gam import FittedGAM, gam, terms_to_formula
from pymgcv import terms

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
    test_id: str
    mgcv_call: str
    add_to_r_env: dict[str, ro.RObject]

    @abstractmethod
    def get_data(self) -> pd.DataFrame | dict[str, np.ndarray | pd.Series]:
        """Returns data dictionary."""
        pass

    @abstractmethod
    def pymgcv_gam(
        self,
        data: pd.DataFrame | dict[str, np.ndarray | pd.Series],
    ) -> FittedGAM:
        """Returns pymgcv gam."""
        pass

    def mgcv_gam(self, data: pd.DataFrame | dict[str, np.ndarray | pd.Series]):
        with ro.local_context() as env:
            env["data"] = data_to_rdf(data)
            for k, v in self.add_to_r_env.items():
                env[k] = v
            return ro.r(self.mgcv_call)


def get_test_cases():

    class SimpleGAM(AbstractTestCase):
        test_id = "simple"
        mgcv_call = """
        gam(y~x0+s(x1)+s(x2)+s(x3), data=data)
        """
        add_to_r_env = {}

        def get_data(self) -> pd.DataFrame | dict[str, np.ndarray | pd.Series]:
            data = to_py(mgcv.gamSim(5, n=200, scale=2))
            return {k: v.to_numpy() for k, v in data.items()}

        def pymgcv_gam(self, data) -> FittedGAM:
            return gam(
                dependent="y",
                terms=(terms.Linear("x0"), terms.Smooth("x1"), terms.Smooth("x2"), terms.Smooth("x3")),
                data=data,
            )

    class TensorGAM(AbstractTestCase):
        test_id = "simple-tensor"
        mgcv_call = """
        gam(y~te(x0, x1), data=data)
        """
        add_to_r_env = {}

        def get_data(self) -> pd.DataFrame | dict[str, np.ndarray | pd.Series]:
            rng = np.random.default_rng(seed=42)
            n = 200
            x0 = rng.uniform(0, 1, n)
            x1 = rng.uniform(0, 1, n)
            y = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1) + rng.normal(0, 0.2, n)
            return {"y": y, "x0": x0, "x1": x1}

        def pymgcv_gam(self, data) -> FittedGAM:
            return gam(
                dependent="y",
                terms=(TensorSmooth("x0", "x1"),),
                data=data,
            )

    class FactorGAM(AbstractTestCase):  # TODO fix
        test_id = "simple-factor"
        mgcv_call = "gam(y~s(x) + s(group, bs='re'), data=data)"
        add_to_r_env = {}

        def get_data(self) -> pd.DataFrame | dict[str, np.ndarray | pd.Series]:
            rng = np.random.default_rng(1)
            n = 50
            group = pd.Series(pd.Categorical(rng.choice(["a", "b", "c"], n)))
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

    class MarkovRandomFieldGAM(AbstractTestCase):
        test_id = "markov-random-field-gam"
        mgcv_call = """
        gam(crime ~ s(district,bs="mrf",xt=list(polys=polys)),data=columb,method="REML")
        """
        add_to_r_env: dict

        def __init__(self):
            polys = ro.packages.data(mgcv).fetch("columb.polys")["columb.polys"]
            self.add_to_r_env = {"polys": polys}

        def get_data(self) -> dict[str, Any]:

            data = ro.packages.data(mgcv).fetch("columb")["columb"]
            data = to_py(data)

            rng = np.random.default_rng(seed=42)
            n = 200
            x0 = rng.uniform(0, 1, n)
            x1 = rng.uniform(0, 1, n)
            y = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1) + rng.normal(0, 0.2, n)
            return {"y": y, "x0": x0, "x1": x1}

        def pymgcv_gam(self, data) -> FittedGAM:
            polys = list(rlistvec_to_dict(self.add_to_r_env["polys"]).values())

            return gam(
                dependent="y",
                terms=(Smooth("district", bs=MarkovRandomField(polys=polys)),),
                data=data,
            )

    # TODO:
    # multivariate gam
    # list formulas: gam(list(y0~s(x0)+s(x1),y1~s(x2)+s(x3)),family=mvn(d=2),data=dat)
    # Offset gam

    return [SimpleGAM(), TensorGAM(), FactorGAM()]


test_cases = get_test_cases()


@pytest.mark.parametrize("test_case", test_cases, ids=[t.test_id for t in test_cases])
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
