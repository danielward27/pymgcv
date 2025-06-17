import numpy as np


def _get_intercepts_and_se(rgam):
    coef = rgam.rx2("coefficients")
    vp_diag = np.diag(np.array(rgam.rx2("Vp")))
    coef_names = list(coef.names)
    assert len(coef_names) == len(vp_diag)
    intercept_names = [n for n in coef_names if n.startswith("(Intercept)")]

    return {
        name: {
            "fit": coef[coef_names.index(name)],
            "se": np.sqrt(vp_diag[coef_names.index(name)]).item(),
        }
        for name in intercept_names
    }
