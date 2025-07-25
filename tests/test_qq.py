import numpy as np
import pandas as pd

from pymgcv.gam import GAM
from pymgcv.qq import qq_uniform
from pymgcv.terms import L, S


def test_qq_uniform():
    """Test that qq_uniform runs without error and returns expected structure."""
    rng = np.random.default_rng(42)
    n = 100
    x0, x1, x2, x3 = [rng.uniform(-1, 1, n) for _ in range(4)]
    y = (
        0.5 * x0
        + np.sin(np.pi * x1)
        + np.cos(np.pi * x2) * np.sin(np.pi * x3)
        + rng.normal(0, 0.3, n)
    )
    data = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3, "y": y})

    gam = GAM({"y": L("x0") + S("x1") + S("x2", "x3")})
    gam.fit(data)

    result = qq_uniform(gam, n=5)
    assert isinstance(result, pd.DataFrame)
    assert "theoretical" in result.columns
    assert "residuals" in result.columns
    assert len(result) == n
    assert np.all(np.isfinite(result["theoretical"]))
    assert np.all(np.isfinite(result["residuals"]))
