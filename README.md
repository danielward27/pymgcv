# pymgcv: Generalized Additive Models in Python

**pymgcv** provides a Pythonic interface to R's powerful [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html) library for fitting Generalized Additive Models (GAMs). It combines the flexibility and statistical rigor of mgcv with the convenience of Python's data science ecosystem.

Currently in development. As this is a multilanguage project (R and Python), we use
[pixi](https://pixi.sh/latest/), a package management tool which supports this (via
conda). For development, the ``pymgcv`` can be installed by installing
[pixi](https://pixi.sh/latest/) and running:

```bash
git clone https://github.com/danielward27/pymgcv.git
cd pymgcv
pixi shell --environment=dev
```
### Documentation
[Documentation](https://danielward27.github.io/pymgcv/)

### Installation options
.

### Simple example
```python
import pandas as pd
import numpy as np
from pymgcv.gam import GAM
from pymgcv.terms import S, T, L
from pymgcv.plot import plot_gam
import matplotlib.pyplot as plt

# Generate sample data with non-linear relationship
np.random.seed(42)
n = 100
x0 = np.random.uniform(-1, 1, n)
x1 = np.random.uniform(-1, 1, n)
y = 0.5 * x0 + np.sin(np.pi * x1) + np.random.normal(0, 0.5, n)
data = pd.DataFrame({'x0': x0, 'x1': x1, 'y': y})

# Define model: linear effect of x0, smooth function of x1
gam = GAM({'y': L('x0') + S('x1')})

gam.fit(data)
plot_gam(gam, residuals=True)
```
