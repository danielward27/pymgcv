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

### Installation options
Installing the python package only includes the python package and dependencies. This means an R installation with ``mgcv`` is also required.
Conda and pixi provide two convenient options for handling both Python and R dependencies:

Using conda:
- Install ``conda`` e.g. [miniforge](https://github.com/conda-forge/miniforge)
- Install [uv](https://github.com/astral-sh/uv) (or use pip).

```bash
conda create --name my_env python r-base r-mgcv
conda activate my_env
uv pip install pymgcv
```

Using pixi:
- Install [pixi](https://github.com/prefix-dev/pixi)
```bash
pixi init
pixi add python r-base r-mgcv
pixi add --pypi pymgcv
pixi shell
```

Using either method the below example should now run e.g. in the terminal after running ``python``,
or in an IDE after selecting the pixi/conda environment.

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
model = GAM({'y': L('x0') + S('x1')})

# Fit the model
model = model.fit(data)
plot_gam(fit=model, residuals=True)
plt.show()
```
