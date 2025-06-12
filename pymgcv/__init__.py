"""pymgcv: Generalized Additive Models in Python.

pymgcv provides a Pythonic interface to R's mgcv library for fitting Generalized
Additive Models (GAMs). It combines the statistical rigor and flexibility of mgcv
with the convenience of Python's data science ecosystem.

Key Features:
- Comprehensive GAM support with various smooth terms
- Multiple basis functions (thin plate splines, cubic splines, B-splines, etc.)
- Flexible model specification for univariate and multivariate responses
- Built-in plotting and visualization capabilities
- Statistical inference with confidence intervals and model diagnostics

Quick Start:
    ```python
    import pandas as pd
    import numpy as np
    from pymgcv import gam, ModelSpecification, Smooth, Linear

    # Generate sample data
    n = 200
    x = np.random.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)
    data = pd.DataFrame({'x': x, 'y': y})

    # Define and fit model
    spec = ModelSpecification(response_predictors={'y': [Smooth('x')]})
    model = gam(spec, data)

    # Print summary and make predictions
    print(model.summary())
    predictions = model.predict(data)
    ```

Main Components:
- gam: Main function for fitting GAM models
- ModelSpecification: Class for defining model structure
- Term types: Linear, Smooth, TensorSmooth, Interaction, Offset
- Basis functions: Various spline and basis types in pymgcv.bases
- Plotting utilities: Visualization functions in pymgcv.plot

See the documentation for detailed examples and API reference.
"""

from .gam import ModelSpecification, gam
from .terms import Interaction, Linear, Offset, Smooth, TensorSmooth

__all__ = [
    # Core functionality
    "gam",
    "ModelSpecification",
    # Term types
    "Linear",
    "Smooth",
    "TensorSmooth",
    "Interaction",
    "Offset",
]

# Version information
__version__ = "0.0.0"
__author__ = "Daniel Ward"
__email__ = "danielward27@outlook.com"
