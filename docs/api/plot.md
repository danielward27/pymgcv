# Plotting

`pymgcv.plot` provides plotting utilities for visualizing GAM models. Plotting is performed using matplotlib.
Across the package and examples, we use the import convention
```python
import pymgcv.plot as gplt
```

- For an overall plot of the partial effects of a gam model, use [`plot`][pymgcv.plot.plot].
- For more fine control over the plotting, specific terms can be plotted onto a single matplotlib axis, using the functions:
    * [`continuous_1d`][pymgcv.plot.continuous_1d]
    * [`continuous_2d`][pymgcv.plot.continuous_2d]
    * [`categorical`][pymgcv.plot.categorical]
    * [`random_effect`][pymgcv.plot.random_effect]

::: pymgcv.plot.plot
::: pymgcv.plot.continuous_1d
::: pymgcv.plot.continuous_2d
::: pymgcv.plot.categorical
::: pymgcv.plot.random_effect
::: pymgcv.plot.qq
::: pymgcv.plot.residuals_vs_linear_predictor
::: pymgcv.plot.hexbin_residuals
