"""Plotting utilities for visualizing GAM models and their components.

This module provides functions to create diagnostic and interpretive plots for
GAM models, including:
- 1D plots for univariate smooth terms with confidence intervals
- 2D plots for bivariate smooth terms as colored surfaces
- Partial residual plots for model diagnostics
- Integration with matplotlib for custom styling

The plotting functions are designed to work seamlessly with the term types
defined in pymgcv.terms and support various customization options for
publication-ready figures.

Currently supports plotting of continuous terms. Categorical and factor
smooth terms may be added in future versions.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.axes import Axes

from pymgcv import terms
from pymgcv.gam import FittedGAM


def plot_all(
    gam: FittedGAM,
):
    """Utility for plotting all terms in a model."""
    raise NotImplementedError()


def plot_continuous_1d(
    target: str,
    term: terms.Linear | terms.Smooth,
    gam: FittedGAM,
    data: pd.DataFrame,
    eval_density: int = 100,
    by_val: None | float | int | str = None,
    n_standard_errors: int | float = 2,
    plot_kwargs: dict[str, Any] | None = None,
    fill_between_kwargs: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot 1D smooth or linear terms with confidence intervals and partial residuals.

    Creates a plot showing:
    - The estimated smooth function or linear relationship
    - Confidence intervals around the estimate
    - Partial residuals as scatter points if available

    Args:
        target: Name of the response variable from the model specification.
        term: The model term to plot. Must be a univariate term (single variable).
        gam: FittedGAM model containing the term to plot.
        data: DataFrame used for plotting partial residuals and determining
            axis limits. Should typically be the training data.
        eval_density: Number of evaluation points along the variable range
            for plotting the smooth curve. Higher values give smoother curves
            but increase computation time. Default is 100.
        by_val: Value of the "by" variable to use when plotting the smooth. Must be
            provided if the term has a "by" variable. If the by variable is categorical,
            this should be a string representing the category to plot, and only partial
            residuals for that category will be plotted.
        n_standard_errors: Number of standard errors for confidence intervals.
            Common values: 1 (≈68% CI), 1.96 (95% CI), 2 (≈95% CI). Default is 2.
        plot_kwargs: Keyword arguments passed to ``matplotlib.pyplot.plot`` for
            the main curve.
        fill_between_kwargs: Keyword arguments passed to `matplotlib.pyplot.fill_between`
            for the confidence interval band.
        scatter_kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`
            for partial residuals.
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    Returns:
        The matplotlib Axes object with the plot.

    Raises:
        ValueError: If the term has more than one variable (not univariate).

    Note:
        For terms with 'by' variables, the by variable is set to 1 during
        evaluation, showing the term's effect before scaling by the by variable.
        This allows visualization of the underlying functional form.
    """
    data = data.copy()
    if term.by is not None and by_val is None:
        raise ValueError("by_val must be provided for terms with 'by' variables.")

    ax = plt.gca() if ax is None else ax
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    fill_between_kwargs = {} if fill_between_kwargs is None else fill_between_kwargs
    fill_between_kwargs.setdefault("alpha", 0.2)
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs.setdefault("s", 0.1 * rcParams["lines.markersize"] ** 2)

    # Matching color, particularly nice for plotting categorical by smooths on same ax
    current_color = ax._get_lines.get_next_color()
    for kwargs in (plot_kwargs, fill_between_kwargs, scatter_kwargs):
        if "c" not in kwargs and "color" not in kwargs:
            kwargs["color"] = current_color

    if len(term.varnames) != 1:
        raise ValueError(
            f"Expected term to be a function of a single variable, got {term.varnames}",
        )

    if term.by is not None and isinstance(by_val, str):
        data = data[data[term.by] == by_val]

    x = data[term.varnames[0]]
    x0_linspace = np.linspace(x.min(), x.max(), num=eval_density)
    spaced_data = pd.DataFrame({term.varnames[0]: x0_linspace})

    if term.by is not None:
        if isinstance(by_val, str):
            categories = data[
                term.by
            ].cat.categories  # TODO maintain ordered/unordered?
            spaced_data[term.by] = pd.Categorical(
                [by_val] * len(spaced_data),
                categories=categories,
            )
        else:
            spaced_data[term.by] = by_val

    pred = gam.partial_effect(target, term, spaced_data)

    # Add partial residuals
    partial_residuals = gam.partial_residuals(target, term, data)
    ax.scatter(x, partial_residuals, **scatter_kwargs)

    # Plot interval
    ax.fill_between(
        x0_linspace,
        pred["fit"] - n_standard_errors * pred["se"],
        pred["fit"] + n_standard_errors * pred["se"],
        **fill_between_kwargs,
    )

    ax.plot(x0_linspace, pred["fit"], **plot_kwargs)
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(f"partial effect: {term.simple_string()}")
    return ax


def plot_categorical(
    target: str,
    term: terms.Linear,
    gam: FittedGAM,
    data: pd.DataFrame,
    n_standard_errors: int | float = 2,
    errorbar_kwargs: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    ax: Axes | None = None,
):
    errorbar_kwargs = errorbar_kwargs or {}
    errorbar_kwargs.setdefault("capsize", 10)
    errorbar_kwargs.setdefault("fmt", ".")

    scatter_kwargs = scatter_kwargs or {}
    scatter_kwargs.setdefault("s", 0.1 * rcParams["lines.markersize"] ** 2)
    ax = plt.gca() if ax is None else ax

    # TODO: level ordered/order invariance
    levels = pd.Series(
        data[term.varnames[0]].cat.categories,
        dtype="category",
        name=term.varnames[0],
    )
    partial_residuals = gam.partial_residuals(target, term, data)

    jitter = np.random.uniform(-0.25, 0.25, size=len(data))
    scatter_kwargs.setdefault("alpha", 0.2)

    ax.scatter(
        data[term.varnames[0]].cat.codes + jitter,
        partial_residuals,
        **scatter_kwargs,
    )
    ax.set_xticks(ticks=levels.cat.codes, labels=levels)

    pred = gam.partial_effect(
        target=target,
        term=term,
        data=pd.DataFrame(levels),
    )
    ax.errorbar(
        x=levels.cat.codes,
        y=pred["fit"],
        yerr=n_standard_errors * pred["se"],
        **errorbar_kwargs,
    )
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(f"partial effect: {term.simple_string()}")
    return ax


def plot_continuous_2d(
    target: str,
    term: terms.Smooth | terms.TensorSmooth,
    gam: FittedGAM,
    data: pd.DataFrame,
    eval_density: int = 50,
    by_val: None | float | int | str = None,
    colormesh_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot 2D smooth surfaces as colored mesh plots with data overlay.

    Creates a comprehensive 2D visualization showing:
    - The estimated smooth surface as a colored mesh
    - Original data points overlaid as scatter plot
    - Automatic colorbar for interpreting surface values
    - Proper axis labels from variable names

    This function is essential for understanding bivariate relationships
    and interactions between two continuous variables.

    Args:
        response: Name of the response variable from the model specification.
        term: The bivariate term to plot. Must have exactly two variables.
            Can be Smooth('x1', 'x2') or TensorSmooth('x1', 'x2').
        gam: Fitted GAM model containing the term to plot.
        data: DataFrame containing the variables for determining plot range
            and showing data points. Should typically be the training data.
        eval_density: Number of evaluation points along each axis, creating
            an eval_density × eval_density grid. Higher values give smoother
            surfaces but increase computation time. Default is 50.
        by_val: Value of the "by" variable to use when plotting the smooth. Must be
            provided if the term has a "by" variable. If the by variable is categorical,
            this should be a string representing the category to plot, and only partial
            residuals for that category will be plotted.
        colormesh_kwargs: Keyword arguments passed to matplotlib.pyplot.pcolormesh()
            for the surface plot. Useful for controlling colormap, shading, etc.
        scatter_kwargs: Keyword arguments passed to matplotlib.pyplot.scatter()
            for the data points overlay. Default color is black with small points.
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    Returns:
        The matplotlib Axes object with the plot, allowing further customization.

    Raises:
        ValueError: If the term doesn't have exactly two variables.

    Examples:
        ```python
        import matplotlib.pyplot as plt
        from pymgcv.plot import plot_2d

        # Basic 2D surface plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_2d('y', TensorSmooth('x1', 'x2'), model, data, ax=ax)
        plt.show()

        # Customized surface plot
        plot_2d('y', Smooth('longitude', 'latitude'), model, data,
                eval_density=100,  # High resolution
                colormesh_kwargs={'cmap': 'RdBu', 'shading': 'gouraud'},
                scatter_kwargs={'alpha': 0.6, 's': 20, 'c': 'white'})

        # Multiple subplots for comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        plot_2d('y1', TensorSmooth('x1', 'x2'), model1, data, ax=axes[0])
        plot_2d('y2', TensorSmooth('x1', 'x2'), model2, data, ax=axes[1])
        ```

    Note:
        The function automatically adds a colorbar to interpret the surface values.
        The colorbar represents the partial effect of the term (contribution to
        the linear predictor on the link scale).
    """
    data = data.copy()

    if term.by is not None and by_val is None:
        raise ValueError("by_val must be provided for terms with 'by' variables.")

    ax = plt.gca() if ax is None else ax
    colormesh_kwargs = {} if colormesh_kwargs is None else colormesh_kwargs
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs

    scatter_kwargs.setdefault("color", "black")
    scatter_kwargs.setdefault("s", 0.1 * rcParams["lines.markersize"] ** 2)

    if len(term.varnames) != 2:
        raise ValueError(
            f"Expected term to be a function of a two variables, got {term.varnames}",
        )
    x0_lims = (data[term.varnames[0]].min(), data[term.varnames[0]].max())
    x1_lims = (data[term.varnames[1]].min(), data[term.varnames[1]].max())
    x0_mesh, x1_mesh = np.meshgrid(
        np.linspace(*x0_lims, eval_density),
        np.linspace(*x1_lims, eval_density),
    )

    # TODO by factor

    if term.by is not None:
        if isinstance(by_val, str):
            categories = data[
                term.by
            ].cat.categories  # TODO maintain ordered/unordered?
            spaced_data[term.by] = pd.Categorical(
                [by_val] * len(spaced_data),
                categories=categories,
            )
        else:
            spaced_data[term.by] = by_val

    pred = gam.partial_effect(
        target,
        term,
        data=pd.DataFrame(
            {term.varnames[0]: x0_mesh.ravel(), term.varnames[1]: x1_mesh.ravel()},
            index=x0_mesh.ravel(),
        ),
    )["fit"]
    mesh = ax.pcolormesh(
        x0_mesh,
        x1_mesh,
        pred.to_numpy().reshape(x0_mesh.shape),
        **colormesh_kwargs,
    )
    ax.figure.colorbar(mesh, ax=ax)

    ax.scatter(
        data[term.varnames[0]],
        data[term.varnames[1]],
        **scatter_kwargs,
    )

    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(term.varnames[1])
    return ax
