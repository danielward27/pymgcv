"""Plotting utilities for visualizing GAM models and their components."""

import types
from math import ceil
from typing import Any, TypeGuard

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.axes import Axes
from pandas import CategoricalDtype
from pandas.api.types import is_numeric_dtype

from pymgcv.basis_functions import RandomWigglyCurve
from pymgcv.gam import FittedGAM
from pymgcv.terms import (
    L,
    S,
    T,
    TermLike,
    _RandomWigglyToByInterface,
)


def plot_gam(
    fit: FittedGAM,
    *,
    ncols=2,
    residuals: bool = False,
    to_plot: type | types.UnionType | dict[str, list[TermLike]] = TermLike,
):
    """Plot a gam model.

    Args:
        fit: The fitted gam object to plot.
        ncols: The number of columns before wrapping axes.
        residuals: Whether to plot the residuals (where possible). Defaults to False.
        to_plot: Which terms to plot. If a type, only plots terms
            of that type (e.g. ``to_plot = S | T`` to plot smooths).
            If a dictionary, it should map the target names to
            an iterable of terms to plot (similar to how models are specified).
    """
    if isinstance(to_plot, type | types.UnionType):
        to_plot = {
            k: [v for v in terms if isinstance(v, to_plot)]
            for k, terms in fit.gam.all_predictors.items()
        }

    data = fit.data
    n_axs = []
    plotters = []
    for target, terms in to_plot.items():
        for term in terms:
            try:
                n_ax, plotter = get_term_plotter(
                    target,
                    term=term,
                    fit=fit,
                    data=data,
                    residuals=residuals,
                )
            except NotImplementedError:
                continue
            n_axs.append(n_ax)
            plotters.append(plotter)

    if sum(n_axs) == 0:
        raise ValueError("Do not know how to plot any terms in the model.")

    ncols = min(sum(n_axs), ncols)
    fig, axes = plt.subplots(
        nrows=ceil(sum(n_axs) / ncols),
        ncols=ncols,
        layout="constrained",
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    idx = 0
    for n_ax, plotter in zip(n_axs, plotters, strict=True):
        plotter(axes[idx : (idx + n_ax)])
        idx += n_ax

    return fig, axes


def get_term_plotter(
    target: str,
    term: TermLike,
    fit: FittedGAM,
    data: pd.DataFrame | None = None,
    *,
    residuals: bool = False,
    **kwargs,
):
    """Utility for plotting a term in a model.

    Because some terms need multiple axes for plotting, this returns the number of axes
    required, and a function that applies the plotting to an iterable of axes, taking
    only the axes as an argument. This allows us to setup the axes before plotting,
    when plotting multiple terms.
    """
    data = data if data is not None else fit.data

    if _is_random_wiggly(term):
        term = _RandomWigglyToByInterface(term)

    dtypes = data[list(term.varnames)].dtypes
    by_dtype = data[term.by].dtype if term.by is not None else None
    dim = len(term.varnames)
    is_categorical_by = term.by is not None and isinstance(by_dtype, CategoricalDtype)
    levels = by_dtype.categories if is_categorical_by else [None]

    def _all_numeric(dtypes):
        return all(is_numeric_dtype(dtype) for dtype in dtypes)

    match (dim, term):
        case (1, L()) if isinstance(dtypes[term.varnames[0]], CategoricalDtype):

            def _plot_wrapper(axes):
                axes[0] = plot_categorical(
                    target=target,
                    term=term,
                    fit=fit,
                    data=data,
                    ax=axes[0],
                    residuals=residuals,
                    **kwargs,
                )
                return axes

            return (1, _plot_wrapper)

        # TODO "re" basis?

        case (1, TermLike()) if _all_numeric(dtypes):

            def _plot_wrapper(axes):
                for level in levels:
                    axes[0] = plot_continuous_1d(
                        target=target,
                        term=term,
                        fit=fit,
                        data=data,
                        level=level,
                        ax=axes[0],
                        residuals=residuals,
                        plot_kwargs={"label": level},
                        **kwargs,
                    )
                if is_categorical_by:
                    axes[0].legend()
                return axes

            return (1, _plot_wrapper)

        case (2, TermLike()) if _all_numeric(dtypes):

            def _plot_wrapper(axes):
                for i, level in enumerate(levels):
                    axes[i] = plot_continuous_2d(
                        target=target,
                        term=term,
                        fit=fit,
                        data=data,
                        level=level,
                        ax=axes[i],
                        **kwargs,
                    )
                    if is_categorical_by:
                        axes[i].set_title(f"Level={level}")
                return axes

            return (len(levels), _plot_wrapper)

        case _:
            raise NotImplementedError(f"Did not know how to plot term {term}.")


def plot_continuous_1d(
    *,
    target: str,
    term: TermLike,
    fit: FittedGAM,
    data: pd.DataFrame | None = None,
    eval_density: int = 100,
    level: str | None = None,
    n_standard_errors: int | float = 2,
    residuals: bool = False,
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

    .. Note::
        For terms with numeric "by" variables, the "by" variable is set to 1,
        showing the unscaled effect of the smooth.

    Args:
        target: Name of the response variable from the model specification.
        term: The model term to plot. Must be a univariate term (single variable).
        fit: FittedGAM model containing the term to plot.
        data: DataFrame used for plotting partial residuals and determining
            axis limits. Defaults to the data used for training.
        eval_density: Number of evaluation points along the variable range
            for plotting the smooth curve. Higher values give smoother curves
            but increase computation time. Default is 100.
        level: Must be provided for smooths with a categorical "by" variable or a
            [`RandomWigglyCurve`][pymgcv.terms.RandomWigglyCurve] basis.
            Specifies the level to plot.
        n_standard_errors: Number of standard errors for confidence intervals.
        residuals: Whether to plot partial residuals.
        plot_kwargs: Keyword arguments passed to ``matplotlib.pyplot.plot`` for
            the main curve.
        fill_between_kwargs: Keyword arguments passed to
            `matplotlib.pyplot.fill_between` for the confidence interval band.
        scatter_kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`
            for partial residuals (ignored if `residuals=False`).
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    Returns:
        The matplotlib Axes object with the plot.
    """
    data = data if data is not None else fit.data
    data = data.copy()
    term = _RandomWigglyToByInterface(term) if _is_random_wiggly(term) else term
    is_categorical_by = term.by and isinstance(data[term.by].dtype, CategoricalDtype)

    if len(term.varnames) != 1:
        raise ValueError(
            f"Expected varnames to be one continuous variable, got {term.varnames}",
        )
    if is_categorical_by and level is None:
        raise ValueError(
            "level must be provided for terms with 'by' variables, or "
            "RandomWigglyCurves.",
        )

    if level is not None:
        data = data.loc[data[term.by] == level]

    # TODO handling of partial residuals with numeric by?
    x0_linspace = np.linspace(
        data[term.varnames[0]].min(),
        data[term.varnames[0]].max(),
        num=eval_density,
    )
    spaced_data = pd.DataFrame({term.varnames[0]: x0_linspace})

    if term.by is not None:
        if is_numeric_dtype(data[term.by].dtype):
            spaced_data[term.by] = 1
        else:
            spaced_data[term.by] = pd.Series(
                [level] * eval_density,
                dtype=data[term.by].dtype,
            )

    ax = plt.gca() if ax is None else ax
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    fill_between_kwargs = {} if fill_between_kwargs is None else fill_between_kwargs
    fill_between_kwargs.setdefault("alpha", 0.2)
    scatter_kwargs.setdefault("s", 0.1 * rcParams["lines.markersize"] ** 2)

    # Matching color, particularly nice for plotting categorical by smooths on same ax
    current_color = ax._get_lines.get_next_color()
    for kwargs in (plot_kwargs, fill_between_kwargs, scatter_kwargs):
        if "c" not in kwargs and "color" not in kwargs:
            kwargs["color"] = current_color

    pred = fit.partial_effect(target, term, spaced_data)

    # Add partial residuals
    if residuals and target in data.columns:
        partial_residuals = fit.partial_residuals(target, term, data)
        ax.scatter(data[term.varnames[0]], partial_residuals, **scatter_kwargs)

    # Plot interval
    ax.fill_between(
        x0_linspace,
        pred["fit"] - n_standard_errors * pred["se"],
        pred["fit"] + n_standard_errors * pred["se"],
        **fill_between_kwargs,
    )

    ax.plot(x0_linspace, pred["fit"], **plot_kwargs)
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(f"link({target})~{term.label()}")
    return ax


def plot_continuous_2d(
    *,
    target: str,
    term: TermLike,
    fit: FittedGAM,
    data: pd.DataFrame,
    eval_density: int = 50,
    level: str | None = None,
    contour_kwargs: dict | None = None,
    contourf_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot 2D smooth surfaces as contour plots with data overlay.

    This function is essential for understanding bivariate relationships
    and interactions between two continuous variables.

    Args:
        target: Name of the response variable from the model specification.
        term: The bivariate term to plot. Must have exactly two variables.
            Can be S('x1', 'x2') or T('x1', 'x2').
        fit: Fitted GAM model containing the term to plot.
        data: DataFrame containing the variables for determining plot range
            and showing data points. Should typically be the training data.
        eval_density: Number of evaluation points along each axis, creating
            an eval_density × eval_density grid. Higher values give smoother
            surfaces but increase computation time. Default is 50.
        level: Must be provided for smooths with a categorical "by" variable or a
            [`RandomWigglyCurve`][pymgcv.terms.RandomWigglyCurve] basis.
            Specifies the level to plot.
        contour_kwargs: Keyword arguments passed to `matplotlib.pyplot.contour`
            for the contour lines.
        contourf_kwargs: Keyword arguments passed to `matplotlib.pyplot.contourf`
            for the filled contours.
        scatter_kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`
            for the data points overlay.
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    Returns:
        The matplotlib Axes object with the plot, allowing further customization.

    Raises:
        ValueError: If the term doesn't have exactly two variables.
    """
    data = data.copy()
    term = _RandomWigglyToByInterface(term) if _is_random_wiggly(term) else term
    is_categorical_by = term.by and isinstance(data[term.by].dtype, CategoricalDtype)

    if len(term.varnames) != 2:
        raise ValueError(
            f"Expected varnames to be one continuous variable, got {term.varnames}",
        )

    if is_categorical_by and level is None:
        raise ValueError(
            "level must be provided for terms with 'by' variables, or "
            "RandomWigglyCurves.",
        )

    if level is not None:
        data = data.loc[data[term.by] == level]

    x0_lims = (data[term.varnames[0]].min(), data[term.varnames[0]].max())
    x1_lims = (data[term.varnames[1]].min(), data[term.varnames[1]].max())
    x0_mesh, x1_mesh = np.meshgrid(
        np.linspace(*x0_lims, eval_density),
        np.linspace(*x1_lims, eval_density),
    )
    spaced_data = pd.DataFrame(
        {term.varnames[0]: x0_mesh.ravel(), term.varnames[1]: x1_mesh.ravel()},
    )
    if term.by is not None:
        if is_numeric_dtype(data[term.by].dtype):
            spaced_data[term.by] = 1
        else:
            spaced_data[term.by] = pd.Series(
                [level] * eval_density**2,
                dtype=data[term.by].dtype,
            )

    ax = plt.gca() if ax is None else ax
    contour_kwargs = {} if contour_kwargs is None else contour_kwargs
    contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs

    contour_kwargs.setdefault("levels", 14)
    contourf_kwargs.setdefault("levels", 14)
    contourf_kwargs.setdefault("alpha", 0.8)
    scatter_kwargs.setdefault("color", "black")
    scatter_kwargs.setdefault("s", 0.1 * rcParams["lines.markersize"] ** 2)

    pred = fit.partial_effect(
        target,
        term,
        data=spaced_data,
    )["fit"]

    mesh = ax.contourf(
        x0_mesh,
        x1_mesh,
        pred.to_numpy().reshape(x0_mesh.shape),
        **contourf_kwargs,
    )
    ax.contour(
        x0_mesh,
        x1_mesh,
        pred.to_numpy().reshape(x0_mesh.shape),
        **contour_kwargs,
    )
    color_bar = ax.figure.colorbar(mesh, ax=ax, pad=0)
    color_bar.set_label(f"link({target})~{term.label()}")
    ax.scatter(
        data[term.varnames[0]],
        data[term.varnames[1]],
        **scatter_kwargs,
    )
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(term.varnames[1])
    return ax


def plot_categorical(
    *,
    target: str,
    term: L,
    fit: FittedGAM,
    data: pd.DataFrame | None = None,
    residuals: bool = False,
    n_standard_errors: int | float = 2,
    errorbar_kwargs: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot categorical terms with error bars and partial residuals.

    Creates a plot showing:

    - The estimated effect of each category level as points.
    - Error bars representing confidence intervals.
    - Partial residuals as jittered scatter points.

    Args:
        target: Name of the response variable from the model specification.
        term: The categorical term to plot. Must be a L term with a single
            categorical variable.
        fit: Fitted GAM model containing the term to plot.
        data: DataFrame containing the categorical variable and response.
        n_standard_errors: Number of standard errors for confidence intervals.
        errorbar_kwargs: Keyword arguments passed to `matplotlib.pyplot.errorbar`.
        scatter_kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`.
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    """
    data = fit.data if data is None else data
    errorbar_kwargs = {} if errorbar_kwargs is None else errorbar_kwargs
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs.setdefault("s", 0.1 * rcParams["lines.markersize"] ** 2)
    errorbar_kwargs.setdefault("capsize", 10)
    errorbar_kwargs.setdefault("fmt", ".")

    ax = plt.gca() if ax is None else ax

    # TODO: level ordered/order invariance
    levels = pd.Series(
        data[term.varnames[0]].cat.categories,
        dtype="category",
        name=term.varnames[0],
    )

    if residuals and target in data.columns:
        partial_residuals = fit.partial_residuals(target, term, data)

        jitter = np.random.uniform(-0.25, 0.25, size=len(data))
        scatter_kwargs.setdefault("alpha", 0.2)

        ax.scatter(
            data[term.varnames[0]].cat.codes + jitter,
            partial_residuals,
            **scatter_kwargs,
        )

    ax.set_xticks(ticks=levels.cat.codes, labels=levels)

    pred = fit.partial_effect(
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
    ax.set_ylabel(f"partial effect: {term.label()}")
    return ax


def _is_random_wiggly(term: TermLike) -> TypeGuard[T | S]:
    if isinstance(term, S | T):
        return isinstance(term.bs, RandomWigglyCurve)
    return False
