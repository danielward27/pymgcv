"""Plotting utilities for visualizing GAM models."""

import types
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from typing import Any, Literal, TypeGuard

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import CategoricalDtype
from pandas.api.types import is_numeric_dtype
from rpy2.robjects.packages import importr

from pymgcv.basis_functions import RandomWigglyCurve
from pymgcv.gam import AbstractGAM
from pymgcv.qq import qq_simulate, qq_uniform
from pymgcv.terms import (
    L,
    S,
    T,
    TermLike,
    _RandomWigglyToByInterface,
)

rbase = importr("base")
rstats = importr("stats")


def plot_gam(
    gam: AbstractGAM,
    *,
    ncols: int = 2,
    plot_scatter: bool = False,
    to_plot: type | types.UnionType | dict[str, list[TermLike]] = TermLike,
    kwargs_mapper: dict[Callable, dict[str, Any]] | None = None,
) -> tuple[Figure, plt.Axes | np.ndarray]:
    """Plot a gam model.

    Args:
        gam: The fitted gam object to plot.
        ncols: The number of columns before wrapping axes.
        plot_scatter: Whether to plot the residuals (where possible), and the overlayed
            datapoints on 2D plots. Defaults to False.
        to_plot: Which terms to plot. If a type, only plots terms
            of that type (e.g. ``to_plot = S | T`` to plot smooths).
            If a dictionary, it should map the target names to
            an iterable of terms to plot (similar to how models are specified).
        kwargs_mapper: Used to pass keyword arguments to the underlying `pymgcv.plot`
            functions. A dictionary mapping the plotting function to kwargs. For
            example, to disable the confidence intervals on the 1d plots, set
            ``kwargs_mapper`` to
            ```python
            from pymgcv.plot import plot_continuous_1d
            {plot_continuous_1d: {"fill_between_kwargs": {"disable": True}}}
            ```
    """
    if gam.fit_state is None:
        raise ValueError("Cannot plot before fitting the model.")

    kwargs_mapper = {} if kwargs_mapper is None else kwargs_mapper

    if plot_scatter:
        # TODO manually check this. Does providing a kwargs mapper interfer with this?
        kwargs_mapper.setdefault(plot_categorical, {}).setdefault("residuals", True)
        kwargs_mapper.setdefault(plot_continuous_1d, {}).setdefault("residuals", True)
        kwargs_mapper.setdefault(plot_continuous_2d, {}).setdefault(
            "scatter_kwargs",
            {"disable": False},
        )

    if isinstance(to_plot, type | types.UnionType):
        to_plot = {
            k: [v for v in terms if isinstance(v, to_plot)]
            for k, terms in gam.all_predictors.items()
        }

    data = gam.fit_state.data
    plotters = []
    for target, terms in to_plot.items():
        for term in terms:
            try:
                plotter = get_term_plotter(
                    target,
                    term=term,
                    gam=gam,
                    data=data,
                )
            except NotImplementedError:
                continue
            plotters.append(plotter)

    n_axs = sum(p.required_axes for p in plotters)
    if n_axs == 0:
        raise ValueError("Do not know how to plot any terms in the model.")

    ncols = min(n_axs, ncols)
    fig, axes = plt.subplots(
        nrows=ceil(n_axs / ncols),
        ncols=ncols,
        layout="constrained",
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    idx = 0
    for plotter in plotters:
        kwargs = kwargs_mapper.get(plotter.underlying_function, {})
        plotter.make_plot(axes[idx : (idx + plotter.required_axes)], **kwargs)
        idx += plotter.required_axes

    return fig, axes


@dataclass
class _TermPlotter:
    make_plot: Callable
    underlying_function: Callable
    required_axes: int = 1


def get_term_plotter(
    target: str,
    term: TermLike,
    gam: AbstractGAM,
    data: pd.DataFrame | None = None,
) -> _TermPlotter:
    """Utility for plotting a term in a model.

    Because some terms need multiple axes for plotting, this returns the number of axes
    required, and a function that applies the plotting to an iterable of axes, taking
    only the axes as an argument. This allows us to setup the axes before plotting
    when plotting multiple terms.
    """
    if gam.fit_state is None:
        raise ValueError("Cannot plot before fitting the model.")
    data = data if data is not None else gam.fit_state.data
    data = data.copy()

    if _is_random_wiggly(term):
        term = _RandomWigglyToByInterface(term)

    dtypes = data[list(term.varnames)].dtypes
    by_dtype = data[term.by].dtype if term.by is not None else None
    dim = len(term.varnames)
    is_categorical_by = term.by is not None and isinstance(by_dtype, CategoricalDtype)
    levels = by_dtype.categories if is_categorical_by else [None]

    def _all_numeric(dtypes):
        return all(is_numeric_dtype(dtype) for dtype in dtypes)

    # TODO improve passing of kwargs, or output the underlying method name
    # such that we can select an appropriate set of kwargs in plot_gam.
    match (dim, term):
        case (1, L()) if isinstance(dtypes[term.varnames[0]], CategoricalDtype):

            def _plot_wrapper(axes, **kwargs):
                axes[0] = plot_categorical(
                    target=target,
                    term=term,
                    gam=gam,
                    data=data,
                    ax=axes[0],
                    **kwargs,
                )
                return axes

            return _TermPlotter(_plot_wrapper, plot_categorical)

        # TODO "re" basis?

        case (1, TermLike()) if _all_numeric(dtypes):

            def _plot_wrapper(axes, **kwargs):
                for level in levels:
                    axes[0] = plot_continuous_1d(
                        target=target,
                        term=term,
                        gam=gam,
                        data=data,
                        level=level,
                        ax=axes[0],
                        plot_kwargs={"label": level},
                        **kwargs,
                    )
                if is_categorical_by:
                    axes[0].legend()
                return axes

            return _TermPlotter(_plot_wrapper, plot_continuous_1d)

        case (2, TermLike()) if _all_numeric(dtypes):

            def _plot_wrapper(axes, **kwargs):
                for i, level in enumerate(levels):
                    axes[i] = plot_continuous_2d(
                        target=target,
                        term=term,
                        gam=gam,
                        data=data,
                        level=level,
                        ax=axes[i],
                        **kwargs,
                    )
                    if is_categorical_by:
                        axes[i].set_title(f"Level={level}")
                return axes

            return _TermPlotter(
                _plot_wrapper,
                plot_continuous_2d,
                required_axes=len(levels),
            )

        case _:
            raise NotImplementedError(f"Did not know how to plot term {term}.")


def plot_continuous_1d(
    *,
    target: str,
    term: TermLike,
    gam: AbstractGAM,
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
    """Plot 1D smooth or linear terms with confidence intervals.

    !!! note

        - For terms with numeric "by" variables, the "by" variable is set to 1,
        showing the unscaled effect of the smooth.

    Args:
        target: Name of the response variable from the model specification.
        term: The model term to plot. Must be a univariate term (single variable).
        gam: GAM model containing the term to plot.
        data: DataFrame used for plotting partial residuals and determining
            axis limits. Defaults to the data used for training.
        eval_density: Number of evaluation points along the variable range
            for plotting the smooth curve. Higher values give smoother curves
            but increase computation time. Default is 100.
        level: Must be provided for smooths with a categorical "by" variable or a
            [`RandomWigglyCurve`][pymgcv.basis_functions.RandomWigglyCurve] basis.
            Specifies the level to plot.
        n_standard_errors: Number of standard errors for confidence intervals.
        residuals: Whether to plot partial residuals.
        plot_kwargs: Keyword arguments passed to ``matplotlib.pyplot.plot`` for
            the main curve.
        fill_between_kwargs: Keyword arguments passed to
            `matplotlib.pyplot.fill_between` for the confidence interval band.
            Pass `{"disable": True}` to disable the confidence interval band.
        scatter_kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`
            for partial residuals (ignored if `residuals=False`).
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    Returns:
        The matplotlib Axes object with the plot.
    """
    if gam.fit_state is None:
        raise ValueError("Cannot plot before fitting the model.")
    data = data if data is not None else gam.fit_state.data
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
        assert isinstance(data, pd.DataFrame)

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

    pred = gam.partial_effect(target, term, spaced_data, compute_se=True)

    # Add partial residuals
    if residuals and target in data.columns:
        partial_residuals = gam.partial_residuals(target, term, data)
        ax.scatter(data[term.varnames[0]], partial_residuals, **scatter_kwargs)

    # Plot interval
    assert pred.se is not None
    _with_disable(ax.fill_between)(
        x0_linspace,
        pred.fit - n_standard_errors * pred.se,
        pred.fit + n_standard_errors * pred.se,
        **fill_between_kwargs,
    )

    ax.plot(x0_linspace, pred.fit, **plot_kwargs)
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(f"link({target})~{term.label()}")
    return ax


def plot_continuous_2d(
    *,
    target: str,
    term: TermLike,
    gam: AbstractGAM,
    data: pd.DataFrame | None = None,
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
        gam: GAM model containing the term to plot.
        data: DataFrame containing the variables for determining plot range
            and showing data points. Should typically be the training data.
        eval_density: Number of evaluation points along each axis, creating
            an eval_density × eval_density grid. Higher values give smoother
            surfaces but increase computation time. Default is 50.
        level: Must be provided for smooths with a categorical "by" variable or a
            [`RandomWigglyCurve`][pymgcv.basis_functions.RandomWigglyCurve] basis.
            Specifies the level to plot.
        contour_kwargs: Keyword arguments passed to `matplotlib.pyplot.contour`
            for the contour lines.
        contourf_kwargs: Keyword arguments passed to `matplotlib.pyplot.contourf`
            for the filled contours.
        scatter_kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`
            for the data points overlay. Pass `{"disable": True}` to avoid plotting.
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    Returns:
        The matplotlib Axes object with the plot, allowing further customization.

    Raises:
        ValueError: If the term doesn't have exactly two variables.
    """
    if gam.fit_state is None:
        raise ValueError("Cannot plot before fitting the model.")
    data = data if data is not None else gam.fit_state.data
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
        assert isinstance(data, pd.DataFrame)

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

    pred = gam.partial_effect(
        target,
        term,
        data=spaced_data,
    ).fit

    mesh = ax.contourf(
        x0_mesh,
        x1_mesh,
        pred.reshape(x0_mesh.shape),
        **contourf_kwargs,
    )
    _with_disable(ax.contour)(
        x0_mesh,
        x1_mesh,
        pred.reshape(x0_mesh.shape),
        **contour_kwargs,
    )
    color_bar = ax.figure.colorbar(mesh, ax=ax, pad=0)
    color_bar.set_label(f"link({target})~{term.label()}")
    _with_disable(ax.scatter)(
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
    gam: AbstractGAM,
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
        gam: GAM model containing the term to plot.
        data: DataFrame containing the categorical variable and response.
        residuals: Whether to plot partial residuals (jittered on x-axis).
        n_standard_errors: Number of standard errors for confidence intervals.
        errorbar_kwargs: Keyword arguments passed to `matplotlib.pyplot.errorbar`.
        scatter_kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`.
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    """
    if gam.fit_state is None:
        raise RuntimeError("The model must be fitted before plotting.")
    data = gam.fit_state.data if data is None else data
    errorbar_kwargs = {} if errorbar_kwargs is None else errorbar_kwargs
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs.setdefault("s", 0.1 * rcParams["lines.markersize"] ** 2)
    errorbar_kwargs.setdefault("capsize", 10)
    errorbar_kwargs.setdefault("fmt", ".")

    ax = plt.gca() if ax is None else ax

    levels = pd.Series(
        data[term.varnames[0]].cat.categories,
        dtype=data[term.varnames[0]].dtype,
        name=term.varnames[0],
    )

    if residuals and target in data.columns:
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
        compute_se=True,
    )

    assert pred.se is not None

    ax.errorbar(
        x=levels.cat.codes,
        y=pred.fit,
        yerr=n_standard_errors * pred.se,
        **errorbar_kwargs,
    )
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(f"partial effect: {term.label()}")
    return ax


def _is_random_wiggly(term: TermLike) -> TypeGuard[T | S]:
    if isinstance(term, S | T):
        return isinstance(term.bs, RandomWigglyCurve)
    return False


def plot_qq(
    gam: AbstractGAM,
    *,
    n: int = 10,
    scatter_kwargs: dict | None = None,
    plot_kwargs: dict | None = None,
    ax: Axes | None = None,
    method: Literal["uniform", "simulate"] = "uniform",
) -> Axes:
    """A Q-Q plot of deviance residuals.

    Args:
        gam: The fitted GAM model.
        n: The number of simulated sets to use for generating the theoretical
            quantiles.
        scatter_kwargs: Key word arguments passed to `matplotlib.pyplot.scatter`.
        plot_kwargs: Key word arguments passed to `matplotlib.pyplot.plot` for
            plotting the reference line. Pass {"disable": True} to avoid plotting.
        ax: Matplotlib axes to use for the plot.

    Returns:
        The matplotlib axes object.

    !!! example

        As an example, we will create a heavy tailed response variable,
        and fit a [`Gaussian`][pymgcv.families.Gaussian] model, and a
        [`Scat`][pymgcv.families.Scat] model.

        ```python
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        from pymgcv.families import Gaussian, Scat
        from pymgcv.gam import GAM
        from pymgcv.plot import plot_qq
        from pymgcv.terms import S

        rng = np.random.default_rng(1)
        n = 1000
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.standard_t(df=3, size=n)  # Heavy-tailed
        data = pd.DataFrame({"x": x, "y": y})

        models = [
            GAM({"y": S("x")}, family=Gaussian()),
            GAM({"y": S("x")}, family=Scat()),  # Better for heavy-tailed data
        ]

        fig, axes = plt.subplots(ncols=2)

        for model, ax in zip(models, axes, strict=False):
            model.fit(data)
            plot_qq(model, ax=ax, method="simulate")
            ax.set_title(model.family.__class__.__name__)
            ax.set_box_aspect(1)

        fig.show()
        ```
    """
    method_map = {
        "uniform": qq_uniform,
        "simulate": qq_simulate,
    }
    qq_fun = method_map[method]
    if gam.fit_state is None:
        raise RuntimeError("The model must be fitted before plotting.")

    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs.setdefault("s", 0.5 * rcParams["lines.markersize"] ** 2)

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    if "c" not in plot_kwargs and "color" not in plot_kwargs:
        plot_kwargs["color"] = "gray"
    plot_kwargs.setdefault("linestyle", "--")

    ax = plt.gca() if ax is None else ax
    qq_data = qq_fun(gam, n=n)
    ax.scatter(qq_data.theoretical, qq_data.residuals, **scatter_kwargs)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Residuals")

    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    _with_disable(ax.plot)([min_val, max_val], [min_val, max_val], **plot_kwargs)
    return ax


def _with_disable(plot_func):
    """Wraps a plot function to easily disable with disable=True."""

    def wrapper(*args, disable=False, **kwargs):
        if disable:
            return None
        return plot_func(*args, **kwargs)

    return wrapper
