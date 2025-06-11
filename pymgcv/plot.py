"""Plotting utility functions for gam models.

We will likely refactor this, e.g. to dispatch using structural subtyping of terms,
or be a method of the terms, but for now this gets us started. Only plotting of
coninuous terms is currently supported.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.axes import Axes
from rpy2.robjects.packages import importr

from pymgcv import terms
from pymgcv.gam import FittedGAM

mgcViz = importr("mgcViz")


def plot_1d(
    response: str,
    term: terms.Linear | terms.Smooth,
    gam: FittedGAM,
    data: pd.DataFrame,
    eval_density: int = 100,
    n_standard_errors: int | float = 2,
    plot_kwargs: dict[str, Any] | None = None,
    fill_between_kwargs: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    ax: Axes | None = None,
) -> Axes:  # TODO data default to None
    """Plot 1D numeric terms (smooths or linear terms).

    Note, "by" variables are imputed as 1, i.e. showing the term prior to scaling
    by the "by" variable.

    Args:
        response: The target variable.
        term: The term to plot.
        gam: The fitted gam model.
        data: Data used for adding partial residuals and inferring the plot limits.
        plot_kwargs: Argumnets passed to plotting of the line. Defaults to None.
        fill_between_kwargs: Arguments passed to plotting of the confidence interval
            lines. Defaults to None.
        scatter_kwargs: Arguments passed to plotting of the partial
            residuals. Defaults to None.
        ax: Axes on to which the plot is applied. Defaults to None.
        eval_density: Number of points to use for plotting the term. Defaults to 100.
    """
    ax = plt.gca() if ax is None else ax
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    fill_between_kwargs = {} if fill_between_kwargs is None else fill_between_kwargs
    fill_between_kwargs.setdefault("alpha", 0.2)
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs.setdefault("s", 0.1 * rcParams["lines.markersize"] ** 2)

    if "c" not in plot_kwargs and "color" not in plot_kwargs:
        plot_kwargs["color"] = "black"

    if "c" not in fill_between_kwargs and "color" not in fill_between_kwargs:
        fill_between_kwargs["color"] = "black"

    if len(term.varnames) != 1:
        raise ValueError(
            f"Expected term to be a function of a single variable, got {term.varnames}",
        )

    x = data[term.varnames[0]]
    x0_linspace = np.linspace(x.min(), x.max(), num=eval_density)
    spaced_data = pd.DataFrame({term.varnames[0]: x0_linspace})

    if term.by is not None:
        spaced_data[term.by] = 1  # TODO could interpret by numeric by as 2 dimensional?

    pred = gam.partial_effect(response, term, spaced_data)

    # Add partial residuals
    partial_residuals = gam.partial_residuals(response, term, data)
    ax.scatter(x, partial_residuals, **scatter_kwargs)

    # Plot interval
    ax.fill_between(
        x0_linspace,
        pred["fit"] - n_standard_errors * pred["se"],
        pred["fit"] + n_standard_errors * pred["se"],
        **fill_between_kwargs,
    )

    ax.plot(x0_linspace, pred["fit"], **plot_kwargs)
    ax.set_xlabel(term.simple_string())
    ax.set_ylabel(f"{response} (link scale)")
    return ax


def plot_2d(
    response: str,
    term: terms.Smooth | terms.TensorSmooth,
    gam: FittedGAM,
    data: pd.DataFrame,
    eval_density: int = 50,
    colormesh_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot smooth or tensor smooth terms of two variables as a colormesh.

    Args:
        response: The name of the response variable.
        term: The term to plot.
        gam: The fitted gam model
        density: The density of the points (the square root of the evaluations).
            Defaults to 50.
        ax: Matplotlib axes to use. Defaults to None.
        colormesh_kwargs (dict | None, optional): Any keyword arguments passed to
            ``pcolormesh``. Defaults to None.
        scatter_kwargs (dict | None, optional): Any keyword arguments passed to
            ``matplotlib.pyplot.scatter`` for plotting the data points.
    """
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

    pred = gam.partial_effect(
        response,
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


# def plot_1d(
#     *,
#     idx: int,
#     gam: FittedGAM,
#     fitted_terms: pd.DataFrame,
#     lims: tuple[int, int],
#     fit_kwargs: dict[str, Any] | None = None,
#     interval_kwargs:  dict[str, Any] | None = None,
#     partial_residual_kwargs:  dict[str, Any] | None = None,
#     n_standard_errors: int | float = 2,
#     ax: Axes | None = None,
#     ) -> Axes:  # TODO data default to None
#     """Plot 1D numeric terms (smooths or linear terms).

#     Args:
#         term: The term to plot.
#         gam: The fitted gam model.
#         data: Data used for adding partial residuals.
#         fit_kwargs: Argumnets passed to plotting of the line. Defaults to None.
#         interval_kwargs: Arguments passed to plotting of the confidence interval
#             lines. Defaults to None.
#         partial_residual_kwargs: Arguments passed to plotting of the partial
#             residuals. Defaults to None.
#         ax: Axes on to which the plot is applied. Defaults to None.
#     """
#     ax = plt.gca() if ax is None else ax
#     fit_kwargs = {} if fit_kwargs is None else fit_kwargs
#     interval_kwargs = {} if interval_kwargs is None else interval_kwargs
#     partial_residual_kwargs = {} if partial_residual_kwargs is None else partial_residual_kwargs

#     data = get_plot_data(idx=idx, fitted_terms=fitted_terms, gam=gam, lims=lims)
#     fit = to_py(data.rx2["fit"])
#     residuals = to_py(data.rx2["res"])  # TODO should be done by get_gata

#     # Add partial residuals first (better to be underneath)
#     color = partial_residual_kwargs.get("color", "black")
#     ax.scatter(residuals["x"], residuals["y"], color=color, **partial_residual_kwargs)

#     ax.plot(fit["x"], fit["y"], **fit_kwargs)

#     # Plot interval
#     color = interval_kwargs.get("color", ax.lines[-1].get_color())  # Match previous line color
#     linestyle = interval_kwargs.get("linestyle", "dotted")

#     for direction in [-1, 1]:
#         ax.plot(
#             fit["x"],
#             fit["y"] + direction*n_standard_errors*fit["se"],
#             linestyle=linestyle,
#             color=color,
#             **interval_kwargs,
#             )

#         # TODO Lost the x label :(
#     return ax


# def plot_2d(
#     idx: int,
#     gam: FittedGAM,
#     fitted_terms: pd.DataFrame,
#     x_lims: tuple[int, int],
#     y_lims: tuple[int, int],
#     colormesh_kwargs: dict,
#     n_standard_errors: int | float = 2,
#     ax: Axes | None = None,
# ) -> Axes:
#     """Plot smooth or tensor smooth terms of two variables as a colormesh.

#     Args:
#         term: The term to plot.
#         gam: The fitted gam model
#         density: _description_. Defaults to 50.
#         ax: _description_. Defaults to None.
#         colormesh_kwargs (dict | None, optional): _description_. Defaults to None.

#     Raises:
#         ValueError: _description_

#     Returns:
#         _type_: _description_
#     """
#     ax = plt.gca() if ax is None else ax
#     colormesh_kwargs = {} if colormesh_kwargs is None else colormesh_kwargs

#     # TODO argcheck for 2D
#     get_plot_data(
#         idx=idx,
#         fitted_terms=fitted_terms,
#         gam=gam,
#     )
#     mesh = ax.pcolormesh(
#         x1_mesh,
#         x2_mesh,
#         pred.reshape(x1_mesh.shape),
#         cmap=colormesh_kwargs.get("cmap", "RdBu"),
#         **colormesh_kwargs,
#     )
#     ax.figure.colorbar(mesh, ax=ax)
#     ax.set_xlabel(term.varnames[0])
#     ax.set_ylabel(term.varnames[1])
#     return ax


def plot_spline_on_sphere():
    raise NotImplementedError()

    # Get the term index in the gam

    # data = mgcViz.get_data(
    #     mgcViz.sm(
    #         mgcViz.getViz(
    #             gam.rgam,
    #             )
    #     )
    # )
