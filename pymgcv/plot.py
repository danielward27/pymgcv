"""Plotting utility functions for gam models.

We will likely refactor this, e.g. to dispatch using structural subtyping of terms,
or be a method of the terms, but for now this gets us started. Only plotting of
coninuous terms is currently supported.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from rpy2.robjects.packages import importr

from pymgcv import terms
from pymgcv.gam import FittedGAM

mgcViz = importr("mgcViz")


def plot_1d(
    term: terms.Linear | terms.Smooth,
    gam: FittedGAM,
    data: pd.DataFrame,
    plot_fit_kwargs: dict[str, Any] | None = None,
    plot_interval_kwargs: dict[str, Any] | None = None,
    plot_partial_residual_kwargs: dict[str, Any] | None = None,
    numeric_by_value: int | float = 1,
    ax: Axes | None = None,
) -> Axes:  # TODO data default to None
    """Plot 1D numeric terms (smooths or linear terms).

    Note, "by" variables are imputed as 1, i.e. showing the term prior to scaling
    by the "by" variable.

    Args:
        term: The term to plot.
        gam: The fitted gam model.
        data: Data used for adding partial residuals.
        plot_fit_kwargs: Argumnets passed to plotting of the line. Defaults to None.
        plot_interval_kwargs: Arguments passed to plotting of the confidence interval
            lines. Defaults to None.
        plot_partial_residual_kwargs: Arguments passed to plotting of the partial
            residuals. Defaults to None.
        ax: Axes on to which the plot is applied. Defaults to None.
    """
    ax = plt.gca() if ax is None else ax
    plot_fit_kwargs = {} if plot_fit_kwargs is None else plot_fit_kwargs
    plot_interval_kwargs = {} if plot_interval_kwargs is None else plot_interval_kwargs
    plot_partial_residual_kwargs = (
        {} if plot_partial_residual_kwargs is None else plot_partial_residual_kwargs
    )

    if len(term.varnames) != 1:
        raise ValueError(
            f"Expected term to be a function of a single variable, got {term.varnames}",
        )

    x = data[term.varnames[0]]

    # Add partial residuals first (better to be underneath)
    partial_residuals = gam.partial_residuals(term, data)
    color = plot_partial_residual_kwargs.get("color", "black")
    ax.scatter(x, partial_residuals, color=color, **plot_partial_residual_kwargs)

    # TODO again risky with different data types?
    x0_linspace = np.linspace(x.min(), x.max(), num=100)
    spaced_data = pd.DataFrame({term.varnames[0]: x0_linspace})

    if term.by is not None:
        spaced_data[term.by] = 1

    pred = gam.predict_term(term, data)
    ax.plot(x0_linspace, pred["fit"], **plot_fit_kwargs)

    # Plot interval
    color = plot_interval_kwargs.get(
        "color",
        ax.lines[-1].get_color(),
    )  # Match previous line color
    linestyle = plot_interval_kwargs.get("linestyle", "dotted")

    ax.plot(
        x0_linspace,
        pred["fit"] + 1.96 * pred["se"],
        linestyle=linestyle,
        color=color,
        **plot_interval_kwargs,
    )
    ax.plot(
        x0_linspace,
        pred["fit"] - 1.96 * pred["se"],
        linestyle=linestyle,
        color=color,
        **plot_interval_kwargs,
    )
    ax.set_xlabel(term.simple_string)
    return ax


# TODO get default by saving limits in fitted_gam? Or by saving data in fitted_gam


def plot_2d(
    term: terms.Smooth | terms.TensorSmooth,
    gam: FittedGAM,
    x1_lims: tuple[int, int],
    x2_lims: tuple[int, int],
    eval_points: int = 50,
    ax: Axes | None = None,
    colormesh_kwargs: dict | None = None,
) -> Axes:
    """Plot smooth or tensor smooth terms of two variables as a colormesh.

    Args:
        term: The term to plot.
        gam: The fitted gam model
        density: _description_. Defaults to 50.
        ax: _description_. Defaults to None.
        colormesh_kwargs (dict | None, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    ax = plt.gca() if ax is None else ax
    colormesh_kwargs = {} if colormesh_kwargs is None else colormesh_kwargs

    if len(term.varnames) != 2:
        raise ValueError(
            f"Expected term to be a function of a two variables, got {term.varnames}",
        )

    x1_mesh, x2_mesh = np.meshgrid(
        np.linspace(*x1_lims, eval_points),
        np.linspace(*x2_lims, eval_points),
    )

    pred = gam.predict_term(
        term,
        data=pd.DataFrame(
            {term.varnames[0]: x1_mesh.ravel(), term.varnames[1]: x2_mesh.ravel()},
            index=x1_mesh.ravel(),
        ),
    )["fit"]
    mesh = ax.pcolormesh(
        x1_mesh,
        x2_mesh,
        pred.reshape(x1_mesh.shape),
        cmap=colormesh_kwargs.get("cmap", "RdBu"),
        **colormesh_kwargs,
    )
    ax.figure.colorbar(mesh, ax=ax)
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(term.varnames[1])
    return ax


# def get_plot_data(
#     *,
#     idx: int,  # TODO name better imo
#     fitted_terms: pd.DataFrame,
#     gam: FittedGAM,
#     lims: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
# ):
#     data = mgcViz.get_data(
#         mgcViz.sm(gam.rgam, idx),
#         fitted_terms=data_to_rdf(fitted_terms),
#         gam=gam.rgam,
#         lims=ro.vectors.FloatVector(lims),
#     )
#     return data


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
