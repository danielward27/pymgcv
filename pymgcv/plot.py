"""Plotting utility functions for gam models.

We will likely refactor this, e.g. to dispatch using structural subtyping of terms,
or be a method of the terms, but for now this gets us started
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from collections.abc import Iterable
from typing import Any
from pymgcv import terms
from pymgcv.gam import FittedGAM


# mgcViz = importr("mgcViz")


# TODO get default by saving limits in fitted_gam? Or by saving data in fitted_gam

def plot_1d(
    term: terms.Linear | terms.Smooth,
    gam: FittedGAM,
    xlims: tuple[int, int],
    data: dict[str, np.ndarray],
    fit_kwargs: dict[str, Any] | None = None,
    interval_kwargs:  dict[str, Any] | None = None,
    partial_residual_kwargs:  dict[str, Any] | None = None,
    ax: Axes | None = None,
    ):  # TODO data default to None
    """_summary_

    Args:
        term (terms.Linear | terms.Smooth): _description_
        gam (FittedGAM): _description_
        xlims (tuple[int, int]): _description_
        data (dict[str, np.ndarray]): Data used for adding partial residuals.
        fit_kwargs (_type_, optional): _description_. Defaults to None.
        interval_kwargs (_type_, optional): _description_. Defaults to None.
        partial_residual_kwargs (_type_, optional): _description_. Defaults to None.
        ax (Axes | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    ax = plt.gca() if ax is None else ax
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    interval_kwargs = {} if interval_kwargs is None else interval_kwargs
    partial_residual_kwargs = {} if partial_residual_kwargs is None else partial_residual_kwargs


    # Add partial residuals
    partial_residuals = gam.partial_residuals(term, data)
    color = partial_residual_kwargs.get("color", "black")
    ax.scatter(data[term.varnames[0]], partial_residuals, color=color, **partial_residual_kwargs)
    
    # TODO varnames or another attribute for extracting required data?
    x0_linspace = np.linspace(*xlims, num=100)
    pred = gam.predict_term(term, data={term.varnames[0]: x0_linspace})
    ax.plot(x0_linspace, pred["fit"], **fit_kwargs)


    # Plot interval
    color = interval_kwargs.get("color", ax.lines[-1].get_color())  # Match previous line color
    linestyle = interval_kwargs.get("linestyle", "dotted")

    ax.plot(
        x0_linspace,
        pred["fit"] + 1.96*pred["se_fit"],
        linestyle=linestyle,
        color=color,
        **interval_kwargs,
        )
    ax.plot(x0_linspace, pred["fit"] - 1.96*pred["se_fit"], linestyle=linestyle, color=color, **interval_kwargs)
    ax.set_xlabel(term.simple_string)
    return ax


def plot_smooth_2d(
    term: terms.Smooth,
    gam: FittedGAM,
    x1_lims: tuple[int, int],
    x2_lims: tuple[int, int],
    density: int = 50,
    ax: Axes | None = None,
    colormesh_kwargs: dict | None = None,
    ): # TODO does this also work with ti/te?
    ax = plt.gca() if ax is None else ax
    colormesh_kwargs = {} if colormesh_kwargs is None else colormesh_kwargs

    x1_mesh, x2_mesh = np.meshgrid(
        np.linspace(*x1_lims, density),
        np.linspace(*x2_lims, density),
    )

    pred = gam.predict_term(term, data={term.varnames[0]: x1_mesh.ravel(), term.varnames[1]: x2_mesh.ravel()})["fit"]
    mesh = ax.pcolormesh(
        x1_mesh,
        x2_mesh,
        pred.reshape(x1_mesh.shape),
        cmap=colormesh_kwargs.get("cmap", "RdBu"),
        )
    ax.figure.colorbar(mesh, ax=ax)
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(term.varnames[1])
    return ax


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

    


    

