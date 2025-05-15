"""Plotting utility functions for gam models.

We will likely refactor this, e.g. to dispatch using structural subtyping of terms,
or be a method of the terms, but for now this gets us started
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from collections.abc import Iterable

from pymgcv import terms
from pymgcv.gam import FittedGAM

# mgcViz = importr("mgcViz")


# TODO get default by saving limits in fitted_gam? Or by saving data in fitted_gam

def plot_1d(
    term: terms.Linear | terms.Smooth,
    gam: FittedGAM,
    xlims: tuple[int, int],
    fit_kwargs = None,
    interval_kwargs = None,
    ax: Axes | None = None,
    ):
    ax = plt.gca() if ax is None else ax
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    interval_kwargs = {} if interval_kwargs is None else interval_kwargs

    # TODO varnames or another attribute for extracting required data?
    x0_linspace = np.linspace(*xlims, num=100)
    pred = gam.predict_term(term, data={term.varnames[0]: x0_linspace})
    ax.plot(x0_linspace, pred["fit"])

    # Plot interval
    linestyle = interval_kwargs.get("linestyle", "dotted")
    color = interval_kwargs.get("color", ax.lines[-1].get_color())

    ax.plot(
        x0_linspace,
        pred["fit"] + 1.96*pred["se_fit"],
        linestyle=linestyle,
        color=color,
        )
    ax.plot(x0_linspace, pred["fit"] - 1.96*pred["se_fit"], linestyle=linestyle, color=color)
    ax.set_xlabel(term.simple_string)

    # TODO y axis label?
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
    ax.set_xlabel(term.simple_string)


    return ax

    # TODO add data points?
    # plt.scatter(data["x1"], data["x2"], c="black", s=5)
    # plt.ylabel("y")
    # plt.xlabel(terms[1].simple_string)
    # plt.show()

    # x0_linspace = np.linspace(data["x0"].min(), data["x0"].max(), num=100)
    # pred = fitted_gam.predict_term(terms[2], data={"x3": x0_linspace})
    # plt.plot(x0_linspace, pred["fit"])
    # plt.plot(x0_linspace, pred["fit"] + 2*pred["se_fit"], linestyle="dotted", color="black")
    # plt.plot(x0_linspace, pred["fit"] - 2*pred["se_fit"], linestyle="dotted", color="black")
    # plt.xlabel(str(terms[2]))
    # plt.ylabel("y")
    # plt.show()




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

    


    

