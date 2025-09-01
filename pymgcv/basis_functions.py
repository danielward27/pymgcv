"""Basis functions for smooth terms in GAM models.

This module provides various basis function types that can be used with smooth
terms to control the shape and properties of the estimated smooth functions.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

# The design here is a bit strange and warrants some explanation. We want a clean
# interface, which means that ideally we can pass arguments to the basis that relate to
# setting up the basis. MGCV instead uses xt and m to mean various things for different
# basis functions, and passes this to s. To avoid that we store them with the basis and
# provide a _pass_to_s method. Whilst it would be nice to just create the ro.RObjects as
# a dictionary of kwargs and pass them to s, because s is evaluated in a formula
# context, we have to map xt and m to strings (evaluated literally) or variable names.
# xt can be large (e.g. arrays) so we cannot map this to a string, so we map it to
# a variable name. m tends to be small, so we store it as a python variable and convert
# to a string in the formula.
from typing import TypedDict

import numpy as np
import rpy2.robjects as ro


class _PassToS(TypedDict, total=False):
    xt: ro.ListVector
    m: int | float | ro.IntVector | ro.FloatVector


class AbstractBasis(ABC):
    """Abstract class defining the interface for GAM basis functions.

    All basis function classes must implement this protocol to be usable
    with smooth terms. The protocol ensures basis functions can be converted
    to appropriate mgcv R syntax and provide any additional parameters needed.
    """

    @abstractmethod
    def __str__(self) -> str:
        """Convert basis to mgcv string identifier.

        Returns:
            String identifier used by mgcv (e.g., 'tp', 'cr', 'bs')
        """
        ...

    @abstractmethod
    def _pass_to_s(self) -> _PassToS:
        """Basis specific arguments to pass to mgcv.s.

        This handles:
        - `m`:  Basis penalties, orders, and more.
        - `xt`: Various arguments for specific basis functions.

        Some arguments to `mgcv.s` are specific to the basis funcions used,
        primarily `m` and `xt`. To make a more intuitive interface, we handle
        passing of these arguments to `mgcv.s`. Note the string should
        include the leading comma if it is not empty.



        Returns:
            A dictionary, mapping a keyword (e.g. xt or m) to a dictionary of
            variable values.
        """
        ...


@dataclass
class RandomEffect(AbstractBasis):
    """Random effect basis for correlated grouped data.

    This can be used with any mixture of numeric or categorical variables. Acts
    similarly to an [`Interaction`][pymgcv.terms.Interaction] but penalizes
    the corresponding coefficients with a multiple of the identity matrix (i.e. a ridge
    penalty), corresponding to an assumption of i.i.d. normality of the parameters.

    !!! warning

        Numeric variables (int/float), will be treated as a linear term with a single
        penalized slope parameter. Do not use an integer variable to encode
        categorical groups!

    !!! example

        For an example, see the
        [supplement vs placebo example](../../examples/supplement_vs_placebo).

    """

    def __str__(self) -> str:
        """Return mgcv identifier for random effects."""
        return "re"

    def _pass_to_s(self) -> _PassToS:
        return {}


@dataclass(kw_only=True, frozen=True)
class ThinPlateSpline(AbstractBasis):
    """Thin plate regression spline basis.

    Args:
        shrinkage: If True, the penalty is modified so that the term is shrunk to zero
            for a high enough smoothing parameter.
        m: The order of the derivative in the thin plate spline penalty. If $d$ is the
            number of covariates for the smooth term, this must satisfy $m>(d+1)/2$. If
            left to None, the smallest value satisfying $m>(d+1)/2$ will be used, which
            creates "visually smooth" functions.
        max_knots: The maximum number of knots to use. Defaults to 2000.
    """

    shrinkage: bool | None = False
    m: int | None = None
    max_knots: int | None = None

    def __str__(self) -> str:
        """Return mgcv identifier: 'ts' for shrinkage, 'tp' for standard."""
        return "ts" if self.shrinkage else "tp"

    def _pass_to_s(self) -> _PassToS:
        pass_to_s: _PassToS = {}
        if self.m is not None:
            pass_to_s["m"] = self.m
        if self.max_knots is not None:
            listvec = ro.ListVector([self.max_knots])
            listvec.names = ["max.knots"]
            pass_to_s["xt"] = listvec
        return pass_to_s


@dataclass
class FactorSmooth(AbstractBasis):
    """S for each level of a categorical variable.

    When using this basis, the first variable of the smooth should
    be a numeric variable, and the second should be a categorical variable.

    Unlike using a categorical by variable e.g. `S(x, by="group")`:

    - The terms share a smoothing parameter.
    - The terms are fully penalized, with seperate penalties on each null space
        component (e.g. intercepts). The terms are non-centered, and can
        be used with an intercept without introducing indeterminacy, due to the
        penalization.

    Args:
        bs: Any singly penalized basis function. Defaults to
            `ThinPlateSpline`. Only the type of the basis is passed
            to mgcv (i.e. what is returned by `str(bs)`). This is a limitation
            of mgcv (e.g. you cannot do )
            mgcv provides no way to pass more details for setting up the
            basis function.
    """

    bs: AbstractBasis = field(default_factory=ThinPlateSpline)

    def __str__(self) -> str:
        """Return mgcv identifier for random effects."""
        return "fs"

    def _pass_to_s(self) -> _PassToS:
        listvec = ro.ListVector({"bs": str(self.bs)})
        to_s = self.bs._pass_to_s()
        if "xt" in to_s:
            to_s["xt"] = to_s["xt"] + listvec
        else:
            to_s["xt"] = listvec
        return to_s


@dataclass(kw_only=True)
class CubicSpline(AbstractBasis):
    """Cubic regression spline basis.

    Cubic splines use piecewise cubic polynomials with knots placed throughout
    the data range. They tend to be computationally efficient, but often
    performs slightly worse than thin plate splines and are limited to
    univariate smooths. Note the limitation of being restricted to
    one-dimensional smooths does not imply they cannot be used for
    multivariate [`T`][pymgcv.terms.T] smooths,
    which are constructed from marginal bases.

    Args:
        cyclic: If True, creates a cyclic spline where the function values
            and derivatives match at the boundaries. Use for periodic data
            like time of day, angles, or seasonal patterns. Default is False.
        shrinkage: If True, adds penalty to the null space (linear component).
            Helps with model selection and identifiability. Default is False.
            Cannot be used with cyclic=True.

    Raises:
        ValueError: If both cyclic and shrinkage are True (incompatible options)
    """

    shrinkage: bool = False
    cyclic: bool = False

    def __post_init__(self):
        """Validate cubic spline configuration."""
        if self.cyclic and self.shrinkage:
            raise ValueError("Cannot use both cyclic and shrinkage simultaneously.")

    def __str__(self) -> str:
        """Return mgcv identifier: 'cs', 'cc', or 'cr'."""
        return "cs" if self.shrinkage else "cc" if self.cyclic else "cr"

    def _pass_to_s(self) -> _PassToS:
        """No additional parameters needed for cubic splines."""
        return {}


@dataclass(kw_only=True)
class DuchonSpline(AbstractBasis):
    """Duchon spline basis - a generalization of thin plate splines.

    These smoothers allow the use of lower orders of derivative in the penalty than
    conventional thin plate splines, while still yielding continuous functions.

    The description, adapted from mgcv is as follows: Duchon’s (1977) construction
    generalizes the usual thin plate spline penalty as follows. The usual thin plate
    spline penalty is given by the integral of the squared Euclidian norm of a vector of
    mixed partial $m$-th order derivatives of the function w.r.t. its arguments. Duchon
    re-expresses this penalty in the Fourier domain, and then weights the squared norm
    in the integral by the Euclidean norm of the fourier frequencies, raised to the
    power $2s$, where $s$ is a user selected constant.

    If $d$ is the number of arguments of the smooth:

    - It is required that $-d/2 < s < d/2$.
    - If $s=0$ then the usual thin plate spline is recovered.
    - To obtain continuous functions we further require that $m + s > d/2$.

    For example, ``DuchonSpline(m=1, s=d/2)`` can be used in order to use first
    derivative penalization for any $d$, and still yield continuous functions.

    Args:
        m : Order of derivative to penalize.
        s : $s$ as described above, should be an integer divided by 2.
    """

    m: int = 2
    s: float | int = 0

    def __str__(self) -> str:
        """Return mgcv identifier for Duchon splines."""
        return "ds"

    def _pass_to_s(self) -> _PassToS:
        return {"m": ro.FloatVector([self.m, self.s])}


@dataclass(kw_only=True)
class SplineOnSphere(AbstractBasis):
    """Isotropic smooth for data on a sphere (latitude/longitude coordinates).

    This should be used with exactly two variables, where the first represents latitude
    on the interval [-90, 90] and the second represents longitude on the interval [-180,
    180].

    Args:
        m : An integer in [-1, 4]. Setting `m=-1` uses
            [`DuchonSpline(m=2,s=1/2)`](`pymgcv.basis_functions.DuchonSpline`). Setting
            `m=0` signals to use the 2nd order spline on the sphere, computed by
            Wendelberger’s (1981) method. For m>0, (m+2)/2 is the penalty order, with
            m=2 equivalent to the usual second derivative penalty.
    """

    m: int = 0

    def __str__(self) -> str:
        """Return mgcv identifier for splines on sphere."""
        return "sos"

    def _pass_to_s(self) -> _PassToS:
        """No additional parameters needed for splines on sphere."""
        return {}


@dataclass
class BSpline(AbstractBasis):
    """B-spline basis with derivative-based penalties.

    These are univariate (but note univariate smooths can be used for multivariate
    smooths constructed with [`T`][pymgcv.terms.T]).
    ``BSpline(degree=3, penalty_orders=[2])`` constructs a conventional cubic spline.

    Args:
        degree: The degree of the B-spline basis (e.g. 3 for a cubic spline).
        penalty_orders: The derivative orders to penalize. Default to [degree - 1].
    """

    degree: int
    penalty_orders: list[int]

    def __init__(self, *, degree: int = 3, penalty_orders: Iterable[int] | None = None):
        if penalty_orders is None:
            penalty_orders = [degree - 1]
        self.degree = degree
        self.penalty_orders = list(penalty_orders)

    def __str__(self) -> str:
        """Return mgcv identifier for B-splines."""
        return "bs"

    def _pass_to_s(self) -> _PassToS:
        return {"m": ro.IntVector([self.degree] + self.penalty_orders)}


@dataclass(kw_only=True)
class PSpline(AbstractBasis):
    """P-spline (penalized spline) basis as proposed by Eilers and Marx (1996).

    Uses B-spline bases penalized by discrete penalties applied directly to the basis
    coefficients. Note for most use cases splines with derivative-based penalties (e.g.
    [`ThinPlateSpline`][pymgcv.basis_functions.ThinPlateSpline] or
    [`CubicSpline`][pymgcv.basis_functions.CubicSpline]) tend to yield better
    MSE performance. ``BSpline(degree=3, penalty_order=2)`` is
    cubic-spline-like.

    Args:
        degree: Degree of the B-spline basis (e.g. 3 for cubic).
        penalty_order: The difference order to penalize. 0-th order is ridge penalty.
            Default to `degree-1`.
    """

    cyclic: bool = False
    degree: int
    penalty_order: int

    def __init__(self, *, degree: int = 3, penalty_order: int | None = None):
        self.degree = degree
        self.penalty_order = penalty_order if penalty_order is not None else degree - 1

    def __str__(self) -> str:
        """Return mgcv identifier: 'cp' for cyclic, 'ps' for standard."""
        return "cp" if self.cyclic else "ps"

    def _pass_to_s(self) -> _PassToS:
        # Note (unlike b-splines) seems mgcv uses m[1] for the penalty order, not degree so subtract 1
        return {"m": ro.IntVector([self.degree - 1, self.penalty_order])}


@dataclass(kw_only=True)
class MarkovRandomField(AbstractBasis):
    """Markov Random Field basis for discrete spatial data with neighborhood structure.

    The smoothing penalty encourages similar value in neighboring locations. When using
    this basis, the variable passed to [`S`][pymgcv.terms.S] should be a
    categorical variable representing the area labels.

    Args:
        polys: List of numpy arrays defining the spatial polygons or
            neighborhood structure. Each array represents the boundary
            or connectivity information for a spatial unit.
    """

    polys: list[np.ndarray]

    def __str__(self) -> str:
        """Return mgcv identifier for Markov Random Fields."""
        return "mrf"

    def _pass_to_s(self) -> _PassToS:
        """Return spatial structure parameters - NOT YET IMPLEMENTED."""
        raise NotImplementedError()
