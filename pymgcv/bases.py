"""Basis functions for smooth terms in GAM models.

This module provides various basis function types that can be used with smooth
terms to control the shape and properties of the estimated smooth functions.
Different basis types are suitable for different kinds of data and modeling
requirements.

Available basis types:
- ThinPlateSpline: General-purpose, good default choice
- CubicSpline: Fast, interpretable, good for large datasets
- BSpline: Flexible with local support properties
- PSpline: Penalized B-splines with difference penalties
- RandomEffect: For grouped data with random intercepts/slopes
- DuchonSpline: Generalization of thin plate splines
- SplineOnSphere: For latitude/longitude geographical data
- MarkovRandomField: For spatial data with neighborhood structure

All basis functions implement the BasisLike protocol.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from rpy2.robjects import ListVector


@runtime_checkable
class BasisLike(Protocol):
    """Protocol defining the interface for GAM basis functions.

    All basis function classes must implement this protocol to be usable
    with smooth terms. The protocol ensures basis functions can be converted
    to appropriate mgcv R syntax and provide any additional parameters needed.

    Methods:
        __str__: Convert to mgcv basis string identifier (e.g., 'tp', 'cr', 'cs')
        get_xt: Return additional parameters for complex basis types
    """

    def __str__(self) -> str:
        """Convert basis to mgcv string identifier.

        Returns:
            Two-letter string identifier used by mgcv (e.g., 'tp', 'cr', 'bs')
        """
        ...

    def get_xt(self) -> ListVector | None:
        """Get additional parameters for the basis function.

        Some complex basis types require additional parameters beyond the
        standard mgcv arguments. This method returns an R ListVector with
        any such parameters, or None if no additional parameters are needed.

        Returns:
            R ListVector with additional parameters, or None
        """
        ...


class RandomEffect(BasisLike):
    """Random effect basis for grouped data.

    Creates random intercepts or slopes for grouped data. This is equivalent
    to adding group-specific adjustments that are penalized toward zero,
    implementing a form of shrinkage estimation.

    Use random effects when:
    - You have grouped/clustered data (subjects, sites, batches, etc.)
    - You want to account for group-level variation
    - You need shrinkage estimation for groups with little data
    - You're modeling hierarchical or nested data structures

    Examples:
        ```python
        # Random intercepts by subject
        random_subject = Smooth('subject_id', bs=RandomEffect())

        # Random slopes (requires careful data setup)
        random_slope = Smooth('subject_slope', bs=RandomEffect())

        # Use in hierarchical model
        spec = ModelSpecification(
            response_predictors={'y': [
                Smooth('x'),                    # Fixed smooth
                Smooth('subject', bs=RandomEffect())  # Random intercepts
            ]}
        )
        ```

    Note:
        The grouping variable should typically be a factor or integer identifier.
        For random slopes, you need to create an appropriate design variable
        that represents the slope term for each group.
    """

    def __str__(self) -> str:
        """Return mgcv identifier for random effects."""
        return "re"

    def get_xt(self) -> ListVector | None:
        """No additional parameters needed for basic random effects."""
        return None


@dataclass(kw_only=True)
class ThinPlateSpline(BasisLike):
    """Thin plate regression spline basis.

    Thin plate splines are the default basis for smooth terms in mgcv. They
    provide a good general-purpose smooth that works well for most applications.
    The basis functions are defined to minimize a roughness penalty based on
    integrated squared second derivatives.

    Thin plate splines are:
    - Scale-invariant (rotation and translation invariant)
    - Good for moderate sample sizes (< 10,000 observations)
    - Suitable for both univariate and multivariate smoothing
    - Theoretically well-founded (minimize roughness functionals)

    Use thin plate splines when:
    - You want a reliable default choice
    - Working with moderate-sized datasets
    - Need good performance across different data types
    - Unsure which basis to choose

    Args:
        shrinkage: If True, applies additional penalty to the null space
            (unpenalized part) of the smooth. This helps with identifiability
            and can improve model selection. Default is False.

    Examples:
        ```python
        # Standard thin plate spline (default)
        tps_default = ThinPlateSpline()

        # With shrinkage penalty on null space
        tps_shrink = ThinPlateSpline(shrinkage=True)

        # Use in smooth term
        smooth_tps = Smooth('x', bs=ThinPlateSpline(shrinkage=True))
        ```

    Note:
        For large datasets (> 10,000 observations), consider using CubicSpline
        or PSpline for better computational efficiency.
    """

    shrinkage: bool | None = False

    def __str__(self) -> str:
        """Return mgcv identifier: 'ts' for shrinkage, 'tp' for standard."""
        return "ts" if self.shrinkage else "tp"

    def get_xt(self):
        """No additional parameters needed for thin plate splines."""
        return


@dataclass(kw_only=True)
class CubicSpline(BasisLike):
    """Cubic regression spline basis.

    Cubic splines use piecewise cubic polynomials with knots placed throughout
    the data range. They're computationally efficient and work well for large
    datasets. The splines are constrained to be continuous with continuous
    first and second derivatives at the knots.

    Cubic splines are:
    - Computationally fast and memory efficient
    - Good for large datasets (> 10,000 observations)
    - Interpretable (piecewise polynomials)
    - Suitable for univariate smoothing

    Use cubic splines when:
    - Working with large datasets where speed matters
    - You need computational efficiency
    - The relationship is reasonably smooth
    - You want interpretable smooth functions

    Args:
        cyclic: If True, creates a cyclic spline where the function values
            and derivatives match at the boundaries. Use for periodic data
            like time of day, angles, or seasonal patterns. Default is False.
        shrinkage: If True, adds penalty to the null space (linear component).
            Helps with model selection and identifiability. Default is False.
            Cannot be used with cyclic=True.

    Examples:
        ```python
        # Standard cubic spline
        cubic_default = CubicSpline()

        # Cyclic spline for periodic data
        cubic_cyclic = CubicSpline(cyclic=True)

        # With shrinkage penalty
        cubic_shrink = CubicSpline(shrinkage=True)

        # Use with smooth terms
        time_smooth = Smooth('hour', bs=CubicSpline(cyclic=True))  # 0-24 hours
        trend_smooth = Smooth('x', bs=CubicSpline(shrinkage=True))
        ```

    Raises:
        ValueError: If both cyclic and shrinkage are True (incompatible options)

    Note:
        For multivariate smoothing, consider ThinPlateSpline or use tensor
        products of univariate cubic splines.
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

    def get_xt(self):
        """No additional parameters needed for cubic splines."""
        return


@dataclass(kw_only=True)
class DuchonSpline(BasisLike):
    """Duchon spline basis - a generalization of thin plate splines.

    Duchon splines extend thin plate splines by allowing different smoothness
    assumptions. They're particularly useful for spatial modeling and when
    you need more control over the smoothness properties than standard thin
    plate splines provide.

    Duchon splines are:
    - More flexible than thin plate splines in smoothness assumptions
    - Good for spatial and geographical data
    - Suitable for irregular data distributions
    - Theoretically robust for various smoothness requirements

    Use Duchon splines when:
    - Standard thin plate splines are too restrictive
    - Working with spatial or geographical data
    - You need more flexible smoothness assumptions
    - Data has irregular spatial distribution

    Examples:
        ```python
        # Duchon spline for spatial coordinates
        spatial_smooth = Smooth('x_coord', 'y_coord', bs=DuchonSpline())

        # Use in spatial model
        spec = ModelSpecification(
            response_predictors={'response': [
                Smooth('longitude', 'latitude', bs=DuchonSpline()),
                Linear('elevation')
            ]}
        )
        ```

    Note:
        Currently, the penalty order (m parameter) is not configurable and
        uses mgcv defaults. This may be extended in future versions.
    """

    def __str__(self) -> str:
        """Return mgcv identifier for Duchon splines."""
        return "ds"

    def get_xt(self):
        """No additional parameters needed for basic Duchon splines."""
        return


@dataclass(kw_only=True)
class SplineOnSphere(BasisLike):
    """Spline basis for data on a sphere (latitude/longitude coordinates).

    This basis is specifically designed for smooth functions of latitude and
    longitude coordinates, respecting the spherical geometry of the Earth.
    It correctly handles the fact that longitude lines converge at the poles
    and that the sphere wraps around.

    Splines on sphere:
    - Respect spherical geometry (no edge effects at poles)
    - Handle longitude wraparound correctly (e.g., -180° = 180°)
    - Account for variable longitude spacing at different latitudes
    - Appropriate for global spatial data

    Use splines on sphere when:
    - Working with latitude/longitude coordinates
    - Modeling global spatial patterns
    - Data spans large geographical areas
    - You need to respect Earth's spherical geometry

    Requirements:
    - Must be used with exactly two variables
    - Variables should represent latitude and longitude in degrees
    - Latitude should be in range [-90, 90]
    - Longitude should be in range [-180, 180] or [0, 360]

    Examples:
        ```python
        # Smooth over geographical coordinates
        geo_smooth = Smooth('latitude', 'longitude', bs=SplineOnSphere())

        # Global climate model
        spec = ModelSpecification(
            response_predictors={'temperature': [
                Smooth('lat', 'lon', bs=SplineOnSphere()),
                Smooth('elevation'),
                Linear('year')
            ]}
        )

        # Ocean data with geographic coordinates
        ocean_model = Smooth('lat_deg', 'lon_deg', bs=SplineOnSphere(), k=100)
        ```

    Note:
        This basis is computationally more expensive than standard thin plate
        splines but is essential for proper spatial modeling on geographical scales.
    """

    def __str__(self) -> str:
        """Return mgcv identifier for splines on sphere."""
        return "sos"

    def get_xt(self):
        """No additional parameters needed for splines on sphere."""
        return


@dataclass(kw_only=True)
class BSpline(BasisLike):
    """B-spline basis with integrated squared derivative penalties.

    B-splines (basis splines) are piecewise polynomials with local support,
    meaning each basis function is non-zero only over a small part of the
    domain. This gives them good numerical properties and allows for local
    control of the smooth function.

    B-splines are:
    - Numerically stable and well-conditioned
    - Have local support (changing data in one region affects only nearby fit)
    - Computationally efficient
    - Flexible and can approximate complex functions well

    Use B-splines when:
    - You need local control over the smooth function
    - Working with data that has local features or discontinuities
    - Numerical stability is important
    - You want good approximation properties

    Examples:
        ```python
        # Standard B-spline basis
        bspline = BSpline()

        # Use in smooth term
        local_smooth = Smooth('x', bs=BSpline(), k=20)

        # Good for functions with local features
        spec = ModelSpecification(
            response_predictors={'y': [
                Smooth('x', bs=BSpline()),  # Local control
                Linear('z')                # Global linear effect
            ]}
        )
        ```

    Note:
        B-splines work particularly well when you have prior knowledge about
        where the function might have features or changes in behavior, as you
        can control the knot placement through the basis dimension (k parameter).
    """

    def __str__(self) -> str:
        """Return mgcv identifier for B-splines."""
        return "bs"

    def get_xt(self):
        """No additional parameters needed for B-splines."""
        return


@dataclass(kw_only=True)
class PSpline(BasisLike):
    """P-spline (penalized spline) basis as proposed by Eilers and Marx (1996).

    P-splines combine B-spline basis functions with difference penalties on
    the coefficients. This approach provides a good balance between flexibility
    and smoothness, with excellent computational properties for large datasets.

    P-splines are:
    - Computationally efficient for large datasets
    - Numerically stable
    - Provide good balance of flexibility and smoothness
    - Have well-understood statistical properties

    The penalty is applied to differences between adjacent coefficients,
    encouraging smooth transitions between basis functions while allowing
    flexibility where the data demands it.

    Use P-splines when:
    - Working with large datasets where efficiency matters
    - You want robust, reliable smoothing
    - Computational speed is important
    - You need a good general-purpose basis

    Args:
        cyclic: If True, creates a cyclic P-spline where the function values
            match at the boundaries. Use for periodic data like seasonal
            patterns, time of day, or angular measurements. Default is False.

    Examples:
        ```python
        # Standard P-spline
        pspline = PSpline()

        # Cyclic P-spline for seasonal data
        seasonal = PSpline(cyclic=True)

        # Use in smooth terms
        trend_smooth = Smooth('time', bs=PSpline())
        seasonal_smooth = Smooth('day_of_year', bs=PSpline(cyclic=True))

        # Efficient model for large dataset
        spec = ModelSpecification(
            response_predictors={'y': [
                Smooth('x1', bs=PSpline()),           # Efficient smooth
                Smooth('month', bs=PSpline(cyclic=True))  # Seasonal
            ]}
        )
        ```

    Note:
        P-splines are often a good alternative to thin plate splines when
        computational efficiency is important, especially for univariate smoothing.
    """

    cyclic: bool = False

    def __str__(self) -> str:
        """Return mgcv identifier: 'cp' for cyclic, 'ps' for standard."""
        return "cp" if self.cyclic else "ps"

    def get_xt(self):
        """No additional parameters needed for P-splines."""
        return


@dataclass(kw_only=True)
class MarkovRandomField(BasisLike):
    """Markov Random Field basis for spatial data with neighborhood structure.

    MRF basis functions are designed for spatial data where observations have
    a clear neighborhood structure (e.g., geographical regions, pixels in an
    image, nodes in a network). The smoothing penalty encourages similar values
    in neighboring locations.

    This basis type is particularly useful for:
    - Spatial data with irregular boundaries (e.g., countries, states, districts)
    - Network data where nodes have connections
    - Image analysis where pixels have spatial relationships
    - Any data with a predefined neighborhood structure

    Args:
        polys: List of numpy arrays defining the spatial polygons or
            neighborhood structure. Each array represents the boundary
            or connectivity information for a spatial unit.

    Examples:
        ```python
        # This basis type requires complex setup and is not fully implemented
        # It would typically be used for data like:

        # Administrative regions (countries, states, etc.)
        # mrf_basis = MarkovRandomField(polys=region_boundaries)
        # region_smooth = Smooth('region_id', bs=mrf_basis)

        # Network nodes with adjacency structure
        # mrf_network = MarkovRandomField(polys=adjacency_matrices)
        # network_smooth = Smooth('node_id', bs=mrf_network)
        ```

    Warning:
        This basis type is not fully implemented in the current version.
        The get_xt() method needs to be completed to properly pass the
        spatial structure information to mgcv.

    Note:
        Implementation of this basis requires careful handling of the spatial
        structure data and conversion to appropriate R objects for mgcv.
    """

    polys: list[np.ndarray]

    def __str__(self) -> str:
        """Return mgcv identifier for Markov Random Fields."""
        return "mrf"

    def get_xt(self):
        """Return spatial structure parameters - NOT YET IMPLEMENTED.

        Raises:
            NotImplementedError: This method needs to be implemented to
                convert the polys data to appropriate R objects for mgcv.
        """
        return NotImplementedError()
