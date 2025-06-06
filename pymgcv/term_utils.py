"""Utilities for constructing multiple related terms."""

from typing import TypeVar

import pandas as pd

from pymgcv.terms import Smooth, TensorSmooth

T = TypeVar("T", bound=Smooth | TensorSmooth)


def smooth_by_factor(
    *varnames: str,
    smooth_type: type[T],
    factor: pd.Series,
    drop_first: bool = False,
    **kwargs,
) -> tuple[list[T], pd.DataFrame]:
    """Constructs smooth terms and indicator variables for each level of a factor.

    This is a manual implementation - you should add the returned terms to the model,
    and add the covariates to your data.

    Warning:
        I have not tested the efficiency of this compared to the mgcv (non manual) way.
        For now though, this provides an option until we can better support the mgcv
        way.
    """
    if factor.dtype != "category" or factor.name is None:
        raise ValueError("Factor must be a series of categorical dtype with a name.")

    indicator_df = pd.get_dummies(
        factor,
        prefix=factor.name,
        prefix_sep="_",
        drop_first=drop_first,
        dtype=int,  # TODO Maybe bool is ok here?
    )

    smooths = [smooth_type(*varnames, **kwargs, by=col) for col in indicator_df.columns]
    return smooths, indicator_df
