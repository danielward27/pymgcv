from collections.abc import Sequence
from functools import partial

from pymgcv.bases import AbstractBasis

# TODO not supported sp, pc.
# xt not needed as will be handled with basis.


def smooth(
    *varnames: str,
    k: int | None = None,
    bs: AbstractBasis | None = None,
    m: int | None = None,
    by: str | None = None,
    id: str | None = None,
    fx: bool = False,
) -> str:
    """Returns the mgcv smooth term as a string."""
    fx = None if not fx else fx  # type: ignore # Map to None for filtering
    kwargs = {"k": k, "bs": bs, "m": m, "by": by, "id": id, "fx": fx}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    map_to_str = {  # Define how to map to argument in formula string
        "bs": lambda bs: f"'{bs}'",
        "id": lambda id: f"'{id}'",
        "fx": lambda fx: str(fx).upper(),
    }

    smooth_string = f"s({",".join(varnames)}"
    for key, val in kwargs.items():
        smooth_string += f",{key}={map_to_str.get(key, str)(val)}"
    return smooth_string + ")"


def tensor_smooth(
    *varnames: str,
    k: Sequence[int] | None = None,
    bs: Sequence[AbstractBasis] | None = None,
    d: Sequence[int] | None = None,
    m: Sequence[int] | None = None,
    by: str | None = None,
    id: str | None = None,
    fx: bool = False,
    np: bool = True,
    interaction_only: bool = False,
) -> str:
    """Returns the mgcv tensor smooth term as a string."""
    # Map to None if matching default
    np = None if np else np  # type: ignore
    fx = None if not fx else fx  # type: ignore

    kwargs = {"k": k, "bs": bs, "d": d, "m": m, "by": by, "id": id, "fx": fx, "np": np}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    map_to_str = {  # Define how to map to argument in formula string
        "k": _sequence_to_rvec_str,
        "bs": partial(_sequence_to_rvec_str, converter=lambda bs: f"'{bs}'"),
        "d": _sequence_to_rvec_str,
        "m": _sequence_to_rvec_str,
        "id": lambda id: f"'{id}'",
        "fx": lambda fx: str(fx).upper(),
        "np": lambda np: str(np).upper(),
    }

    smooth_string = f"{"ti" if interaction_only else "te"}({",".join(varnames)}"
    for key, val in kwargs.items():
        smooth_string += f",{key}={map_to_str.get(key, str)(val)}"
    return smooth_string + ")"


def _sequence_to_rvec_str(sequence: Sequence, converter=str):
    component = ",".join(converter(el) for el in sequence)
    return f"c({component})"
