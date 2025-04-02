"First module."

from dataclasses import dataclass
from typing import Literal

import pandas as pd

type BSOptionStr = Literal[
    "tp",
    "ts",
    "ds",
    "cr",
    "cs",
    "cc",
    "sos",
    "bs",
    "ps",
    "cp",
    "re",
    "mrf",
    "gp",
    "so",
    "sw",
    "sf",
]


@dataclass
class Smooth:  # Not supported m, by, and more.
    variables: tuple[str, ...]
    k: int
    bs: str

    def __init__(
        self,
        *variables: str,
        k: int = -1,
        bs: BSOptionStr = "tp",
    ):

        self.variables = variables
        self.k = k
        self.bs = bs

    def __str__(self):
        return f"s({','.join(self.variables)},k={self.k},bs='{self.bs}')"
