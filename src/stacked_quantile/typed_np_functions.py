"""Constrained typing for numpy functions.

I only use a few numpy functions in this project. These typed versions are restricted
to the subset of functionality I need.

:author: Shay Hill
:created: 2023-01-17
"""

from typing import Any, Callable, Protocol, cast

import numpy as np
import numpy.typing as npt

FPArray = npt.NDArray[np.floating[Any]]
SIArray = npt.NDArray[np.signedinteger[Any]]


class SearchSorted(Protocol):
    """Subset of np.searchsorted functionality."""

    def __call__(self, values: FPArray, target: float, side: str) -> int:
        """Return the index where target would be inserted in values."""
        ...


np_argsort = cast(Callable[[npt.NDArray[Any]], SIArray], np.argsort)
np_cumsum = cast(Callable[[FPArray], FPArray], np.cumsum)
np_isclose = cast(Callable[[float, float], bool], np.isclose)
np_searchsorted = cast(SearchSorted, np.searchsorted)
