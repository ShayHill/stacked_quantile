"""Constrained typing for numpy functions.

I only use a few numpy functions in this project. These typed versions are restricted
to the subset of functionality I need.

:author: Shay Hill
:created: 2023-01-17
"""

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

FPArray: TypeAlias = npt.NDArray[np.floating[Any]]
SIArray: TypeAlias = npt.NDArray[np.signedinteger[Any]]


class SearchSorted(Protocol):
    """Subset of np.searchsorted functionality."""

    def __call__(self, values: FPArray, target: float, side: str) -> int:
        """Return the index where target would be inserted in values."""
        ...


class IsClose(Protocol):
    """Subset of np.isclose functionality."""

    def __call__(
        self, a: float, b: float, rtol: float = 1e-9, atol: float = 1e-9
    ) -> bool:
        """Return True if a and b are close within tolerance."""
        ...


np_argsort = cast("Callable[[npt.NDArray[Any]], SIArray]", np.argsort)
np_cumsum = cast("Callable[[FPArray], FPArray]", np.cumsum)
np_isclose = cast("IsClose", np.isclose)
np_searchsorted = cast("SearchSorted", np.searchsorted)
