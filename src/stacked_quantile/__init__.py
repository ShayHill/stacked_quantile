"""Import functions in main.py to the top level of the package.

:author: Shay Hill
:created: 2023-01-17
"""

from stacked_quantile.main import (
    get_stacked_median,
    get_stacked_medians,
    get_stacked_quantile,
    get_stacked_quantiles,
)

__all__ = [
    "get_stacked_quantile",
    "get_stacked_quantiles",
    "get_stacked_median",
    "get_stacked_medians",
]
