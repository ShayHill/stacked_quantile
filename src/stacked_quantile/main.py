"""'Stacked' quantile functions. Close to weighted quantile functions.

These functions are used to calculate quantiles of a set of values, where each value
has a weight. The typical process for calculating a weighted quantile is to create a
CDF from the weights, then interpolate the values to find the quantile.

These functions, however, treat weighted values (given integer weights) exactly as
multiple values. So values (1, 2, 3) with weights (4, 5, 6) will be treated as
(1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3). If the quantile falls exactly between
two values, the non-weighted average of the two values is returned. This is
consistent with the "weights as occurrences" interpretation. Strips all zero-weight
values, so these will never be included in such averages.

If using non-integer weights, the results will be as if some scalar were applied to
make all weights into integers.

This "weights as occurrences" interpretation has two pitfalls:

    1.  Identical values will be returned for different quantiles (e.g., the results
        for quantiles == 0.5, 0.6, and 0.7 might be identical). The effect of this is
        that some some common data practices like "robust scalar" will *not* be
        robust because of the potential for a 0 interquartile range. Again this is
        consistent, because the same thing could happen with repeated, non-weighted
        values.

    2.  With any number of values, the stacked_median could still be the first or
        last value (if it has enough weight), so separating by the median is not
        robust. This could also happen with repeaded, non-weighted values. The
        workaround is to divide the values into group_a = values strictly < median,
        group_b = values strictly > median, then add == median to the smaller group.

:author: Shay Hill
:created: 2023-01-17
"""

from typing import cast

import numpy as np
import numpy.typing as npt

from stacked_quantile.typed_np_functions import (
    FPArray,
    np_argsort,
    np_cumsum,
    np_isclose,
    np_searchsorted,
)


def _sort_values_by_weights(
    values: FPArray, weights: FPArray
) -> tuple[FPArray, FPArray]:
    """Sort values by weights."""
    sorter = np_argsort(values)
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]
    return sorted_values, sorted_weights


def validate_values_and_weights(
    values: npt.ArrayLike, weights: npt.ArrayLike
) -> tuple[FPArray, FPArray]:
    """Validate values and weights.

    :param values: array of values with shape (n,)
    :param weights: array of weights with shape (n,)
    :return: tuple of validated values and weights (cast as arrays)
    :raises ValueError: if values and weights are not at least 1-dimensional
    :raises ValueError: if values array is empty
    :raises ValueError: if values and weights are not the same length
    :raises ValueError: if weights are not non-negative
    """
    avalues = np.asarray(values)
    aweights = np.asarray(weights)
    if avalues.ndim == 0 or aweights.ndim == 0:
        msg = "values and weights must be at least 1-dimensional"
        raise ValueError(msg)
    if len(avalues) == 0:
        msg = "values array is empty"
        raise ValueError(msg)
    if avalues.shape[-1] != aweights.shape[-1]:
        msg = "values and weights must be the same length"
        raise ValueError(msg)
    if np.any(aweights < 0):
        msg = "weights must be non-negative"
        raise ValueError(msg)
    if np.all(aweights == 0):
        aweights = np.ones_like(aweights)
    return avalues, aweights


def get_stacked_quantile(
    values: npt.ArrayLike, weights: npt.ArrayLike, quantile: float
) -> float:
    """Get a weighted quantile for a vector of values.

    :param values: array of values with shape (n,)
    :param weights: array of weights where weights.shape == values.shape
    :param quantile: quantile to calculate, in [0, 1]
    :return: weighted quantile of values
    :raises ValueError: if values and weights do not have the same length
    :raises ValueError: if quantile is not in interval [0, 1]
    :raises ValueError: if values array is empty (after removing zero-weight values)
    :raises ValueError: if weights are not all positive

    Exclude any zero-weight values from the calculation. If all weights are zero, then
    return the unweighted median.
    """
    if quantile < 0 or quantile > 1:
        msg = "quantile must be in interval [0, 1]"
        raise ValueError(msg)

    avalues, aweights = validate_values_and_weights(values, weights)
    avalues, aweights = avalues[aweights != 0], aweights[aweights != 0]

    sorted_values, sorted_weights = _sort_values_by_weights(avalues, aweights)
    cum_aweights = np_cumsum(sorted_weights)
    target = cum_aweights[-1] * quantile
    index = np_searchsorted(cum_aweights, target, side="right")
    if index == 0:
        return sorted_values[0]
    if index == len(sorted_values):
        return sorted_values[-1]
    if np_isclose(cum_aweights[index - 1], target, rtol=1e-9, atol=1e-9):
        lower = sorted_values[index - 1]
        upper = sorted_values[index]
        avg = (lower + upper) / 2
        if not isinstance(avg, float):
            # strictly for type narrowing
            msg = "avg is not a float"
            raise TypeError(msg)
        return avg

    at_quantile = sorted_values[index]
    if not isinstance(at_quantile, float):
        # strictly for type narrowing
        msg = "at_quantile is not a float"
        raise TypeError(msg)
    return at_quantile


def get_stacked_quantiles(
    values: FPArray, weights: FPArray, quantile: float
) -> FPArray:
    """Get a weighted quantile for an array of vectors.

    :param values: array of vectors with shape (..., m)
        will return one m-length vector
    :param weights: array of weights with shape (..., 1)
        where shape[:-1] == values.shape[:-1]
    :param quantile: quantile to calculate, in [0, 1]
    :return: axiswise weighted quantile of an m-length vector
    :raises ValueError: if values and weights do not have the same shape[:-1]

    The "gotcha" here is that the weights must be passed as 1D vectors, not scalars.
    """
    if values.shape[:-1] != weights.shape[:-1]:
        msg = "values and weights must have the same shape up to the last axis"
        raise ValueError(msg)
    flat_vectors: FPArray = values.reshape(-1, values.shape[-1]).T
    flat_weights = weights.flatten()
    by_axis = [get_stacked_quantile(x, flat_weights, quantile) for x in flat_vectors]
    return cast("FPArray", np.array(by_axis))


def get_stacked_median(values: FPArray, weights: FPArray) -> float:
    """Get a weighted median for a value.

    :param values: array of values with shape (n,)
    :param weights: array of weights where weights.shape == values.shape
    :return: weighted median of values
    :raises ValueError: if values and weights do not have the same length
    :raises ValueError: if values array is empty (after removing zero-weight values)
    :raises ValueError: if weights are not all positive
    """
    return get_stacked_quantile(values, weights, 0.5)


def get_stacked_medians(values: FPArray, weights: FPArray) -> FPArray:
    """Get a weighted median for an array of vectors.

    :param values: array of vectors with shape (..., m)
        will return one m-length vector
    :param weights: array of weights with shape (..., 1)
        where shape[:-1] == values.shape[:-1]
    :return: axiswise weighted median of an m-length vector
    :raises ValueError: if values and weights do not have the same shape[:-1]

    The "gotcha" here is that the weights must be passed as 1D vectors, not scalars.
    """
    return get_stacked_quantiles(values, weights, 0.5)
