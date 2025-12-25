"""Test stacked_quantile.py

:author: Shay Hill
:created: 2023-01-17
"""

from typing import Iterator

import numpy as np
import numpy.typing as npt
import pytest

import stacked_quantile

_QuantileArgs = tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], float]


def _get_factors(num: int) -> list[int]:
    """Return a list numbers that could be multiplied to get num.

    :param num: the number to factor

       >>> _get_factors(12)
       [3, 4]

       >>> _get_factors(7)
       [7]

       >>> _get_factors(34)
       [17, 2]

       >>> _get_factors(36)
       [3, 3, 4]

    This is a helper function for testing to break a reshape a 1 x num array into
    potentially more dimensions.
    """

    def iter_factor(num_: int) -> Iterator[int]:
        """Iter factors of num_.

        :param num_: the number to factor

        If num is 0, return
        If num is < 3, yeild num and return
        Find a factor >= 3, yeild it, then call iter_factor on the quotient
        """
        if num_ == 1:
            return
        if num_ == 2:
            yield num_
            return
        for factor in range(3, num_ + 1):
            if num_ % factor == 0:
                yield factor
                yield from iter_factor(num_ // factor)
                return

    return list(iter_factor(num)) or [1]


@pytest.fixture(scope="function", params=range(1, 100))
def quantile_args(request: pytest.FixtureRequest) -> _QuantileArgs:
    values: npt.NDArray[np.int_] = np.random.randint(1, 1000, request.param)
    weights = np.random.randint(1, 1000, request.param)
    quantile = float(np.random.random())
    return values, weights, quantile


@pytest.fixture(scope="function", params=range(1, 100))
def quantiles_args(request: pytest.FixtureRequest) -> _QuantileArgs:
    shape = _get_factors(request.param)
    wshape = shape[:-1] + [1]
    cvalues = request.param
    cweights = np.prod(wshape)
    values = np.random.randint(1, 1000, cvalues).reshape(shape)
    weights = np.random.randint(1, 1000, cweights).reshape(wshape)
    quantile = float(np.random.random())
    return values, weights, quantile


class TestStackedQuantile:
    def test_interval_occurrences(self, quantile_args: _QuantileArgs):
        """Matches results as if integer weights were occurrences"""
        xs = quantile_args[0] * 1.0
        ys = quantile_args[1]
        weight_as_qty = float(np.median(np.repeat(xs, ys)))
        weight_as_size = stacked_quantile.get_stacked_quantile(xs, ys * 1.0, 0.5)
        assert np.isclose(weight_as_qty, weight_as_size)

    def test_allow_all_zero_weights(self):
        """Allows all zero weights"""
        xs = np.random.randint(1, 1000, 10) / 1.0
        ys_0 = np.zeros(10)
        ys_1 = np.ones(10)
        all_0 = stacked_quantile.get_stacked_quantile(xs, ys_0, 0.5)
        all_1 = stacked_quantile.get_stacked_quantile(xs, ys_1, 0.5)
        np.testing.assert_array_equal(all_0, all_1)

    def test_all_zero_weights_returns_unweighted_median(self):
        """Returns unweighted median when all weights are zero"""
        xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ys = np.zeros(7)
        result = stacked_quantile.get_stacked_quantile(xs, ys, 0.5)
        expected = float(np.median(xs))
        assert np.isclose(result, expected)

    def test_all_zero_weights_returns_unweighted_quantile(self):
        """Returns unweighted quantile when all weights are zero for any quantile"""
        xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ys = np.zeros(10)
        # When all weights are zero, they're treated as equal weights (ones)
        # This should match the result with equal weights
        for quantile in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result_zero_weights = stacked_quantile.get_stacked_quantile(
                xs, ys, quantile
            )
            result_equal_weights = stacked_quantile.get_stacked_quantile(
                xs, np.ones_like(xs), quantile
            )
            assert np.isclose(result_zero_weights, result_equal_weights)

    def test_floats_match_ints(self, quantile_args: _QuantileArgs):
        """Matches results as if float weights were fractions of occurrences"""
        xs = quantile_args[0] * 1.0
        ys = quantile_args[1] * 1.0
        q = quantile_args[2]
        int_weights = stacked_quantile.get_stacked_quantile(xs, ys, q)
        float_weights = stacked_quantile.get_stacked_quantile(xs, ys / 7, q)
        assert np.isclose(int_weights, float_weights)

    def test_interpolate_when_boundary_hit(self):
        """Interpolates when quantile is on boundary"""
        xs: stacked_quantile._FPArray = np.random.randint(1, 1000, 2) * 1.0
        ys: stacked_quantile._FPArray = np.ones(2)
        assert np.isclose(
            stacked_quantile.get_stacked_quantile(xs, ys, 0.5), (xs[0] + xs[1]) / 2
        )

    def test_one_value(self):
        """Returns value if there is only one value"""
        xs = np.array([1])
        ys = np.array([8])
        assert stacked_quantile.get_stacked_quantile(xs, ys, 0.5) == 1

    def test_strips_zero_weight_values(self):
        """Strips zero-weight values from calculation"""
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([0, 0, 0, 0, 1])
        assert stacked_quantile.get_stacked_quantile(xs, ys, 0.5) == 5

    def test_value_error_on_lengths_do_not_match(self):
        """Raises ValueError if lengths of x and y do not match"""
        xs = np.random.randint(1, 1000, 10) / 1.0
        ys = np.random.randint(1, 1000, 11) / 1.0
        with pytest.raises(ValueError) as excinfo:
            _ = stacked_quantile.get_stacked_quantile(xs, ys, 0.5)
        assert "values and weights must be the same length" in str(excinfo.value)

    def test_value_error_on_quantile_out_of_range(self):
        """Raises ValueError if quantile is not in [0, 1]"""
        xs = np.random.randint(1, 1000, 10) / 1.0
        ys = np.random.randint(1, 1000, 10) / 1.0
        with pytest.raises(ValueError) as excinfo:
            _ = stacked_quantile.get_stacked_quantile(xs, ys, 1.5)
        assert "quantile must be in interval [0, 1]" in str(excinfo.value)

    # def test_value_error_on_no_values(self):
    #     """Raises ValueError if there are no values"""
    #     xs = np.array([])
    #     ys = np.array([])
    #     with pytest.raises(ValueError) as excinfo:
    #         _ = stacked_quantile.get_stacked_quantile(xs, ys, 0.5)
    #     assert "values empty" in str(excinfo.value)

    def test_value_error_if_any_weights_are_below_zero(self):
        """Raises ValueError if any weights are below zero"""
        xs = np.random.randint(1, 1000, 10) / 1.0
        ys = np.random.randint(1, 1000, 10) / 1.0
        ys[0] = -1
        with pytest.raises(ValueError) as excinfo:
            _ = stacked_quantile.get_stacked_quantile(xs, ys, 0.5)
        assert "weights must be non-negative" in str(excinfo.value)


class TestStackedQuantiles:
    def test_any_dimensions(self, quantiles_args: _QuantileArgs):
        """Treat last dimension as vectors."""
        values = quantiles_args[0] * 1.0
        weights = quantiles_args[1] * 1.0
        quantile = quantiles_args[2]
        with_nd_array = stacked_quantile.get_stacked_quantiles(
            values, weights, quantile
        )
        values = values.reshape(-1, values.shape[-1])
        weights = weights.reshape(-1, weights.shape[-1])
        with_2d_array = stacked_quantile.get_stacked_quantiles(
            values, weights, quantile
        )
        assert np.allclose(with_nd_array, with_2d_array)  # type: ignore
