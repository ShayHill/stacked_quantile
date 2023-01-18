# stacked_quantile

'Stacked' quantile functions. Close to weighted quantile functions.

These functions are used to calculate quantiles of a set of values, where each value
has a weight. The typical process for calculating a weighted quantile is to create a
CDF from the weights, then interpolate the values to find the quantile.

These functions, however, treat weighted values (given integer weights) exactly as
multiple values.

So, values `(1, 2, 3)` with weights `(4, 5, 6)` will be treated as

```
(1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3)
```

If the quantile falls exactly between
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
    robust. This could also happen with repeaded, non-weighted values. One
    workaround is to divide the values into group_a = values strictly < median,
    group_b = values strictly > median, then add == median to the smaller group.
