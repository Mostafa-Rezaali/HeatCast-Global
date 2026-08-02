"""Regression tests for the exact HeatIndex target calculation."""

import numpy as np
import pytest

from heat_index import (
    heat_index_c,
    heat_index_from_tmax_dewpoint_c,
    relative_humidity_from_dewpoint,
)


def test_noaa_heat_index_reference_value():
    # NOAA Rothfusz example: 90 F and 70% RH gives approximately 105.9 F.
    result_c = heat_index_c(np.array([32.2222222222]), np.array([70.0]))[0]
    assert result_c * 9.0 / 5.0 + 32.0 == pytest.approx(105.922, abs=0.01)


def test_dewpoint_relative_humidity_and_heat_index_are_consistent():
    tmax = np.array([30.0, 35.0])
    dewpoint = np.array([20.0, 24.0])
    rh = relative_humidity_from_dewpoint(tmax, dewpoint)
    assert np.all((rh > 0.0) & (rh < 100.0))
    np.testing.assert_allclose(
        heat_index_from_tmax_dewpoint_c(tmax, dewpoint),
        heat_index_c(tmax, rh),
    )


def test_heat_index_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes must match"):
        heat_index_from_tmax_dewpoint_c(np.zeros((2, 2)), np.zeros((2, 3)))
