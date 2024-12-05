import pytest
from oscarplus.processing import filtering
from oscarplus.tools.utils import no_of_NN
import xarray as xr
import numpy as np


@pytest.fixture
def L2_NaNs():
    """Create a sample L2 OSCAR dataset"""
    Direction = np.array(
        [
            [0, 5, 15, np.nan],
            [355, 345, 270, 35],
            [205, 325, 295, np.nan],
            [200, 320, 290, np.nan],
        ]
    )
    Magnitude = np.array(
        [[2, 2, 2, np.nan], [2, 3, 3, 4], [3, 3, 3, np.nan], [4, 4, 4, np.nan]]
    )
    return xr.Dataset(
        data_vars={
            "CurrentVelocity": (["CrossRange", "GroundRange"], Magnitude),
            "CurrentDirection": (["CrossRange", "GroundRange"], Direction),
            "EarthRelativeWindSpeed": (["CrossRange", "GroundRange"], Magnitude),
            "EarthRelativeWindDirection": (["CrossRange", "GroundRange"], Direction),
        },
        coords={"CrossRange": np.arange(4), "GroundRange": np.arange(4)},
    )


def test_angle_median_filter(L2_NaNs):
    correct_answer = np.array(
        [
            [357.5, 357.5, 5.0, np.nan],
            [350.0, 345.0, 345.0, np.nan],
            [322.5, 295.0, 320, np.nan],
            [262.5, 292.5, 307.5, np.nan],
        ]
    )
    test_result = filtering.angle_median_filter(
        L2_NaNs["CurrentDirection"], no_of_NN(L2_NaNs["CurrentDirection"]), min_NN=2
    )
    assert np.array_equal(test_result.values, correct_answer, equal_nan=True)
