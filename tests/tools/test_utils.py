import pytest
from oscarplus.tools import utils
import xarray as xr
import numpy as np


@pytest.fixture
def L2_NaNs():
    """Create a sample L2 OSCAR dataset"""
    values = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [3, 3, 3, np.nan],
            [3, 3, 3, 0],
            [3, 3, 3, np.nan],
            [3, 3, 3, np.nan],
        ]
    )
    return xr.Dataset(
        data_vars={
            "CurrentU": (["CrossRange", "GroundRange"], values),
            "CurrentV": (["CrossRange", "GroundRange"], values),
        },
        coords={"CrossRange": np.arange(5), "GroundRange": np.arange(4)},
    )


def test_no_of_NN(L2_NaNs):
    """Test the no_of_NN function"""
    correct_answer = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [2, 3, 2, np.nan],
            [3, 4, 4, 1],
            [3, 4, 3, np.nan],
            [2, 3, 2, np.nan],
        ]
    )
    test_result = utils.no_of_NN(L2_NaNs.CurrentU)
    assert np.array_equal(test_result.values, correct_answer, equal_nan=True)


def test_cut_NaNs(L2_NaNs):
    correct_answer = np.array(
        [
            [3, 3, 3, np.nan],
            [3, 3, 3, 0],
            [3, 3, 3, np.nan],
            [3, 3, 3, np.nan],
        ]
    )
    test_result = utils.cut_NaNs(L2_NaNs)
    assert np.array_equal(test_result.CurrentU.values, correct_answer, equal_nan=True)
