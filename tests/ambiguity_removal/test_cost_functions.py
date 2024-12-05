import pytest
import oscarplus.ambiguity_removal.cost_functions as cost_functions
import numpy as np
import numpy.testing as npt
import xarray as xr


@pytest.fixture
def L2_small2D():
    """Create a sample L2 OSCAR dataset"""
    values = np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
    return xr.Dataset(
        data_vars={
            "CurrentU": (["CrossRange", "GroundRange"], values),
            "CurrentV": (["CrossRange", "GroundRange"], values),
            "EarthRelativeWindU": (["CrossRange", "GroundRange"], values),
            "EarthRelativeWindV": (["CrossRange", "GroundRange"], values),
        },
        coords={"CrossRange": np.arange(3), "GroundRange": np.arange(3)},
    )


@pytest.fixture
def lmout():
    """Create a sample L2 OSCAR dataset"""
    values = np.zeros((4, 5, 4))
    values[1, :, :] = np.full((5, 4), 1)
    values[2, :, :] = np.full((5, 4), 2)
    values[3, :, :] = np.full((5, 4), 3)
    return xr.Dataset(
        data_vars={
            "CurrentU": (["Ambiguities", "CrossRange", "GroundRange"], values),
            "CurrentV": (["Ambiguities", "CrossRange", "GroundRange"], values),
            "EarthRelativeWindU": (
                ["Ambiguities", "CrossRange", "GroundRange"],
                values,
            ),
            "EarthRelativeWindV": (
                ["Ambiguities", "CrossRange", "GroundRange"],
                values,
            ),
        },
        coords={
            "Ambiguities": np.arange(4),
            "CrossRange": np.arange(5),
            "GroundRange": np.arange(4),
        },
    )


def test_calculate_Euclidian_distance_to_neighbours(L2_small2D, lmout):
    """Test the Euclidian distance cost function without centre cell"""
    total_cost = cost_functions.calculate_Euclidian_distance_to_neighbours(
        lmout.isel(GroundRange=1, CrossRange=1),
        L2_small2D,
        windcurrentratio=1,
        include_centre=False,
        Euclidian_method="standard",
    )
    npt.assert_array_almost_equal(
        total_cost, [67.88225100, 45.25483400, 22.62741700, 0.0]
    )


def test_calculate_Euclidian_distance_to_neigbours_and_centre(L2_small2D, lmout):
    """Test the Euclidian distance cost function with centre cell included"""
    total_cost = cost_functions.calculate_Euclidian_distance_to_neighbours(
        lmout.isel(GroundRange=1, CrossRange=1),
        L2_small2D,
        windcurrentratio=1,
        include_centre=True,
        Euclidian_method="standard",
    )
    npt.assert_array_almost_equal(
        total_cost, [76.36753237, 50.91168825, 25.45584412, 0.0]
    )


def test_calculate_squared_Euclidian_distance_to_neighbours(L2_small2D, lmout):
    """Test the squared Euclidian distance cost function without centre cell"""
    total_cost = cost_functions.calculate_Euclidian_distance_to_neighbours(
        lmout.isel(GroundRange=1, CrossRange=1),
        L2_small2D,
        windcurrentratio=1,
        include_centre=False,
        Euclidian_method="squared",
    )
    assert (total_cost == [288.0, 128.0, 32.0, 0.0]).all()


def test_calculate_squared_Euclidian_distance_to_neighbours_and_centre(
    L2_small2D, lmout
):
    """Test the squared Euclidian distance cost function with centre cell included"""
    total_cost = cost_functions.calculate_Euclidian_distance_to_neighbours(
        lmout.isel(GroundRange=1, CrossRange=1),
        L2_small2D,
        windcurrentratio=1,
        include_centre=True,
        Euclidian_method="squared",
    )
    assert (total_cost == [324.0, 144.0, 36.0, 0.0]).all()
