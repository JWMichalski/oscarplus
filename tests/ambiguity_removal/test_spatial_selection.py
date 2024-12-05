import pytest
import oscarplus.ambiguity_removal.spatial_selection as spatial_selection
import numpy as np
import xarray as xr


@pytest.fixture
def initial():
    """Create a sample L2 OSCAR dataset"""
    values = np.zeros((5, 4))
    return xr.Dataset(
        data_vars={
            "CurrentU": (["CrossRange", "GroundRange"], values),
            "CurrentV": (["CrossRange", "GroundRange"], values),
            "EarthRelativeWindU": (["CrossRange", "GroundRange"], values),
            "EarthRelativeWindV": (["CrossRange", "GroundRange"], values),
        },
        coords={"CrossRange": np.arange(5), "GroundRange": np.arange(4)},
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


def cost(L2_sel, L2_neighbours, **cost_function_kwargs):
    """Fake cost function for testing"""
    return np.array([3, 2, 0, 1])


@pytest.mark.parametrize("i_x, i_y", [(0, 0), (3, 2), (3, 3), (4, 2), (4, 3)])
def test_single_cell_ambiguity_selection(lmout, initial, i_x, i_y):
    """Test the selection of ambiguity with the lowest cost"""
    selected_ambiguity = spatial_selection.single_cell_ambiguity_selection(
        lmout, initial, i_x, i_y, cost, window=3
    )
    assert selected_ambiguity == 2


def test_solve_ambiguity_spatial_selection(lmout, initial):
    """Test the solve ambiguity function"""
    L2_solved = spatial_selection.solve_ambiguity_spatial_selection(
        lmout, initial, cost, pass_number=1, window=3
    )
    correct = np.full((5, 4), 2)
    assert (L2_solved.CurrentU.values == correct).all()
