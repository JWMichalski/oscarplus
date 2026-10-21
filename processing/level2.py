"""
Level 2 OSCAR+ processing module
===============================

This module contains the functions used to process the L2 OSCAR data
and MARS model data

Functions
---------
prepare_dataset:
    Prepare the dataset for use in the OSCAR+ algorithm
prepare_MARS2D:
    Prepare MARS2D dataset the to be consistent with OSCAR naming scheme
remove_unreliable_cells:
    Remove the cells that are unreliable due to inoptimal incidence angle
    or oscillations in mid-beam on the 22.05.2022
rotate_to_geo_coords:
    Rotate the velocity components from model grid coords to geographic coords.
"""

import xarray as xr
from seastar.retrieval.level2 import sol2level2
from seastar.utils import tools as sstools
from oscarplus.processing.level1 import mask_unreliable_cells

####################
# OSCAR processing #
####################


def prepare_dataset(L2_sol):
    """
    Prepare the dataset for use in the OSCAR+ algorithm

    Returns a dataset without 'Antenna' dimension and without 'x' coordinate
    Uses sol2level2 function from seastar module
    to compute the geophysical parameters

    Parameters
    ----------
    L2_sol : ``xarray.DataSet``
        L2 OSCAR data to prepare.
    Returns
    -------
    L2 : ``xarray.DataSet``
        L2 OSCAR data without 'Antenna' dimension, without 'x' coordinate
        and with renamed data variables.
    """
    L2 = xr.Dataset()
    # retrieve geophysical parameters
    L2 = sol2level2(L2_sol)
    L2 = L2.mean(dim="Antenna")  # remove Antenna dimension
    L2 = L2.drop("x")  # remove x coordinate
    return L2


###################
# MARS processing #
###################


def prepare_MARS2D(model):
    """
    Prepare MARS2D dataset the to be consistent with OSCAR naming scheme

    Parameters
    ----------
    model : ``xarray.Dataset``
        MARS2D dataset to prepare.
    Returns
    -------
    ``xarray.DataSet``
        MARS2D dataset with renamed dimensions and data variables
    """
    # add current velocity and direction
    cvel, cdir = sstools.currentUV2VelDir(
        model["U"].values, model["V"].values
    )  # converts u and v components to velocity and direction
    model["CurrentVelocity"] = (("time", "nj", "ni"), cvel)
    model["CurrentDirection"] = (("time", "nj", "ni"), cdir)
    # DS=DS.isel(ni=slice(0, 100)) #selects a region of interest
    model = model.rename({"ni": "GroundRange", "nj": "CrossRange"})
    return model


def remove_unreliable_cells(L1c, L2):
    """
    Remove the cells that are unreliable due to inoptimal incidence angle
    or oscillations in mid-beam on the 22.05.2022.

    Parameters
    ----------
    L1c : ``xarray.DataSet``
        The L1c dataset with 'DateTaken' attribute.
    L2 : ``xarray.DataSet``
        The L2 dataset.

    Returns
    -------
    ``xarray.DataSet``
        The L2 dataset with the unreliable cells removed.
    """
    mask = mask_unreliable_cells(L1c, resolution=L2.attrs["Resolution"])
    L2 = L2.where(~mask)
    return L2


#####################
# MITgcm processing #
#####################


def rotate_to_geo_coords(mitgcm, variable_root, remove_old=True):
    """
    Rotate the velocity components from model grid coords to geographic coords.

    Parameters
    ----------
    mitgcm : ``xarray.Dataset``
        The MITgcm dataset.
        Must contain the variables 'AngleCS' and 'AngleSN' for the rotation.
    variable_root : str
        The root name of the variables to be rotated.
        The function will look for variables with names:
         '{variable_root}X' and '{variable_root}Y'.
    remove_old : bool, optional
        Whether to remove the grid-aligned variables, by default True

    Returns
    -------
    ``xarray.Dataset``
        The dataset with the velocity components in geographic coords.
    """
    variable_x = f"{variable_root}X"
    variable_y = f"{variable_root}Y"
    variable_u = f"{variable_root}U"
    variable_v = f"{variable_root}V"

    if variable_x in mitgcm and variable_y in mitgcm:
        u = mitgcm[variable_x]
        v = mitgcm[variable_y]

        mitgcm[variable_u] = u * mitgcm["AngleCS"] - v * mitgcm["AngleSN"]
        mitgcm[variable_v] = u * mitgcm["AngleSN"] + v * mitgcm["AngleCS"]
    else:
        raise ValueError(
            f"Variables {variable_x} and/or {variable_y} not found in the dataset."
        )

    mitgcm[variable_u].attrs["long_name"] = (
        mitgcm[variable_x]
        .attrs["long_name"]
        .replace(" (NOT zonal; rotate with AngleCS/SN for east)", "")
        .replace("Grid", "Geographic")
    )
    mitgcm[variable_v].attrs["long_name"] = (
        mitgcm[variable_y]
        .attrs["long_name"]
        .replace(" (NOT meridional; rotate with AngleCS/SN for north)", "")
        .replace("Grid", "Geographic")
    )

    for new, original in [[variable_u, variable_x], [variable_v, variable_y]]:
        mitgcm[new].attrs["units"] = mitgcm[original].attrs["units"]

    if remove_old:
        mitgcm = mitgcm.drop_vars([variable_x, variable_y])

    return mitgcm
