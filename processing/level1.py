"""
Level 1 OSCAR+ processing module
===============================

This module contains the functions used to process the L1 OSCAR data

Functions
---------
calculate_sigma0:
    Calculate the sigma0 from the intensity
compute_beam_mask:
    Create a mask for the cells with less than 3 looks
compute_beam_land_mask:
    Create mask for coastal cells and cells with less than 3 looks
build_geo_dataset:
    Builds a dataset from the given wind parameters
mask_unreliable_cells:
    Create a mask for the cells that are unreliable due to inoptimal incidence angle
    or due to oscillations in mid-beam on the 22.05.2022.
"""

import seastar as ss
import xarray as xr
import numpy as np
import warnings
from scipy.ndimage import binary_dilation


def calculate_sigma0(L1b):
    """
    This function calculates the sigma0 from intensity

    Sigma0 is calculated as the difference between
    the intensity and the mean intensity in the CrossRange direction,
    but with the GroundRange mean of the CrossRange mean intensity added to normalize.

    Parameters
    ----------
    L1b : ``xarray.DataSet``
        L1b dataset.
        Must contain 'CrossRange' and 'GroundRange' dimensions
        and 'Intensity' variable.
    Returns
    -------
    ``xarray.DataSet``
        L1b dataset with the 'Sigma0_db' variable added.
    """
    Intensity_db = ss.utils.tools.lin2db(L1b["Intensity"])
    L1b["Sigma0_db"] = (
        Intensity_db  # intensity
        - Intensity_db.mean(
            dim="CrossRange"
        )  # remove mean intensity in CrossRange direction
        + Intensity_db.mean(dim="CrossRange").mean(
            dim="GroundRange"
        )  # add mean of mean intensities
    )
    return L1b


def compute_beam_mask(DA):
    """
    Create a mask for the cells with less than 3 looks

    Parameters
    ----------
    DA : ``xarray.DataArray``
        L1c OSCAR data array to compute the mask on.
    Returns
    -------
    ``xarray.DataArray``
        Mask for the cells with less than 3 looks.
        True where the data is valid and False where it is not.
    """
    # create a mask for each antenna with 1 where there is data and 0 where there is not
    a = [xr.where(np.isnan(DA.sel(Antenna=a)), 0, 1) for a in DA.Antenna.data]
    # merge the antenna masks together
    mask = xr.where((a[0] + a[1] + a[2]) == 3, True, False)
    return mask


def compute_beam_land_mask(L1c, dilation=2):
    """
    Create mask for coastal cells and cells with less than 3 looks

    Uses the GSHHS coast data to create a mask for the land and
    compute_beam_mask for the cells with less than 3 looks.
    Merges the two masks together.

    Parameters
    ----------
    L1c : ``xarray.Dataset``
        L1c OSCAR data to compute the mask on.
    dilation : ``int``, optional
        Number of iterations for binary dilation.
        Default is 2.
    Returns
    -------
    ``xarray.DataArray``
        Mask for the land and points with less than 3 looks.
        True where the data is valid and False where it is not.
    """
    DA = L1c["Sigma0"]
    land_mask = ss.utils.tools.compute_land_mask_from_GSHHS(
        DA.sel(Antenna="Mid"), quiet=True
    )  # compute land masks
    land_mask = xr.DataArray(
        binary_dilation(land_mask, iterations=dilation),
        coords=DA.sel(Antenna="Mid").coords,
        dims=DA.sel(Antenna="Mid").dims,
    )
    land_mask = xr.where(land_mask == 1, False, True)
    # merge the antenna masks together
    beam_mask = compute_beam_mask(DA)
    # merge the land mask and the beam mask
    mask = np.logical_and(beam_mask, land_mask)
    return mask


def build_geo_dataset(L1c, windspeed, winddirection):
    """
    Builds a dataset from the given wind parameters

    Parameters
    ----------
    L1c : ``xarray.Dataset``
        Used for the coordinates and dimensions of the data.
    windspeed: ``float``
        Wind speed in m/s.
    winddirection: ``float``
        Wind direction in degrees.
    Returns
    -------
    ``xarray.DataSet``
        Dataset with the geophysical parameters.
    """
    geo = xr.Dataset()
    geo["EarthRelativeWindSpeed"] = xr.DataArray(
        data=np.full(L1c.Sigma0.shape, windspeed),
        coords=L1c.Sigma0.coords,
        dims=L1c.Sigma0.dims,
    )
    geo["EarthRelativeWindDirection"] = xr.DataArray(
        data=np.full(L1c.Sigma0.shape, winddirection),
        coords=L1c.Sigma0.coords,
        dims=L1c.Sigma0.dims,
    )
    geo["CurrentVelocity"] = xr.DataArray(
        data=np.full(L1c.RSV.shape, 0), coords=L1c.RSV.coords, dims=L1c.RSV.dims
    )
    geo["CurrentDirection"] = xr.DataArray(
        data=np.full(L1c.RSV.shape, 0), coords=L1c.RSV.coords, dims=L1c.RSV.dims
    )
    return geo


def mask_unreliable_cells(L1c, resolution):
    """
    Create a mask for the cells that are unreliable due to inoptimal incidence angle
    or due to oscillations in mid-beam on the 22.05.2022.
    This mask always covers the far range cells
    and the near range cells for the 22.05.2022.

    Parameters
    ----------
    L1c : ``xarray.DataSet``
        The L1c dataset with the 'DateTaken' attribute.
    Returns
    -------
    ``xarray.DataArray``
        The mask covering the untrustworthy cells.
        True for unreliable, False for reliable.
    """
    date = L1c.attrs["DateTaken"]
    if resolution == "200x200m":
        resolution_multiplier = 1
    elif resolution == "100x100m":
        resolution_multiplier = 2
    else:
        resolution_multiplier = 1
        warnings.warn("Resolution not recognized, using a resolution multiplier of 1")

    far_range_mask = xr.DataArray(
        False,
        coords=L1c.Sigma0.sel(Antenna="Mid").coords,
        dims=L1c.Sigma0.sel(Antenna="Mid").dims,
    )
    if date == "20220517":
        far_range_mask[:, -4 * resolution_multiplier :] = True
    else:
        far_range_mask[:, -5 * resolution_multiplier :] = True

    if date == "20220522":
        mid_beam_oscillation_mask = xr.DataArray(
            False,
            coords=L1c.Sigma0.sel(Antenna="Mid").coords,
            dims=L1c.Sigma0.sel(Antenna="Mid").dims,
        )
        mid_beam_oscillation_mask[:, 0 : 6 * resolution_multiplier] = True
        mask = np.logical_or(far_range_mask, mid_beam_oscillation_mask)
    else:
        mask = far_range_mask
    mask = mask.drop_vars("Antenna")
    return mask
