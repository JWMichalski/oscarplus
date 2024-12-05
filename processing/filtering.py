"""
Median filtering tools OSCAR+
============================
This module contains functions for filtering of OSCAR+ data.

Functions
---------
- angle_median_filter :
    Applies an angle median filter to a DataArray
- directionmagnitude_median_windcurrent :
    Filters the direction and velocity data with a median filter
- component_median_windcurrent :
    Component median for wind and current data
- downscale :
    Downscale a dataset by a factor
"""

import numpy as np
import xarray as xr
import seastar as ss
from oscarplus.tools.calc import median_angle
from oscarplus.tools.utils import no_of_NN, get_resolution, set_resolution


def angle_median_filter(DS, NearestNeighbours, min_NN, window_size=3):
    """
    Applies an angle median filter to a DataArray

    Parameters
    ----------
    DS : ``xarray.dataarray``
        dataarray to apply the filter to
        Must have CrossRange and GroundRange coordinates
    NearestNeighbours : ``xarray.dataarray``
        dataarray containing the number of nearest neighbours for each cell
    NN_min : ``int``
        minimum number of nearest neighbours for the filter to be applied
        If the number of nearest neighbours is less than this value,
        the filter will return np.nan
    window_size : ``int``, optional
        size of the window for the median filter
        Default is 3

    Returns
    -------
    DS_filtered : ``xarray.dataarray``
        dataarray with the median filter applied
    """
    DS_rolling = DS.rolling(
        {"CrossRange": window_size, "GroundRange": window_size},
        center=True,
        min_periods=1,
    ).construct(CrossRange="CrossRangeWindow", GroundRange="GroundRangeWindow")
    DS_filtered = DS.copy(deep=True)

    for iCrossRange in DS_rolling.CrossRange.values:
        for iGroundRange in DS_rolling.GroundRange.values:
            if (
                NearestNeighbours.sel(CrossRange=iCrossRange, GroundRange=iGroundRange)
                >= min_NN
            ):
                calc_median_angle = median_angle(
                    DS_rolling.sel(
                        CrossRange=iCrossRange, GroundRange=iGroundRange
                    ).values
                )
                if not np.isnan(calc_median_angle):
                    DS_filtered.loc[
                        {"CrossRange": iCrossRange, "GroundRange": iGroundRange}
                    ] = calc_median_angle
            else:
                DS_filtered.loc[
                    {"CrossRange": iCrossRange, "GroundRange": iGroundRange}
                ] = np.nan
    return DS_filtered


def directionmagnitude_median_windcurrent(L2, min_no_of_NN=2):
    """
    Filters the direction and velocity data with a median filter

    Parameters
    ----------
    L2 : ``xarray.dataset``
        L2 OSCAR dataset
        Must have CrossRange and GroundRange coordinates and
        contain 'CurrentDirection', 'CurrentVelocity',
        'EarthRelativeWindDirection', 'EarthRelativeWindSpeed' data variables

    Returns
    -------
    L2_median : ``xarray.dataset``
        L2 OSCAR dataset with filtered direction and velocity data
        With previous data variables + 'CurrentU', 'CurrentV',
        'EarthRelativeWindU', 'EarthRelativeWindV'
    """

    L2_median = L2.copy(deep=True)
    # calculate median for current
    L2_median["CurrentDirection"] = angle_median_filter(
        L2_median.CurrentDirection,
        no_of_NN(L2_median["CurrentDirection"]),
        min_no_of_NN,
    )
    L2_median["CurrentVelocity"] = L2_median.CurrentVelocity.rolling(
        {"CrossRange": 3, "GroundRange": 3}, center=True, min_periods=1
    ).median()
    # set CurrentVelocity to NaN if CurrentDirection is NaN
    # this way CurrentVelocity will be NaN if min_of_no_NN is not met
    L2_median["CurrentVelocity"] = xr.where(
        np.isnan(L2_median["CurrentDirection"]),
        np.nan,
        L2_median["CurrentVelocity"],
    )
    # calculate current components
    CurU, CurV = ss.utils.tools.currentVelDir2UV(
        L2_median.CurrentVelocity, L2_median.CurrentDirection
    )
    L2_median["CurrentU"] = CurU
    L2_median["CurrentV"] = CurV
    # calculate median for wind
    L2_median["EarthRelativeWindDirection"] = angle_median_filter(
        L2_median.EarthRelativeWindDirection,
        no_of_NN(L2_median["EarthRelativeWindDirection"]),
        min_no_of_NN,
    )
    L2_median["EarthRelativeWindSpeed"] = L2_median.EarthRelativeWindSpeed.rolling(
        {"CrossRange": 3, "GroundRange": 3}, center=True, min_periods=1
    ).median()
    # set EarthRelativeWindSpeed to NaN if EarthRelativeWindDirection is NaN
    # this way EarthRelativeWindSpeed will be NaN if min_of_no_NN is not met
    L2_median["EarthRelativeWindSpeed"] = xr.where(
        np.isnan(L2_median["EarthRelativeWindDirection"]),
        np.nan,
        L2_median["EarthRelativeWindSpeed"],
    )
    # calculate wind components
    WinU, WinV = ss.utils.tools.windSpeedDir2UV(
        L2_median.EarthRelativeWindSpeed, L2_median.EarthRelativeWindDirection
    )
    L2_median["EarthRelativeWindU"] = WinU
    L2_median["EarthRelativeWindV"] = WinV

    return L2_median


def component_median_windcurrent(L2, window_size=3):
    """
    Component median for wind and current data

    Parameters
    ----------
    L2 : ``xarray.dataset``
        L2 OSCAR dataset
        Must have CrossRange and GroundRange coordinates and
        contain 'CurrentU', 'CurrentV',
        'EarthRelativeWindU', 'EarthRelativeWindV' data variables
    window_size : ``int``, optional
        size of the window for the median filter
        Default is 3

    Returns
    -------
    L2 : ``xarray.dataset``
        L2 OSCAR dataset with filtered direction and velocity data
        With previous data variables + 'CurrentVelocity', 'CurrentDirection',
        'EarthRelativeWindSpeed', 'EarthRelativeWindDirection'
    """

    def filter_component(DA, window_size):
        DA = DA.rolling(
            {"CrossRange": window_size, "GroundRange": window_size},
            center=True,
            min_periods=5,
        ).median()
        return DA

    L2["CurrentU"] = filter_component(L2["CurrentU"], window_size)
    L2["CurrentV"] = filter_component(L2["CurrentV"], window_size)
    L2["EarthRelativeWindU"] = filter_component(L2["EarthRelativeWindU"], window_size)
    L2["EarthRelativeWindV"] = filter_component(L2["EarthRelativeWindV"], window_size)
    # calculate current components
    CVel, CDir = ss.utils.tools.currentUV2VelDir(L2.CurrentU, L2.CurrentV)
    L2["CurrentVelocity"] = xr.DataArray(CVel, dims=["CrossRange", "GroundRange"])
    L2["CurrentDirection"] = xr.DataArray(CDir, dims=["CrossRange", "GroundRange"])
    WinVel, WinDir = ss.utils.tools.windUV2SpeedDir(
        L2.EarthRelativeWindU, L2.EarthRelativeWindV
    )
    L2["EarthRelativeWindSpeed"] = xr.DataArray(
        WinVel, dims=["CrossRange", "GroundRange"]
    )
    L2["EarthRelativeWindDirection"] = xr.DataArray(
        WinDir, dims=["CrossRange", "GroundRange"]
    )
    return L2


def downscale(DS, factor):
    """
    Downscale a dataset by a factor

    Parameters
    ----------
    DS : ``xarray.dataset``
        Dataset to downscale
    factor : ``int``
        Factor to downscale by

    Returns
    -------
    DS_downscaled : ``xarray.dataset``
        Downscaled dataset.
        Resolution attribute is updated if present.
    """
    if not isinstance(factor, int):
        raise TypeError("Factor must be an integer")
    if factor < 2:
        raise ValueError("Factor must be greater than 1")

    DS_downscaled = (
        DS.coarsen(CrossRange=factor, boundary="trim")
        .mean()
        .coarsen(GroundRange=factor, boundary="trim")
        .mean()
        .compute()
    )
    assert (
        DS_downscaled.CrossRange.shape[0] == DS.CrossRange.shape[0] // factor
    ), "CrossRange shape is not correct"
    assert (
        DS_downscaled.GroundRange.shape[0] == DS.GroundRange.shape[0] // factor
    ), "GroundRange shape is not correct"

    if "Resolution" in DS_downscaled.attrs:
        resolution = get_resolution(DS_downscaled)
        set_resolution(DS_downscaled, resolution * factor)

    return DS_downscaled
