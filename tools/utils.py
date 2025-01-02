"""
OSCAR+ tools utils module
=========================
This module contains utility functions for the OSCAR+ algorithm

Functions
---------
- set_resolution :
    set the resolution of the dataset
- get_resolution :
    get the resolution of the dataset
- cut_NaNs :
    cut the NaNs from the dataset
- no_of_NN :
    compute the number of non-NaN nearest neighbours for each cell in the DS dataset
- get_data_dirs :
    find the top directory of the data
- find_track_angle :
    finds the bearing of the airplane
- attributes_from_filepath :
    extract the date, track and gmf from the filepath
- find_track_corners :
    find latitude and longitude of corners with non-NaN values
- find_six_track_corners :
    find latitude and longitude of corners with non-NaN values
- transect :
    takes a transect of the bathymetry and the current
- align_with_track :
    rotate the components of the dataset to align with the track
- find_closest_lon_lat :
    find the closest longitude and latitude in the dataset to the given lon and lat
- fit_to_extent :
    fit the dataset to the given extent
"""

import os
import warnings
import xarray as xr
import numpy as np
import pandas as pd
from statistics import fmean
from pyproj import Geod
from oscarplus.tools import calc


def set_resolution(DS, resolution):
    """
    Set the resolution of the dataset

    Parameters
    ----------
    DS : ``xarray.DataSet``
        Dataset to set the resolution of.
    resolution : ``int``
        Resolution to add to the dataset to in meters.
    """
    DS.attrs["Resolution"] = f"{resolution}x{resolution}m"


def get_resolution(DS):
    """
    Get the resolution of the dataset

    Parameters
    ----------
    DS : ``xarray.DataSet``
        Dataset to get the resolution of.
        Must have the 'Resolution' attribute in the form '<int>x<int>m'.
    Returns
    -------
    ``int``
        Resolution of the dataset in meters.
    """
    if "Resolution" not in DS.attrs:
        raise ValueError("Resolution attribute not found in the dataset")
    if (
        not isinstance(DS.attrs["Resolution"], str)
        or "x" not in DS.attrs["Resolution"]
        or "m" not in DS.attrs["Resolution"]
    ):
        raise ValueError(
            "Resolution attribute must be a string in the form '<int>x<int>m'"
        )
    resolution_WIP = DS.attrs["Resolution"][:-1].split("x")
    if (
        len(resolution_WIP) != 2
        or not resolution_WIP[0].isdigit()
        or not resolution_WIP[1].isdigit()
    ):
        raise ValueError("Resolution must be in the form '<int>x<int>m'")
    resolution = int(resolution_WIP[0])
    if int(resolution_WIP[1]) != resolution:
        raise ValueError("Resolution must be the same in both dimensions")
    return resolution


def cut_NaNs(DS, data_variable="CurrentU"):
    """
    Cut the NaNs from the dataset

    Cuts rectangles with only NaNs out of the dataset
    by decreasing GroundRange and CrossRange dimensions.

    Parameters
    ----------
    DS : ``xarray.DataSet``
        OSCAR data to cut the NaNs from.
    data_variable : ``string``, optional
        Name of the data variable to base the NaNs removal on.
        Default is 'CurrentU'.
    Returns
    -------
    ``xarray.DataSet``
        OSCAR data with NaNs removed.
    """
    DS = DS.where(DS[data_variable].notnull(), drop=True)
    return DS


def no_of_NN(DA):
    """
    This function computes the number of non-NaN nearest neighbours
    for each cell in the DS dataset

    Works in place, adds a new data variable 'NoOfNearestNeighbours' to the DS dataset.

    Parameters
    ----------
    DA : ``xarray.DataArray``
        Contains the data variable to be processed.
        Must have 'CrossRange' and 'GroundRange' dimensions.
    Returns
    -------
    ``xarray.DataArray``
        Contains the number of nearest neighbours for each cell.
    """
    assert isinstance(DA, xr.DataArray), "DA must be a DataArray"
    cross_range_size = DA.CrossRange.sizes["CrossRange"]
    ground_range_size = DA.GroundRange.sizes["GroundRange"]
    NN = xr.DataArray(
        np.full([cross_range_size, ground_range_size], np.nan),
        dims=["CrossRange", "GroundRange"],
        coords={
            "CrossRange": DA.CrossRange.values,
            "GroundRange": DA.GroundRange.values,
        },
    )
    if "Ambiguities" in DA.dims:  # checks whether the dataset has ambiguities dimension
        # create a mask for NaN values
        DS_isNaN = xr.where(np.isnan(DA.isel(Ambiguities=0)), True, False)
    else:
        # create a mask for NaN values
        DS_isNaN = xr.where(np.isnan(DA), True, False)

    for i in range(0, cross_range_size):  # iterate along track
        for j in range(0, ground_range_size):  # iterate across track
            if not DS_isNaN[i, j]:  # skip if the cell is nan
                NN[i, j] = 0  # initialize the number of neighbours
                if i - 1 >= 0 and not DS_isNaN[i - 1, j]:
                    NN[i, j] += 1
                if i + 1 < cross_range_size and not DS_isNaN[i + 1, j]:
                    NN[i, j] += 1
                if not DS_isNaN[i, j - 1] and j - 1 >= 0:
                    NN[i, j] += 1
                if j + 1 < ground_range_size and not DS_isNaN[i, j + 1]:
                    NN[i, j] += 1
            else:
                NN[i, j] = np.nan
    return NN


def find_track_angle(DS):
    """
    Finds the bearing of the airplane that collected the data

    Parameters
    ----------
    DS : ``xarray.DataSet``
        Contains the OSCAR data
        with the variables 'latitude' and 'longitude'
        and the dimensions 'GroundRange' and 'CrossRange'.
    Returns
    -------
    ``float``
        Bearing of the airplane in degrees.
    """
    lon_beg = DS["longitude"].isel(GroundRange=0, CrossRange=0).values
    lat_beg = DS["latitude"].isel(GroundRange=0, CrossRange=0).values
    lon_end = DS["longitude"].isel(GroundRange=0, CrossRange=-1).values
    lat_end = DS["latitude"].isel(GroundRange=0, CrossRange=-1).values
    geodesic = Geod(ellps="WGS84")
    azimuth, _, _ = geodesic.inv(lon_beg, lat_beg, lon_end, lat_end)
    return azimuth


def attributes_from_filepath(filepath):
    """
    Extract the date, track and gmf from the filepath

    Parameters
    ----------
    filepath : ``string``
        The filepath to extract the attributes from.
        The name of the file must be in form:
        YYYYMMDD_Track_XX_RESOLUTION_GMF_<...>.nc
        Track number can be any length.
        <...> can be any string.
    Returns
    -------
    date : ``string``
        The date of the track.
    track : ``string``
        The track number.
    gmf : ``string``
        The geophysical model function.
    """
    filename = os.path.basename(filepath[:-3])
    attributes = filename.split("_")
    date = attributes[0]
    track = attributes[2]
    gmf = attributes[4]
    return date, track, gmf


def __find_first_last_notnan_latitude_longitude(row):
    """
    Find lat lon of the first and last non-NaN values in a row.

    Parameters
    ----------
    row : ``xarray.DataArray``
        The row to find the first and last non-NaN values in.
    Returns
    -------
    first_notnan : ``dict``
        The latitude and longitude of the first non-NaN value.
    """
    def find_first_notnan_index(row, reverse=False):
        """ "
        Find the index of the first non-NaN value in a row.
        Reverse to find the last non-NaN value.
        """
        first_notnan_index = np.nan
        row_length = len(row)
        if reverse:
            row = reversed(row)
        for i, value in enumerate(row):
            if not np.isnan(value):
                first_notnan_index = i
                if reverse:
                    first_notnan_index = row_length - i - 1
                break
        return first_notnan_index

    first_notnan_index = find_first_notnan_index(row)
    last_notnan_index = find_first_notnan_index(row, reverse=True)
    # Get the longitudes and latitudes of the first and last non-NaN values
    first_notnan = {
        "latitude": row.latitude.values[first_notnan_index],
        "longitude": row.longitude.values[first_notnan_index],
    }
    last_notnan = {
        "latitude": row.latitude.values[last_notnan_index],
        "longitude": row.longitude.values[last_notnan_index],
    }
    return first_notnan, last_notnan


def find_track_corners(DA):
    """
    Find latitude and longitude of corners with non-NaN values

    The corners are the first and last non-NaN values in the first and last row.

    Parameters
    ----------
    DA : ``xarray.DataArray``
        The data array to find the corners of,

    Returns
    -------
    first_notnan_first_row : ``dict``
        The latitude and longitude of the first non-NaN value in the first row.
    last_notnan_first_row : ``dict``
        The latitude and longitude of the last non-NaN value in the first row.
    first_notnan_last_row : ``dict``
        The latitude and longitude of the first non-NaN value in the last row.
    last_notnan_last_row : ``dict``
        The latitude and longitude of the last non-NaN value in the last row.
    """
    first_notnan_first_row, last_notnan_first_row = (
        __find_first_last_notnan_latitude_longitude(DA[:, 0])
    )
    first_notnan_last_row, last_notnan_last_row = (
        __find_first_last_notnan_latitude_longitude(DA[:, -1])
    )
    return (
        first_notnan_first_row,
        last_notnan_first_row,
        first_notnan_last_row,
        last_notnan_last_row,
    )


def find_six_track_corners(DA):
    """
    Find latitude and longitude of corners with non-NaN values

    The corners are the first and last non-NaN values in the first, second and last row
    Used for the track with corners cut off in median filtering.

    Parameters
    ----------
    DA : ``xarray.DataArray``
        The data array to find the corners of.

    Returns
    -------
    first_notnan_first_row : ``dict``
        The latitude and longitude of the first non-NaN value in the first row.
    last_notnan_first_row : ``dict``
        The latitude and longitude of the last non-NaN value in the first row.
    last_notnan_second_row : ``dict``
        The latitude and longitude of the last non-NaN value in the second row.
    last_notnan_last_row : ``dict``
        The latitude and longitude of the last non-NaN value in the last row.
    first_notnan_last_row : ``dict``
        The latitude and longitude of the first non-NaN value in the last row.
    first_notnan_second_row : ``dict``
        The latitude and longitude of the first non-NaN value in the second row.
    """
    first_notnan_first_row, last_notnan_first_row = (
        __find_first_last_notnan_latitude_longitude(DA[:, 0])
    )
    first_notnan_last_row, last_notnan_last_row = (
        __find_first_last_notnan_latitude_longitude(DA[:, -1])
    )
    first_notnan_second_row, last_notnan_second_row = (
        __find_first_last_notnan_latitude_longitude(DA[:, 1])
    )
    return (
        first_notnan_first_row,
        last_notnan_first_row,
        last_notnan_second_row,
        last_notnan_last_row,
        first_notnan_last_row,
        first_notnan_second_row,
    )


def transect(DS, bathymetry, iGround, jCross, angle, max_length=1000, handiness=None):
    """
    Takes a transect of the bathymetry and the current

    Parameters
    ----------
    DS : ``xarray.DataSet``
        DS with
        'GroundRange' and 'CrossRange' coordinates;
        'longitude' and 'latitude' coordinates;
        'CurrentVelocity', 'CurrentDirection', 'CurrentDivergence' data variables;
        'Resolution' attribute.
        If 'CurrentW' is present it will be added to the transect.
    bathymetry : ``xarray.DataSet``
        Bathymetry data with 'elevation' data variable.
    iGround : ``int``
        GroundRange index of the starting cell.
    jCross : ``int``
        CrossRange index of the starting cell.
    angle : ``float``
        Angle of the transect.
        Must be a multiple of 45 degrees in range [0, 315].
    max_length : ``int``, optional
        Maximum length of the transect in number of cells.
        May be less if the transect reaches a NaN value or dataset edge.
        Default is 1000.
    handiness : ``str``, optional
        Handiness of the data.
        Must be either 'right', 'left' or None:
        - 'right': GroundRange 90 degrees to the right of Cross Range.
        - 'left': GroundRange 90 degrees to the left of Cross Range.
        - None: the script will try to determine the handiness.
        Default is None.
    Returns
    -------
    transectOSCAR : ``pandas.DataFrame``
        DataFrame with the transect of OSCAR data and elevation of bathymetry.
    transectElevation : ``pandas.DataFrame``
        DataFrame with the transect of elevation data.
        Provided at 2 points per OSCAR data point
    """

    def make_OSCAR_dict(DS_row, distance):
        """
        Create a dictionary with the OSCAR data

        Parameters
        ----------
        DS_row : ``xarray.DataSet``
            row of the L2 dataset.
        distance : ``float``
            distance from the starting point.
        Returns
        -------
        ``dict``
            dictionary with the OSCAR data.
        """
        point_in_transect = {
            "distance": distance,
            "longitude": DS_row["longitude"].values,
            "latitude": DS_row["latitude"].values,
            "CurrentU": DS_row["CurrentU"].values,
            "CurrentV": DS_row["CurrentV"].values,
            "CurrentVelocity": DS_row["CurrentVelocity"].values,
            "CurrentDirection": DS_row["CurrentDirection"].values,
            "CurrentDivergence": DS_row["CurrentDivergence"].values,
            "CurrentVelocity_along_transect": DS_row["CurrentU"].values
            * transect_abs_dir_vec[0]
            + DS_row["CurrentV"].values * transect_abs_dir_vec[1],
            "CurrentVelocity_across_transect": DS_row["CurrentU"].values
            * transect_abs_dir_vec[1]
            - DS_row["CurrentV"].values * transect_abs_dir_vec[0],
            "elevation": -1.0,
        }
        if WIncluded:
            point_in_transect["CurrentW"] = DS_row["CurrentW"].values
        return point_in_transect

    def get_elevation(longitude, latitude):
        """
        Helper function to get the elevation of a point

        Parameters
        ----------
        longitude : ``float``
            longitude of the point.
        latitude : ``float``
            latitude of the point.
        Returns
        -------
        float
            elevation of the point.
        """
        return -bathymetry.interp(
            latitude=latitude, longitude=longitude, method="linear"
        ).elevation.values

    resolution = get_resolution(DS)

    track_angle = find_track_angle(DS)
    absolute_angle = (angle + track_angle) % 360

    # Set the step size based on the angle
    steps = {
        0: (0, 1),
        45: (-1, 1),
        90: (-1, 0),
        135: (-1, -1),
        180: (0, -1),
        225: (1, -1),
        270: (1, 0),
        315: (1, 1),
    }
    if handiness is None:
        if "Track" and "Level" in DS.attrs:
            handiness = "right"
        elif "product_name" in DS.attrs and "MARC" in DS.attrs["product_name"]:
            handiness = "left"
        else:
            raise ValueError(
                "Cannot determine handiness of the data."
                "Provide handiness as an argument"
            )
    # unit vector in the direction of the transect w.r.t. the local Earth coords
    if handiness == "right":
        transect_abs_dir_vec = np.array(
            [np.sin(np.deg2rad(absolute_angle)), np.cos(np.deg2rad(absolute_angle))]
        )
    elif handiness == "left":
        transect_abs_dir_vec = np.array(
            [
                -np.sin(np.deg2rad(absolute_angle)),
                np.cos(np.deg2rad(absolute_angle)),
            ]
        )
    else:
        raise ValueError("Handiness must be either 'right', 'left' or None")

    if angle in steps:
        istep, jstep = steps[angle]
    else:
        raise ValueError("Angle must be a multiple of 45 degrees")

    spacing = resolution if angle % 90 == 0 else resolution * np.sqrt(2)

    if "CurrentW" in DS:
        WIncluded = True
    else:
        WIncluded = False

    # Get the starting point
    DS_row = DS.isel(GroundRange=iGround, CrossRange=jCross)
    OSCAR = make_OSCAR_dict(DS_row, 0)
    transectOSCAR = pd.DataFrame(OSCAR, index=[0])

    # Get the rest of the transect
    for i in range(max_length):
        iGround += istep
        jCross += jstep
        if iGround < 0 or jCross < 0:
            break

        try:
            DS_row = DS.isel(GroundRange=iGround, CrossRange=jCross)
            if np.isnan(DS_row["CurrentVelocity"]):
                break
            OSCAR = make_OSCAR_dict(DS_row, i * spacing)
            transectOSCAR.loc[len(transectOSCAR)] = OSCAR
        except Exception:
            break

    # first elevation for starting point
    longitude = transectOSCAR.at[0, "longitude"]
    latitude = transectOSCAR.at[0, "latitude"]
    distance = 0
    elevation = get_elevation(longitude, latitude)
    transectOSCAR.at[0, "elevation"] = elevation
    elevation_row = {
        "distance": distance,
        "longitude": longitude,
        "latitude": latitude,
        "elevation": elevation,
    }
    transectElevation = pd.DataFrame(elevation_row, index=[0])
    # Get the elevation of the transect
    for index, row in transectOSCAR.iloc[1:].iterrows():
        last_longitude = longitude
        last_latitude = latitude
        last_distance = distance
        longitude = row["longitude"]
        latitude = row["latitude"]
        distance = row["distance"]
        mid_longitude = fmean([longitude, last_longitude])
        mid_latitude = fmean([latitude, last_latitude])
        mid_distance = fmean([distance, last_distance])
        elevation = get_elevation(mid_longitude, mid_latitude)
        elevation_row = {
            "distance": mid_distance,
            "longitude": mid_longitude,
            "latitude": mid_latitude,
            "elevation": elevation,
        }
        transectElevation.loc[len(transectElevation)] = elevation_row

        elevation = get_elevation(longitude, latitude)
        elevation_row = {
            "distance": distance,
            "longitude": longitude,
            "latitude": latitude,
            "elevation": elevation,
        }
        transectElevation.loc[len(transectElevation)] = elevation_row
        transectOSCAR.at[index, "elevation"] = elevation

    # Convert to numeric
    transectOSCAR = transectOSCAR.apply(pd.to_numeric, errors="coerce")
    transectElevation = transectElevation.apply(pd.to_numeric, errors="coerce")

    # Calculate the gradient of CurrentVelocity_along_transect
    transectOSCAR["CurrentVelocity_along_transect_gradient"] = (
        np.gradient(
            transectOSCAR["CurrentVelocity_along_transect"], transectOSCAR["distance"]
        )
        / resolution
    )
    # asserts for outputs
    required_columns = [
        "distance",
        "longitude",
        "latitude",
        "CurrentU",
        "CurrentV",
        "CurrentVelocity",
        "CurrentDirection",
        "CurrentDivergence",
        "CurrentVelocity_along_transect",
        "CurrentVelocity_across_transect",
        "elevation",
    ]
    assert all(
        column in transectOSCAR.columns for column in required_columns
    ), "transectOSCAR is missing required columns"
    required_columns = ["distance", "longitude", "latitude", "elevation"]
    assert all(
        column in transectElevation.columns for column in required_columns
    ), "transectElevation is missing required columns"

    return transectOSCAR, transectElevation


def align_with_track(DS):
    """
    Rotate the components of the dataset to align with the track

    Parameters
    ----------
    DS : ``xarray.DataSet``
        Dataset for which to rotate the components.
        Must contain the following coordinates: latitude, longitude
        and following variables: CurrentU, CurrentV
        or/and EarthRelativeWindU, EarthRelativeWindV.
    """
    # Rotate the coords to align with the track
    if "CurrentU" and "CurrentV" in DS:
        CurrentU_rot, CurrentV_rot = calc.rotate_vectors(
            DS["CurrentU"].values,
            DS["CurrentV"].values,
            -find_track_angle(DS),
        )
        DS["CurrentU_rot"] = xr.DataArray(CurrentU_rot, coords=DS["CurrentU"].coords)
        DS["CurrentV_rot"] = xr.DataArray(CurrentV_rot, coords=DS["CurrentV"].coords)
    else:
        warnings.warn("CurrentU and CurrentV not found in L2_AR_MF")
    if "EarthRelativeWindU" and "EarthRelativeWindV" in DS:
        EarthRelativeWindU_rot, EarthRelativeWindV_rot = calc.rotate_vectors(
            DS["EarthRelativeWindU"].values,
            DS["EarthRelativeWindV"].values,
            -find_track_angle(DS),
        )
        DS["EarthRelativeWindU_rot"] = xr.DataArray(
            EarthRelativeWindU_rot, coords=DS["EarthRelativeWindU"].coords
        )
        DS["EarthRelativeWindV_rot"] = xr.DataArray(
            EarthRelativeWindV_rot, coords=DS["EarthRelativeWindV"].coords
        )
    else:
        warnings.warn("EarthRelativeWindU and EarthRelativeWindV not found in L2_AR_MF")


def find_closest_lon_lat(given_longitude, given_latitude, DS):
    """
    Find the closest longitude and latitude in the MARS dataset to
    the given longitude and latitude.
    WARNING: Measures the distance quadratically, not suitable for large distances.

    Parameters
    ----------
    given_latitude : ``float``
        The given latitude.
    given_longitude : ``float``
        The given longitude.
    DS : ``xarray.DataSet``
        The MARS dataset.
        Must contain the following variables:
            - longitude
            - latitude
            - GroundRange
            - CrossRange.
    Returns
    -------
    ground_range_index : ``int``
        The index of the ground range.
    cross_range_index : ``int``
        The index of the cross range.
    """

    # Find the closest longitude and latitude in the MARS dataset
    longitude = DS.longitude.values
    latitude = DS.latitude.values

    # Calculate the distance to the given point
    distances = np.sqrt(
        (longitude - given_longitude) ** 2 + (latitude - given_latitude) ** 2
    )

    # Find the index of the minimum distance
    min_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)

    # Find the corresponding GroundRange and CrossRange indexes
    ground_range_index = DS.GroundRange[min_index[1]].values
    cross_range_index = DS.CrossRange[min_index[0]].values

    ground_range_index = min_index[1]
    cross_range_index = min_index[0]

    return ground_range_index, cross_range_index


def cut_to_extent(DS, extent):
    """
    Cut the dataset to the given extent.
    WARNING: Measures the distance quadratically, not suitable for large distances.

    Parameters
    ----------
    DS : ``xarray.DataSet``
        Dataset to cut to the extent.
        Must contain the following variables:
            - longitude
            - latitude
            - GroundRange
            - CrossRange.
    extent : ``list``
        List of the extent in the form:
        [min_longitude, max_longitude, min_latitude, max_latitude].
    Returns
    -------
    ``xarray.DataSet``
        Dataset fitted to the extent.
    """
    max_GroundRange, max_CrossRange = find_closest_lon_lat(extent[1], extent[3], DS)
    min_GroundRange, min_CrossRange = find_closest_lon_lat(extent[0], extent[2], DS)
    DS_out = DS.isel(
        GroundRange=range(min_GroundRange, max_GroundRange),
        CrossRange=slice(min_CrossRange, max_CrossRange),
    )
    return DS_out
