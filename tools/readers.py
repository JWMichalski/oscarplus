"""
Readers module
==============

This module contains the functions that read the data from the files.

Functions
---------
get_data_dirs : ``dict``
    Find the directories containing the data
read_OSCAR : ``xarray.Dataset, string``
    Read the OSCAR data from the given directory
get_wind_data : ``float, float``
    Load wind data from a CSV file and return the wind direction and velocity
read_MARS : ``xarray.Dataset``
    Read the MARS model data from the given directory
"""

import os
import warnings
import xarray as xr
import pandas as pd
import numpy as np
import seastar as ss


__data_dirs = {}


def __load_data_dirs():
    """
    Load the directories containing the OSCAR data
    """
    global __data_dirs
    data_dir_file_loc = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(data_dir_file_loc, "data_dir.txt"), "r") as file:
        for line in file:
            line = line.strip()
            if not line.startswith("#") and line:  # Skip commented or empty lines
                line = line.split(":")
                path = ":".join(line[1:])  # In case the path contains ':'

                if path.startswith(r"/PATH/TO"):
                    __data_dirs[line[0]] = None
                else:
                    __data_dirs[line[0]] = path

    # Assert that all keys and entries are valid
    for key in __data_dirs.keys():
        assert not key.startswith("#"), f"Entry '{key}' in data_dirs starts with '#'"
        if isinstance(__data_dirs[key], str):
            assert not __data_dirs[key].startswith(
                r"/PATH/TO"
            ), f"Entry '{__data_dirs[key]}' in data_dirs starts with '/PATH/TO'"
        else:
            assert (
                __data_dirs[key] is None
            ), f"Entry '{__data_dirs[key]}' in data_dirs is not a string or None"


def get_data_dirs():
    """
    Find the directories containing the data

    Returns
    -------
    __data_dirs : ``dict``
        Dictionary containing the directories containing the data
    """
    return __data_dirs


def read_OSCAR(
    date,
    track,
    gmf,
    level,
    resolution="200x200m",
    OSCAR_data_dir=None,
    warn=True,
    **kwargs,
):
    """
    Read the OSCAR data from the given directory

    Parameters
    ----------
    date : ``string``
        Date of the OSCAR data
    track : ``string``
        Track of the OSCAR data
    gmf : ``string``
        Geophysical Model Function name
    level : ``string``
        Level of the OSCAR data (L1c, L2 lmout, L2 or L2 MF)
    resolution : ``string``, optional
        Resolution of the OSCAR data
        Default is '200x200m'
    OSCAR_data_dir : ``string``, optional
        Path to the directory containing the OSCAR data
        If none is given, the data directory is selected from data_dir.txt
    warn : ``bool``, optional
        Whether to warn if attributes are not found
    **kwargs : ``dict``
        Additional keyword arguments
    Returns
    -------
    DS : ``xarray.Dataset``
        Dataset containing the OSCAR data
    DS_path : ``string``
        Path to the file containing the OSCAR data
    """
    if OSCAR_data_dir is None:
        OSCAR_data_dir = __data_dirs["OSCAR"]

    match level:
        case "L1b":
            window = f"{kwargs['window']}" if "window" in kwargs else "3"
            DS_path = os.path.join(OSCAR_data_dir, "Iroise Sea L1b", f"window{window}")
            DS = ss.utils.readers.readNetCDFFile(
                os.path.join(DS_path, f"{date}_Track_{track}_OSCAR_L1b.nc")
            )
            resolution = f"window{window}"
        case "L1c":
            DS_path = os.path.join(OSCAR_data_dir, f"Iroise Sea {resolution} L1c")
            DS = ss.utils.readers.readNetCDFFile(
                os.path.join(DS_path, f"{date}_Track_{track}_OSCAR_{resolution}_L1c.nc")
            )
        case "L2 lmout":
            DS_path = os.path.join(
                OSCAR_data_dir,
                f"Iroise Sea {resolution} L2 lmout",
            )
            DS = ss.utils.readers.readNetCDFFile(
                os.path.join(
                    DS_path,
                    f"{date}_Track_{track}_{resolution}_{gmf}_lmout.nc",
                )
            )
        case "L2":
            DS_path = os.path.join(
                OSCAR_data_dir, f"Iroise Sea {resolution} L2"
            )
            DS = ss.utils.readers.readNetCDFFile(
                os.path.join(
                    DS_path,
                    f"{date}_Track_{track}_{resolution}_{gmf}_L2.nc",
                )
            )
        case "L2 MF":
            DS_path = os.path.join(OSCAR_data_dir, f"Iroise Sea {resolution} L2 MF")
            DS = ss.utils.readers.readNetCDFFile(
                os.path.join(DS_path, f"{date}_Track_{track}_{resolution}_{gmf}_MF.nc")
            )
        case "L2a MF":
            DS_path = os.path.join(OSCAR_data_dir, f"Iroise Sea {resolution} L2a MF")
            DS = ss.utils.readers.readNetCDFFile(
                os.path.join(DS_path, f"{date}_Track_{track}_{resolution}_{gmf}_MF.nc")
            )
        case _:
            raise ValueError("level must be L1c, L2 lmout, L2, L2 MF, L2a MF")

    # add atributes to the dataset
    if "DateTaken" not in DS.attrs:
        DS.attrs["DateTaken"] = date
        if warn:
            warnings.warn("DateTaken attribute not found, added to the dataset")
    if "Track" not in DS.attrs:
        DS.attrs["Track"] = track
        if warn:
            warnings.warn("Track attribute not found, added to the dataset")
    if "GMF" not in DS.attrs and level != "L1b":
        DS.attrs["GMF"] = gmf
        if warn:
            warnings.warn("GMF attribute not found, added to the dataset")
    if "Level" not in DS.attrs:
        DS.attrs["Level"] = level
        if warn:
            warnings.warn("Level attribute not found, added to the dataset")
    if "Resolution" not in DS.attrs:
        DS.attrs["Resolution"] = f"{resolution}"
        if warn:
            warnings.warn("Resolution attribute not found, added to the dataset")
    return DS, DS_path


def get_wind_data(date, track, csv_path):
    """
    Load wind data from a CSV file and return the wind direction and velocity
    for a given date and track.

    Parameters
    ----------
    date : ``string``
        Date of the OSCAR data
    track : ``string``
        Track of the OSCAR data
    csv_path : ``string``
        Path to the CSV file containing the wind data

    Returns
    -------
    wind_direction : ``float``
        Wind direction in degrees
    wind_velocity : ``float``
        Wind velocity in m/s
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Convert the DataFrame to an xarray dataset
    ds = xr.Dataset.from_dataframe(df)

    wind_direction = np.nan
    wind_velocity = np.nan

    for index in ds.index:
        if int(ds.Date[index]) == int(date):
            if ds.Track[index] == track:
                wind_direction = ds.WindDirection[index].values
                wind_velocity = ds.WindVelocity[index].values
                break

    if np.isnan(wind_direction) or np.isnan(wind_velocity):
        print("No wind data found for the given date and track")

    return wind_direction, wind_velocity


def read_MARS(model_path, resolution):
    """
    Read the MARS model data from the given directory
    Renames the variables to match the OSCAR data
    Adds resolution attribute

    Parameters
    ----------
    model_path : ``string``
        Path to the directory containing the MARS model data
    resolution : ``string``
        Resolution of the MARS data (in meters)
    Returns
    -------
    MARS : ``xarray.Dataset``
        Dataset containing the MARS model data
    """
    MARS = xr.open_mfdataset(model_path)  # change path to select a different file
    # add current velocity and direction
    cvel, cdir = ss.utils.tools.currentUV2VelDir(
        MARS["U"].values, MARS["V"].values
    )  # converts u and v components to velocity and direction
    MARS["CurrentVelocity"] = (("time", "nj", "ni"), cvel)
    MARS["CurrentDirection"] = (("time", "nj", "ni"), cdir)
    # DS=DS.isel(ni=slice(0, 100)) #selects a region of interest
    MARS = MARS.rename({"ni": "GroundRange", "nj": "CrossRange"})
    current_U, current_V = ss.utils.tools.currentVelDir2UV(
        MARS["CurrentVelocity"].values, MARS["CurrentDirection"].values
    )  # converts velocity and direction to u and v components
    MARS["CurrentU"] = (("time", "CrossRange", "GroundRange"), current_U)
    MARS["CurrentV"] = (("time", "CrossRange", "GroundRange"), current_V)
    MARS.attrs["Resolution"] = f"{resolution}x{resolution}m"
    return MARS


__load_data_dirs()
