"""
Readers module
==============

This module contains the functions that read the data from the files.

Functions
---------
get_data_dirs :
    Return the directories containing the data
read_OSCAR : ``xarray.Dataset, string``
    Read the OSCAR data from the given directory
get_wind_data :
    Load wind data from a CSV file and return the wind direction and velocity
read_MARS2D :
    Read the MARS2D model data and rename the variables to match the OSCAR data
read_MARS3D :
    Read the MARS3D model data and rename the variables to match the OSCAR data
read_SWOT :
    Read the SWOT data and rename the variables to match the OSCAR data
"""

import os
import warnings
import glob
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
    file.close()

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
    Return the directories containing the data

    Returns
    -------
    __data_dirs : ``dict``
        Dictionary listing the directories containing the data.
    """
    return __data_dirs


def read_OSCAR_from_file(
    filepath,
    date,
    track,
    level,
    resolution=None,
    gmf=None,
    warn=True,
):
    """
    Read the OSCAR data from the given file

    Parameters
    ----------
    filepath : ``string``
        Path to the file containing the OSCAR data.
    date : ``string``
        Date of the OSCAR data.
    track : ``string``
        Track of the OSCAR data.
    level : ``string``
        Level of the OSCAR data (L1c, L2 lmout, L2 or L2 MF).
    resolution : ``string``, optional
        Resolution of the OSCAR data.
        REQUIRED for levels other than L1b and L1c,
        if it is not already an attribute in the dataset.
        Not used if level is L1b or L1c.
        Default is None
    gmf : ``string``, optional
        Geophysical Model Function name
        REQUIRED for levels other than L1b and L1c,
        if it is not already an attribute in the dataset.
        Not used if level is L1b or L1c.
        Default is None.
    warn : ``bool``, optional
        Whether to warn if attributes are not found.
        Default is True.
    Returns
    -------
    DS : ``xarray.DataSet``
        Dataset containing the OSCAR data.
    Raises
    ------
    ValueError
        If the GMF or resolution attributes are not found and not provided
        for levels other than L1b and L1c.
    """
    DS = ss.utils.readers.readNetCDFFile(filepath)

    # add atributes to the dataset
    if "DateTaken" not in DS.attrs:
        DS.attrs["DateTaken"] = date
        if warn:
            warnings.warn("DateTaken attribute not found, added to the dataset")
    if "Track" not in DS.attrs:
        DS.attrs["Track"] = track
        if warn:
            warnings.warn("Track attribute not found, added to the dataset")
    if "GMF" not in DS.attrs and level != "L1b" and level != "L1c":
        if gmf is None:
            raise ValueError("GMF attribute not found and not provided")
        DS.attrs["GMF"] = gmf
        if warn:
            warnings.warn("GMF attribute not found, added to the dataset")
    if "Level" not in DS.attrs:
        DS.attrs["Level"] = level
        if warn:
            warnings.warn("Level attribute not found, added to the dataset")
    if "Resolution" not in DS.attrs and level != "L1b" and level != "L1c":
        if resolution is None:
            raise ValueError("Resolution attribute not found and not provided")
        DS.attrs["Resolution"] = f"{resolution}"
        if warn:
            warnings.warn("Resolution attribute not found, added to the dataset")
    return DS


def read_OSCAR(
    date,
    track,
    gmf,
    level,
    resolution="200x200m",
    OSCAR_data_dir=None,
    **kwargs,
):
    """
    Read the OSCAR data from the directory with all the OSCAR data.

    Assumes the data is in the Iroise Sea region
    and different levels are in different subfolders (more in 1.4 in README).
    Automatically selects the file based on the provided attributes.

    Parameters
    ----------
    date : ``string``
        Date of the OSCAR data.
    track : ``string``
        Track of the OSCAR data.
    gmf : ``string``
        Geophysical Model Function name.
    level : ``string``
        Level of the OSCAR data (L1c, L2 lmout, L2 or L2 MF).
    resolution : ``string``, optional
        Resolution of the OSCAR data.
        Default is '200x200m'.
    OSCAR_data_dir : ``string``, optional
        Path to the directory containing the OSCAR data.
        If none is given, the data directory is read from data_dir.txt.
    **kwargs : ``dict``
        Additional keyword arguments.
    Returns
    -------
    DS : ``xarray.DataSet``
        Dataset containing the OSCAR data.
    DS_path : ``string``
        Path to the file containing the OSCAR data.
    """
    if OSCAR_data_dir is None:
        OSCAR_data_dir = __data_dirs["OSCAR"]

    match level:
        case "L1b":
            DS_path = os.path.join(
                OSCAR_data_dir, "Iroise Sea L1b", f"{date}_Track_{track}_OSCAR_L1b.nc"
            )
            DS = read_OSCAR_from_file(
                DS_path, date=date, track=track, level=level, **kwargs
            )
        case "L1c":
            DS_path = os.path.join(
                OSCAR_data_dir,
                f"Iroise Sea {resolution} L1c",
                f"{date}_Track_{track}_OSCAR_{resolution}_L1c.nc",
            )
            DS = read_OSCAR_from_file(
                DS_path, date=date, track=track, level=level, **kwargs
            )
        case "L2 lmout":
            DS_path = os.path.join(
                OSCAR_data_dir,
                f"Iroise Sea {resolution} L2 lmout",
                f"{date}_Track_{track}_{resolution}_{gmf}_lmout.nc",
            )
            DS = read_OSCAR_from_file(
                DS_path,
                date=date,
                track=track,
                level=level,
                resolution=resolution,
                gmf=gmf,
                **kwargs,
            )
        case "L2":
            DS_path = os.path.join(
                OSCAR_data_dir,
                f"Iroise Sea {resolution} L2",
                f"{date}_Track_{track}_{resolution}_{gmf}_L2.nc",
            )
            DS = read_OSCAR_from_file(
                DS_path,
                date=date,
                track=track,
                level=level,
                resolution=resolution,
                gmf=gmf,
                **kwargs,
            )
        case "L2 MF":
            DS_path = os.path.join(
                OSCAR_data_dir,
                f"Iroise Sea {resolution} L2 MF",
                f"{date}_Track_{track}_{resolution}_{gmf}_MF.nc",
            )
            DS = read_OSCAR_from_file(
                DS_path,
                date=date,
                track=track,
                level=level,
                resolution=resolution,
                gmf=gmf,
                **kwargs,
            )
        case "L2a MF":
            DS_path = os.path.join(
                OSCAR_data_dir,
                f"Iroise Sea {resolution} L2a MF",
                f"{date}_Track_{track}_{resolution}_{gmf}_MF.nc",
            )
            DS = read_OSCAR_from_file(
                DS_path,
                date=date,
                track=track,
                level=level,
                resolution=resolution,
                gmf=gmf,
                **kwargs,
            )
        case _:
            raise ValueError("level must be L1b, L1c, L2 lmout, L2, L2 MF, L2a MF")
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


def read_MARS2D(filename, resolution, file_path=None):
    """
    Read the MARS2D model data from the given directory.

    Renames the variables to match the OSCAR data.
    Adds resolution attribute.

    Parameters
    ----------
    filename : ``string``
        Name of the file containing the MARS2D model data.
    resolution : ``string``
        Resolution of the MARS2D data (in meters).
    file_path : ``string``, optional
        Path to the file containing the MARS2D model data.
        If none is given, the data directory is selected from data_dir.txt.
    Returns
    -------
    ``xarray.DataSet``
        Dataset containing the MARS2D model data with the renamed variables.
    """
    if file_path is None:
        file_path = os.path.join(__data_dirs["MARS2D"], filename)
    else:
        file_path = os.path.join(file_path, filename)

    MARS2D = xr.open_mfdataset(file_path)  # change path to select a different file
    # add current velocity and direction
    cvel, cdir = ss.utils.tools.currentUV2VelDir(
        MARS2D["U"].values, MARS2D["V"].values
    )  # converts u and v components to velocity and direction
    MARS2D["CurrentVelocity"] = (("time", "nj", "ni"), cvel)
    MARS2D["CurrentDirection"] = (("time", "nj", "ni"), cdir)
    MARS2D = MARS2D.rename({"ni": "GroundRange", "nj": "CrossRange"})
    current_U, current_V = ss.utils.tools.currentVelDir2UV(
        MARS2D["CurrentVelocity"].values, MARS2D["CurrentDirection"].values
    )  # converts velocity and direction to u and v components
    MARS2D["CurrentU"] = (("time", "CrossRange", "GroundRange"), current_U)
    MARS2D["CurrentV"] = (("time", "CrossRange", "GroundRange"), current_V)
    MARS2D.attrs["Resolution"] = f"{resolution}x{resolution}m"
    return MARS2D


def read_MARS3D(filename, resolution, file_path=None):
    """
    Read the MARS3D model data from the given directory

    Renames the variables to match the OSCAR data.
    Adds resolution attribute.

    Parameters
    ----------
    filename : ``string``
        Name of the file containing the MARS model data.
    resolution : ``string``
        Resolution of the MARS3D data (in meters).
    file_path : ``string``, optional
        Path to the file containing the MARS model data.
        If none is given, the data directory is selected from data_dir.txt.
    Returns
    -------
    MARS3D : ``xarray.DataSet``
        Dataset containing the MARS3D model data with the renamed variables.
    """
    if file_path is None:
        file_path = os.path.join(__data_dirs["MARS3D"], filename)
    else:
        file_path = os.path.join(file_path, filename)

    MARS3D = xr.open_mfdataset(file_path)  # change path to select a different file
    # add current velocity and direction
    cvel, cdir = ss.utils.tools.currentUV2VelDir(
        MARS3D["UZ"].values, MARS3D["VZ"].values
    )  # converts u and v components to velocity and direction
    MARS3D["CurrentVelocity"] = (("time", "level", "nj", "ni"), cvel)
    MARS3D["CurrentDirection"] = (("time", "level", "nj", "ni"), cdir)
    MARS3D = MARS3D.rename({"ni": "GroundRange", "nj": "CrossRange"})
    current_U, current_V = ss.utils.tools.currentVelDir2UV(
        MARS3D["CurrentVelocity"].values, MARS3D["CurrentDirection"].values
    )  # converts velocity and direction to u and v components
    MARS3D["CurrentU"] = (("time", "level", "CrossRange", "GroundRange"), current_U)
    MARS3D["CurrentV"] = (("time", "level", "CrossRange", "GroundRange"), current_V)
    MARS3D.attrs["Resolution"] = f"{resolution}x{resolution}m"
    return MARS3D


def read_SWOT(level, cycle, pass_number, data_dir=None):
    """
    Read the SWOT data.

    Renames the variables to match the OSCAR data.
    Automatically selects the file based on the provided attributes.

    Parameters
    ----------
    level : ``string``
        Level of the SWOT data (L3_unsmoothed, L3_expert).
    cycle : ``string``
        Cycle of the SWOT data (e.g. '001').
    pass_number : ``string``
        Pass number of the SWOT data (e.g. '003').
    data_dir : ``string``, optional
        Path to the directory containing the SWOT data.
        If none is given, the data directory is read from data_dir.txt.
    Returns
    -------
    SWOT : ``xarray.DataSet``
        Dataset containing the SWOT data with the renamed variables.
    Raises
    ------
    ValueError
        If the level is not 'L3_unsmoothed, L3_expert'.
    FileNotFoundError
        If no matching SWOT files are found.
    RuntimeError
        If multiple matching SWOT files are found.
    """

    if data_dir is None:
        SWOT_data_dir = os.path.join(get_data_dirs()["SWOT"], level)
    else:
        SWOT_data_dir = data_dir

    match level:
        case "L3_unsmoothed":
            file_pattern = f"SWOT_L3_LR_SSH_Unsmoothed_{cycle}_{pass_number}_*.nc"
        case "L3_expert":
            file_pattern = f"SWOT_L3_LR_SSH_Expert_{cycle}_{pass_number}_*.nc"
        case _:
            raise ValueError("Level must be 'L3_unsmoothed' or 'L3_expert'")

    file_list = glob.glob(os.path.join(SWOT_data_dir, file_pattern))
    if len(file_list) == 0:
        raise FileNotFoundError("No matching SWOT files found.")
    elif len(file_list) > 1:
        raise RuntimeError("Multiple matching SWOT files found.")

    file_name = os.path.basename(file_list[0])
    SWOT = xr.open_dataset(os.path.join(SWOT_data_dir, file_name))

    SWOT["CurrentU"] = SWOT["ugos_filtered"]
    SWOT["CurrentV"] = SWOT["vgos_filtered"]
    cvel, cdir = ss.utils.tools.currentUV2VelDir(
        SWOT["CurrentU"].values, SWOT["CurrentV"].values
    )
    SWOT["CurrentVelocity"] = (("num_lines", "num_pixels"), cvel)
    SWOT["CurrentDirection"] = (("num_lines", "num_pixels"), cdir)
    SWOT = SWOT.rename_dims({"num_lines": "CrossRange", "num_pixels": "GroundRange"})

    return SWOT


__load_data_dirs()
