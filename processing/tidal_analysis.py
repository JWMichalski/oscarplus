"""
OSCAR+ processing tidal_analysis module
===========================================
This module contains functions to perform tidal analysis

Functions
---------
- get_tidal_frequencies:
    Returns the tidal frequencies for M2 and S2 in 1/s and rad/s.
- fit_tidal_constituents:
    Fit tidal constituents to the given values using the UTide package.
- m2_s2_from_coeff:
    Extract the amplitudes and phases of the M2 and S2 tidal constituents
    from the UTide coefficients.
- calculate_cycle_phase:
    Calculate the tidal cycle phase for a given tidal constituent.
- tidal_cycle_phases(ds):
    calculate the tidal cycle phases and add to the dataset
- split_dataset_by_phase:
    split a dataset into four datasets based on the tidal cycle phase
- spring_neap_phase(ds):
    calculates spring neap cycle phase

"""

import numpy as np
import xarray as xr
from utide import solve
from utide.harmonics import ut_E

F_M2 = 0.0805114
OMEGA_M2 = 2 * np.pi * F_M2

F_S2 = 0.0833333
OMEGA_S2 = 2 * np.pi * F_S2


def get_tidal_frequencies():
    """
    Returns the tidal frequencies for M2 and S2 in 1/s and rad/s.
    Returns
    -------
    dict
        Dictionary with tidal frequencies for M2 and S2.
    """
    return {
        "f_M2": F_M2,
        "f_S2": F_S2,
        "omega_M2": OMEGA_M2,
        "omega_S2": OMEGA_S2,
    }


def fit_tidal_constituents(values, latitude, time):
    """
    Fit tidal constituents to the given values using the UTide package.
    Parameters
    ----------
    values : ``array-like``
        The values to fit the tidal constituents to.
    latitude : ``float``
        Latitude of the location in degrees.
    time : ``array-like``
        Time array in datetime64 format.
    Returns
    -------
    ``tuple``
        A tuple containing the amplitudes and phases of
        the M2 and S2 tidal constituents.
    """
    values = np.asarray(values, dtype=float)
    time = np.asarray(time)
    latitude = float(latitude)

    valid = np.isfinite(values)

    if not np.isfinite(latitude) or valid.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan

    coef = solve(
        time[valid],
        values[valid],
        lat=latitude,
        constit=["M2", "S2", "N2", "K1", "O1"],
        method="ols",
        conf_int="linear",
        trend=False,
        nodal=True,
        phase="Greenwich",
        verbose=False,
    )

    return coef


def m2_s2_from_coeff(coef):
    """
    Extract the amplitudes and phases of the M2 and S2 tidal constituents
    from the UTide coefficients.
    Parameters
    ----------
    coef : ``dict``
        Coefficients from the utide.solve function.
    Returns
    -------
    ``tuple``
        A tuple containing the amplitudes and phases
        of the M2 and S2 tidal constituents.
        [A_M2, g_M2, A_S2, g_S2]
    """
    names = np.asarray(coef["name"]).astype(str)

    i_m2 = np.flatnonzero(names == "M2")[0]
    i_s2 = np.flatnonzero(names == "S2")[0]

    return (
        np.float32(coef["A"][i_m2]),
        np.float32(coef["g"][i_m2]),
        np.float32(coef["A"][i_s2]),
        np.float32(coef["g"][i_s2]),
    )


def calculate_cycle_phase(latitude, phi, time, coef, name):
    """
    Calculate the tidal cycle phase for a given tidal constituent.
    Parameters
    ----------
    latitude : ``float``
        Latitude of the location in degrees.
    phi : ``float``
        Greenwich phase of the tidal constituent in degrees.
    time : ``array-like``
        Time array in datetime64 format.
    coef : ``dict``
        Coefficients from the utide.solve function.
    name : ``str``
        Name of the tidal constituent (e.g., "M2", "S2").
    Returns
    -------
    array
        Tidal cycle phase in degrees, with 0 degrees corresponding to high water.
    """
    latitude = float(latitude)
    phi = float(phi)
    time = np.asarray(time)

    if not np.isfinite(latitude) or not np.isfinite(phi):
        return np.full(time.shape, np.nan, dtype=float)

    names = np.asarray(coef["name"]).astype(str)
    i = np.flatnonzero(names == name)[0]

    reftime = coef["aux"]["reftime"]
    frq = np.asarray(coef["aux"]["frq"])[i : i + 1]
    lind = np.asarray(coef["aux"]["lind"])[i : i + 1]
    prefilt = coef["aux"]["opt"]["prefilt"]

    # Convert datetime64 to numerical days relative to UTide reftime
    time_num = ((time - time[0]) / np.timedelta64(1, "D") + reftime).astype(float)

    E = ut_E(
        time_num,
        reftime,
        frq,
        lind,
        latitude,
        [True, False, False, False],
        prefilt,
    )

    V_M2 = np.angle(E[:, 0], deg=True)

    return (V_M2 - phi) % 360.0


def tidal_cycle_phases(ds, print_progress=True):
    """
    Calculate the tidal cycle phases and add to the dataset.
    Parameters
    ----------
    ds : ``xarray.DataSet``
        Dataset to calculate cycle phases for.
        Must have 'Eta' DataArray.
    Returns
    -------
    coef : ``dict``
        Coefficients from the utide.solve function.
    """
    if print_progress:
        print("Calculating M2 and S2")

    coef = xr.apply_ufunc(
        fit_tidal_constituents,
        ds["Eta"],
        ds["latitude"],
        ds["time"],
        input_core_dims=[
            ["time"],
            [],
            ["time"],
        ],
        output_core_dims=[
            [],
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[
            object,
        ],
    )

    ds["M2_amplitude"], ds["M2_phase"], ds["S2_amplitude"], ds["S2_phase"] = (
        xr.apply_ufunc(
            m2_s2_from_coeff,
            coef,
            input_core_dims=[
                [],
            ],
            output_core_dims=[
                [],
                [],
                [],
                [],
            ],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, float, float],
        )
    )

    if print_progress:
        print("Calculating cycle phase")

    M2_cycle_phase = xr.apply_ufunc(
        calculate_cycle_phase,
        ds["latitude"],
        ds["M2_phase"],
        ds["time"],
        coef,
        "M2",
        input_core_dims=[
            [],
            [],
            ["time"],
            [],
            [],
        ],
        output_core_dims=[
            ["time"],
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
        },
    )
    S2_cycle_phase = xr.apply_ufunc(
        calculate_cycle_phase,
        ds["latitude"],
        ds["S2_phase"],
        ds["time"],
        coef,
        "S2",
        input_core_dims=[
            [],
            [],
            ["time"],
            [],
            [],
        ],
        output_core_dims=[
            ["time"],
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
        },
    )

    ds["M2_cycle_phase"] = M2_cycle_phase.compute()
    ds["S2_cycle_phase"] = S2_cycle_phase.compute()

    ds["M2_amplitude"].attrs["units"] = ds["Eta"].attrs.get("units", "")
    ds["M2_phase"].attrs["units"] = "degrees"
    ds["S2_amplitude"].attrs["units"] = ds["Eta"].attrs.get("units", "")
    ds["S2_phase"].attrs["units"] = "degrees"

    ds["M2_cycle_phase"].attrs["units"] = "degrees"
    ds["M2_cycle_phase"].attrs[
        "description"
    ] = "M2 cycle phase relative to high water; 0/360 degrees = high water"

    # units
    ds["M2_amplitude"].attrs["units"] = ds["Eta"].attrs.get("units", "")
    ds["M2_phase"].attrs["units"] = "degrees"
    ds["S2_amplitude"].attrs["units"] = ds["Eta"].attrs.get("units", "")
    ds["S2_phase"].attrs["units"] = "degrees"

    return coef


def split_dataset_by_phase(ds, cycle_phase_var):
    """
    Split a dataset into four datasets based on the tidal cycle phase.

    Parameters
    ----------
    ds : ``xarray.DataSet``
        Dataset to split.
    cycle_phase_var : ``str``
        Name of the variable in ds that contains the tidal cycle phase.
        The variable must be in degrees, with 0 degrees corresponding to high water.
    Returns
    -------
    dict
        Dictionary with four datasets, one for each tidal phase:
        - "HW": high water (0-45 degrees and 315-360 degrees)
        - "LW": low water (135-225 degrees)
        - "ebb": ebb tide (45-135 degrees)
        - "flood": flood tide (225-315 degrees)
    """
    cycle_phase = ds[cycle_phase_var]

    # calculate masks based on the tidal cycle phase in degrees
    mask_HW = (cycle_phase >= 360 - 45) | (cycle_phase < 45)
    mask_LW = (cycle_phase >= 180 - 45) & (cycle_phase < 180 + 45)
    mask_ebb = (cycle_phase >= 90 - 45) & (cycle_phase < 90 + 45)
    mask_flood = (cycle_phase >= 270 - 45) & (cycle_phase < 270 + 45)

    # make sure that the masks do not overlap
    assert not np.any(mask_HW & mask_LW), "HW overlaps LW"
    assert not np.any(mask_HW & mask_ebb), "HW overlaps ebb"
    assert not np.any(mask_HW & mask_flood), "HW overlaps flood"
    assert not np.any(mask_LW & mask_ebb), "LW overlaps ebb"
    assert not np.any(mask_LW & mask_flood), "LW overlaps flood"
    assert not np.any(mask_ebb & mask_flood), "ebb overlaps flood"

    # apply masks
    ds_HW = ds.where(mask_HW, drop=True)
    ds_LW = ds.where(mask_LW, drop=True)
    ds_ebb = ds.where(mask_ebb, drop=True)
    ds_flood = ds.where(mask_flood, drop=True)

    return {
        "HW": ds_HW,
        "LW": ds_LW,
        "ebb": ds_ebb,
        "flood": ds_flood,
    }


def spring_neap_phase(ds):
    """
    Calculates spring neap cycle phase.
    The phase is added in "spring_neap_cycle_phase" DataArray.
    Parameters
    ----------
    ds : ``xarray.DataSet``
        Dataset to calculate phase for.
        Must have 'M2_cycle_phase' and 'S2_cycle_phase' DataArrays.
    Returns
    -------
    None
    """
    ds["spring_neap_cycle_phase"] = (ds["M2_cycle_phase"] - ["S2_cycle_phase"]) % 360
