"""
OSCAR+ processing tidal_analysis module
===========================================
This module contains functions to perform tidal analysis

Functions
---------
- split_dataset_by_phase:
    split a dataset into four datasets based on the tidal cycle phase
- spring_neap_phase(ds):
    calculates spring neap cycle phase
"""
import numpy as np


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

    #calculate masks based on the tidal cycle phase in degrees
    mask_HW = (cycle_phase >= 360-45) | (cycle_phase < 45)
    mask_LW = (cycle_phase >= 180-45) & (cycle_phase < 180+45)
    mask_ebb = (cycle_phase >= 90-45) & (cycle_phase < 90+45)
    mask_flood = (cycle_phase >= 270-45) & (cycle_phase < 270+45)

    #make sure that the masks do not overlap
    assert not np.any(mask_HW & mask_LW),    "HW overlaps LW"
    assert not np.any(mask_HW & mask_ebb),   "HW overlaps ebb"
    assert not np.any(mask_HW & mask_flood), "HW overlaps flood"
    assert not np.any(mask_LW & mask_ebb),   "LW overlaps ebb"
    assert not np.any(mask_LW & mask_flood), "LW overlaps flood"
    assert not np.any(mask_ebb & mask_flood), "ebb overlaps flood"

    #apply masks
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
    ds["spring_neap_cycle_phase"]=(ds["M2_cycle_phase"]-["S2_cycle_phase"])%360
