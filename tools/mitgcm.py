"""
Tools for working with MITgcm model data.
=========================================
Functions solely for working with MITgcm model data.

Functions
---------
- get_extent :
    Returns the extent for the MITgcm model areas of interest.
- split_into_areas :
    Splits the MITgcm model data into different areas based on the extent.
"""

from oscarplus.tools.utils import cut_to_extent

__EXTENT_USHANT_ISLAND = [-5.25, -5, 48.3, 48.6]
__EXTENT_OPEN_SEA = [-6.75, -5.5, 48.5, 49.5]
__EXTENT_SHELF_EDGE = [-6.75, -5.5, 47, 47.7]


def get_extent():
    """
    Returns the extent for the MITgcm model domain.

    Returns
    -------
    extent : ``list``
        List containing the extent for the MITgcm model domain in the form
        [lon_min, lon_max, lat_min, lat_max].
    """
    return {
        "Ushant Island": __EXTENT_USHANT_ISLAND,
        "Open Sea": __EXTENT_OPEN_SEA,
        "Shelf Edge": __EXTENT_SHELF_EDGE,
    }


def split_into_areas(mitgcm):
    """
    Splits the MITgcm model data into different areas based on the extent.

    Parameters
    ----------
    mitgcm : ``xarray.DataSet``
        Dataset containing the MITgcm model data.

    Returns
    -------
    areas : ``dict``
        Dictionary containing the MITgcm model data split into different areas.
        The keys are the area names and the values are the corresponding datasets.
    """
    areas = {}
    for area, extent in get_extent().items():
        areas[area] = cut_to_extent(mitgcm, extent)
    return areas
