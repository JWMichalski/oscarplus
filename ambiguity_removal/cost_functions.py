"""
Cost functions for ambiguity removal
====================================
This module contains functions to calculate the cost of the ambiguities

Functions
---------
- calculate_Euclidian_distance_to_neighbours :
    Calculates cost using Euclidian distance or squared Euclidian distance
"""


import numpy as np


def calculate_Euclidian_distance_to_neighbours(
    L2_sel,
    L2_neighbours,
    Euclidian_method="standard",
    method="windcurrent",
    windcurrentratio=10,
    include_centre=False,
):
    """
    Calculates cost using Euclidian distance or squared Euclidian distancee

    The cost is the Euclidian distance
    between each ambiguity of the cell and its neighbours:
    a sum of current distance*current_weight and wind distance
    ----------
    L2_sel : ``xarray.dataset``
        OSCAR L2 dataset containing the cell of interest and its ambiguities.
        Must have 'Ambiguities' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV'
        data variables
    L2_neighbours : xarray dataset
        OSCAR L2 dataset containing the neighbours of the cell of interest
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV'
        data variables
        Additional keyword arguments to pass to the cost function
    Euclidian_method : ``str``, optional
        Method to calculate the Euclidian distance
        Must be 'standard' or 'squared'
        Default is 'standard'
    method : ``str``, optional
        Method to calculate the cost
        Must be 'windcurrent', 'wind' or 'current'
        Default is 'windcurrent'
    windcurrentratio : ``int``, optional
        Ratio of the weight of the current to the weight of the wind
        Default is 10
    include_centre : ``bool``, optional
        Whether to include the centre cell in the cost function
        Default is False
    **kwargs : ``**kwargs``, optional
        Additional keyword arguments to pass to the cost function
    Returns
    -------
    TotalCost : ``numpy.array``
        Total cost for each ambiguity
    """
    if Euclidian_method == "standard":
        power = 0.5
    elif Euclidian_method == "squared":
        power = 1
    else:
        raise ValueError("Euclidian_method must be 'standard' or 'squared'")

    if method == "windcurrent":
        current_multiplier = windcurrentratio
        wind_multiplier = 1
    elif method == "wind":
        current_multiplier = 0
        wind_multiplier = 1
    elif method == "current":
        current_multiplier = 1
        wind_multiplier = 0
    else:
        raise ValueError("method must be 'windcurrent', 'wind' or 'current'")

    centre_cross = np.int_(L2_neighbours.CrossRange.sizes["CrossRange"] / 2)
    centre_ground = np.int_(L2_neighbours.GroundRange.sizes["GroundRange"] / 2)

    dif_squared = (L2_neighbours - L2_sel) ** 2
    dif_squared["dist"] = (
        current_multiplier * (dif_squared.CurrentU + dif_squared.CurrentV) ** power
        + wind_multiplier
        * (dif_squared.EarthRelativeWindU + dif_squared.EarthRelativeWindV) ** power
    )
    dif_squared["distsum"] = dif_squared.dist.sum(dim=("CrossRange", "GroundRange"))

    if not include_centre:
        dif_squared["distsum"] = dif_squared.distsum - dif_squared.dist.isel(
            CrossRange=centre_cross, GroundRange=centre_ground
        )

    return dif_squared.distsum
