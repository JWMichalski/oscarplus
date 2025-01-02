"""
Spatial selection method for ambiguity removal
==============================================
This module contains functions to solve the ambiguity using spatial ambiguity selection

Functions
---------
- solve_ambiguity_spatial_selection :
    Solves the ambiguity of the L2_lmout dataset using the spatial selection method
"""

import numpy as np


def __single_cell_ambiguity_selection(
    lmout, initial, i_x, i_y, cost_function, window, **kwargs
):
    """
    Selects the ambiguity with the lowest cost based on a box around the cell

    Parameters
    ----------
    lmout : ``xarray.DataSet``
        OSCAR L2 lmout dataset
        This dataset contains the ambiguities to be selected from.
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV'
        data variables.
    initial : ``xarray.DataSet``
        OSCAR L2 dataset
        This dataset contains the initial solution to compare the ambiguities to.
        Must have 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV'
        data variables.
    i_x : ``int``
        Index of the `CrossRange` dimension
    i_j : ``int``
        Index of the `GroundRange` dimension
    cost_function : ``function``
        Function to calculate the cost of the ambiguities.
        Must take:
            - single cell from `lmout`,
            - a box around it from `initial` as input, current_weight.
        Return total cost for all for ambiguities.
    window : ``int``, optional
        Size of the box around the cell.
        Must be an odd number.
    **kwargs : ``**kwargs``, optional
        Additional keyword arguments to pass to the cost function.
    Returns
    -------
    ``int``
        Index of the selected ambiguity
    """
    if window % 2 == 0:
        raise ValueError("Window size must be an odd number")
    radius = np.int_((window - 1) / 2)
    L2_sel = lmout.isel(CrossRange=i_x, GroundRange=i_y)
    if not np.isnan(L2_sel.isel(Ambiguities=0).CurrentU.values):
        if i_x - radius >= 0:
            CrossRange_slice = slice(i_x - radius, i_x + radius + 1)
        else:
            CrossRange_slice = slice(0, i_x + radius + 1)
        if i_y - radius >= 0:
            GroundRange_slice = slice(i_y - radius, i_y + radius + 1)
        else:
            GroundRange_slice = slice(0, i_y + radius + 1)
        total_cost = cost_function(
            L2_sel,
            initial.isel(
                CrossRange=CrossRange_slice,
                GroundRange=GroundRange_slice,
            ),
            **kwargs
        )
        selected_ambiguity = total_cost.argmin()
    else:
        selected_ambiguity = np.nan
    return selected_ambiguity


def solve_ambiguity_spatial_selection(
    lmout,
    initial_solution,
    cost_function,
    iteration_number=2,
    window=3,
    inplace=True,
    **kwargs
):
    """
    Solves the ambiguity of the L2_lmout dataset using a spatial selection method

    Parameters
    ----------
    lmout : ``xarray.DataSet``
        OSCAR L2 lmout dataset
        This dataset contains the ambiguities to be selected from.
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV'
        data variables.
    initial_solution : ``xarray.DataSet``
        OSCAR L2 dataset
        This dataset contains the initial solution to compare the ambiguities to.
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV'
        data variables.
    cost_function : ``function``
        Function to calculate the cost of the ambiguities.
        Must take:
            - single cell from `lmout`
            - a box around it from `initial`
            - any additional keyword arguments
        and return total cost for each of 4 ambiguities.
    pass_number : ``int``, optional
        Number of passes to iterate through the dataset.
        Default is 2.
    inplace : ``bool``, optional
        Whether to modify the input dataset in place.
        Default is True.
    **kwargs : ``**kwargs``, optional
        Additional keyword arguments to pass to the cost function.
    Returns
    -------
    ``xarray.DataSet``
        OSCAR L2 dataset without ambiguities
    """

    def select_and_replace_ambiguity(i, j):
        # select ambiguity with the lowest cost
        selected_ambiguity = __single_cell_ambiguity_selection(
            lmout,
            initial_copy,
            i,
            j,
            cost_function=cost_function,
            window=window,
            **kwargs
        )
        # replace with the selected ambiguity if it is not nan
        if not np.isnan(selected_ambiguity):
            initial_copy.loc[
                {
                    "CrossRange": initial_copy.CrossRange.isel(CrossRange=i),
                    "GroundRange": initial_copy.GroundRange.isel(GroundRange=j),
                }
            ] = lmout.isel(CrossRange=i, GroundRange=j, Ambiguities=selected_ambiguity)

    def verticalpass(direction):
        # iterate vertically in the given direction
        j = halfway_ground_range
        while j >= 0:  # iterate across track
            for i in range(0, cross_range_size, direction):  # iterate along track
                select_and_replace_ambiguity(i, j)
            if j == ground_range_size - 1:
                j = halfway_ground_range - 1
            elif j >= halfway_ground_range:
                j += 1
            elif j < halfway_ground_range:
                j -= 1

    def horizontalpass(direction):
        for i in range(0, cross_range_size):  # iterate along track
            for j in range(0, ground_range_size, direction):  # iterate across track
                select_and_replace_ambiguity(i, j)

    if inplace:
        initial_copy = initial_solution.copy(deep=False)
    else:
        initial_copy = initial_solution.copy(deep=True)

    # initialize arrays
    cross_range_size = lmout.CrossRange.sizes["CrossRange"]
    ground_range_size = lmout.GroundRange.sizes["GroundRange"]
    halfway_ground_range = np.round(ground_range_size / 2).astype(int)

    for n in range(iteration_number):  # repeat passes
        print("Pass", n + 1)
        # Pass A1: iterate vertically
        verticalpass(1)
        # Pass A2: iterate vertically back
        verticalpass(-1)
        # Pass B1: iterate horizontally
        horizontalpass(1)
        # Pass B2: iterate horizontally back
        horizontalpass(-1)
    return initial_copy
