"""
This module contains helper functions for plotting the data.

Functions
---------
make_axes
    Makes axes for the plot
add_letters
    Adds letters to the plots for reference
extract_transect_range
    Extracts the range of the current transect
print_ranges
    Prints the ranges of the current data
plot_column
    Plot a column of the transect dataframe on the given axis
add_arrow
    Add an arrow to the plot
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from string import ascii_lowercase
from matplotlib.patches import FancyArrowPatch


def __calculate_extent(DS, xoffset=0, yoffset=0):
    """Calculates the extent of the plot based on the dataset and the offset"""
    extent = [
        DS.longitude.min().values + xoffset,
        DS.longitude.max().values - xoffset,
        DS.latitude.min().values + yoffset,
        DS.latitude.max().values - yoffset,
    ]
    return extent


def make_axes(DS, nrows, ncols, figsize, dpi, **kwargs):
    """Makes axes for the plot"""
    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"wspace": 0.2, "hspace": 0.007},
        figsize=figsize,
        dpi=dpi,
    )

    if "xoffset" not in kwargs or "yoffset" not in kwargs:
        extent = __calculate_extent(DS, 0, 0)
    else:
        extent = __calculate_extent(DS, kwargs["xoffset"], kwargs["yoffset"])

    return fig, axes, extent


def add_letters(axes):
    """Adds letters to the plots for reference"""
    for n, ax in enumerate(axes.flatten()):
        ax.text(-0.1, 1.1, f"{ascii_lowercase[n]})", transform=ax.transAxes, size=20)


def extract_transect_range(current_transect):
    """Extracts the range of the current transect"""
    coordinates = {}
    coordinates["lat1"], coordinates["lon1"] = (
        current_transect["latitude"].iloc[0],
        current_transect["longitude"].iloc[0],
    )
    coordinates["lat2"], coordinates["lon2"] = (
        current_transect["latitude"].iloc[len(current_transect) - 1],
        current_transect["longitude"].iloc[len(current_transect) - 1],
    )
    return coordinates


def print_ranges(DS):
    """Prints the ranges of the current data"""
    print(f"Max velocity: {DS['CurrentVelocity'].max().values}")
    if "CurrentDivergence" in DS:
        print(
            f"Divergence range: {DS['CurrentDivergence'].min().values},"
            f"{DS['CurrentDivergence'].max().values}"
        )
    if "CurrentW" in DS:
        print(
            f"Vertical velocity range: {DS['CurrentW'].min().values},"
            f"{DS['CurrentW'].max().values}"
        )


def plot_column(
    column_name,
    ax,
    color,
    label,
    df,
    abs_val=False,
    **kwargs,
):
    """Plot a column of the transect dataframe on the given axis"""
    if abs_val:
        ax.plot(
            df["distance"],
            abs(df[column_name]),
            color=color,
            label=label,
            **kwargs,
        )
    else:
        ax.plot(
            df["distance"],
            df[column_name],
            color=color,
            label=label,
            **kwargs,
        )
    ax.set_ylabel(label, color=color)
    ax.tick_params(axis="y", labelcolor=color)


def add_arrow(ax, lon1, lat1, lon2, lat2, color, **kwargs):
    """
    Add an arrow to the plot.

    Parameters
    ----------
    ax : ``matplotlib.axes.Axes``
        The axes to add the arrow to.
    lon1 : ``float``
        Longitude of the starting point.
    lat1 : ``float``
        Latitude of the starting point.
    lon2 : ``float``
        Longitude of the ending point.
    lat2 : ``float``
        Latitude of the ending point.
    """
    # Create a transformation to convert from data coordinates to display coordinates
    trans = ax.transData

    # Create an arrow patch
    arrow = FancyArrowPatch(
        (lon1, lat1),
        (lon2, lat2),
        transform=trans,
        arrowstyle="->",
        mutation_scale=10,
        color=color,
        **kwargs,
    )

    ax.add_patch(arrow)
