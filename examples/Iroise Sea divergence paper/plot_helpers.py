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
import subplots as splot
import numpy as np
from string import ascii_lowercase
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap


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


# Functions

# Colourmap used for bathymetry on the secondary product plots
Bathymetrycmap = LinearSegmentedColormap.from_list(
    "Bathymetrycmap",
    [
        [0.0, "#777777"],
        [0.5, "#444444"],
        [1.0, "#000000"],
    ],
)


def plot_all_three_on_one(
    DS,
    bathymetry,
    figsize,
    legend_location="upper right",
    xoffset=0,
    yoffset=0,
):
    """Plot the current, divergence and vertical current on the same figure"""
    _, axes, extent = make_axes(
        DS,
        1,
        3,
        figsize=figsize,
        dpi=300,
        title=None,
        xoffset=xoffset,
        yoffset=yoffset,
    )

    depth = -bathymetry["elevation"]

    cmaps = ["YlGn", Bathymetrycmap, Bathymetrycmap]

    # Plot bathymetry
    for ax, cmap in zip(axes, cmaps):
        splot.contours(
            depth,
            ax=ax,
            extent=extent,
            vmin=40,
            vmax=120,
            level_step=20,
            legend_title="Elevation",
            linewidths=0.8,
            legend_location=legend_location,
            cmap=cmap,
        )

    # Plot current
    splot.quiver_with_background(
        DS,
        ax=axes[0],
        selection="Current",
        title="Horizontal surface current",
        extent=extent,
        coarsen_arrows=True,
        vmax=3.2,
    )
    # Plot divergence
    gl2 = splot.single(
        DS["CurrentDivergence"],
        ax=axes[1],
        extent=extent,
        title="Surface current divergence",
        cbar_label="Divergence/f",
        vmax=20,
    )
    # Plot vertical current
    gl3 = splot.single(
        DS["CurrentW"],
        ax=axes[2],
        extent=extent,
        title="Vertical surface current",
        cbar_label="Current velocity [$ms^{-1}$]",
        vmax=0.2,
    )

    # Remove left labels from the middle and right plot
    gl2.left_labels = False
    gl3.left_labels = False

    add_letters(axes)
    plt.subplots_adjust(wspace=0.05)

    return axes


def plot_transects(
    current_transects,
    elevation_transects,
    symmetric_velocity=False,
):
    """Plot the transects of the current velocity and the elevation"""

    def make_axes_lim_symmetric(ax):
        """Make the y-axis symmetric around 0"""
        ylim = ax.get_ylim()
        max_lim = max(abs(ylim[0]), abs(ylim[1]))
        ax.set_ylim(-max_lim, max_lim)

    def set_elevation_axis_limits(ax, df_el):
        """
        Set the limits of the elevation axis
        Limits are set to the maximum depth and 0
        """
        max_el = df_el["elevation"].max()
        ax.set_ylim(max_el, 0)

    def velocity_subplot(df_current, df_elevation, ax, max_velocity):
        """Plot the velocity and elevation on the given axis"""
        # make more axis
        ax1 = ax
        ax_velocity = ax1.twinx()  # velocity full
        ax3 = ax1.twinx()  # velocity along
        ax4 = ax1.twinx()  # velocity across
        subaxes = [ax3, ax4]

        # configure spines to avoid overlap
        starting_distance = 40
        for iax in subaxes:
            # Shift the spine outward from the right
            iax.spines["right"].set_position(("outward", starting_distance))
            iax.spines["right"].set_visible(False)  # Hide the default right spine
            iax.set_yticklabels([])
            iax.set_yticks([])
            starting_distance += 10

        # plot elevation
        ax1.invert_yaxis()
        ax1.set_xlabel("Distance [m]")
        plot_column(
            "elevation",
            ax1,
            depth_col,
            "Depth [$m$]",
            df=df_elevation,
            alpha=0.3,
        )
        ax1.fill_between(
            df_elevation["distance"],
            ax1.get_ylim()[0],
            df_elevation["elevation"],
            color=depth_col,
            alpha=0.3,
        )
        set_elevation_axis_limits(ax1, df_elevation)

        # Plot CurrentVelocity
        plot_column(
            "CurrentVelocity",
            ax_velocity,
            "blue",
            df=df_current,
            label="|Horizontal surface velocity| [$ms^{-1}$]",
        )
        ax_velocity.tick_params(axis="y", labelcolor="black")
        ax_velocity.set_ylim(bottom=0, top=max_velocity)

        # Plot CurrentVelocity_along_transect
        plot_column(
            "CurrentVelocity_along_transect",
            ax3,
            velocity_along_col,
            df=df_current,
            label="|Horizontal surface velocity along transect| [$ms^{-1}$]",
            abs_val=True,
        )
        ax3.set_ylim(ax_velocity.get_ylim())

        # Plot CurrentVelocity_across_transect
        plot_column(
            "CurrentVelocity_across_transect",
            ax4,
            "xkcd:light blue",
            "|Horizontal surface velocity across transect| [$ms^{-1}$]",
            df=df_current,
            abs_val=True,
        )
        ax4.set_ylim(ax_velocity.get_ylim())

        ax1.set_xlim(0, df_current["distance"].max())
        for ax in subaxes:
            ax.set_xlim(0, df_current["distance"].max())

    def secondary_subplot(df_current, df_elevation, ax):
        # Make more y-axis
        ax1 = ax
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax4 = ax1.twinx()

        # Now configure all axis
        ax3.spines["left"].set_position(
            ("outward", 50)
        )  # Shift the spine outward from the left
        ax3.spines["right"].set_visible(False)  # Hide the default right spine
        ax4.spines["right"].set_position(
            ("outward", 60)
        )  # Hide the default right spine

        # plot elevation
        ax1.invert_yaxis()
        ax1.set_xlabel("Distance [m]")
        plot_column(
            "elevation",
            ax1,
            depth_col,
            "Depth [$m$]",
            df=df_elevation,
            alpha=0.3,
        )
        ax1.fill_between(
            df_elevation["distance"],
            ax1.get_ylim()[0],
            df_elevation["elevation"],
            color=depth_col,
            alpha=0.3,
        )
        set_elevation_axis_limits(ax1, df_elevation)

        plot_column(
            "CurrentDivergence",
            ax2,
            "#D81B60",
            "Divergence/f",
            linestyle="dashdot",
            df=df_current,
        )
        make_axes_lim_symmetric(ax2)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)

        # plot current velocity along transect
        plot_column(
            "CurrentVelocity_along_transect",
            ax3,
            velocity_along_col,
            "Horizontal surface velocity along transect [$ms^{-1}$]",
            df=df_current,
        )
        ax3.yaxis.set_label_position("left")
        ax3.yaxis.set_ticks_position("left")
        if symmetric_velocity:
            make_axes_lim_symmetric(ax3)
        # Plot CurrentW
        plot_column(
            "CurrentW",
            ax4,
            "black",
            "Vertical surface velocity [$ms^{-1}$]",
            linestyle=CurrentW_style,
            df=df_current,
        )
        make_axes_lim_symmetric(ax4)

        subaxes = (ax1, ax2, ax3)
        for ax in subaxes:
            ax.set_xlim(0, df_current["distance"].max())

    # Set the colours and styles
    depth_col = "#FFC107"
    velocity_along_col = "#1E88E5"
    CurrentW_style = "dotted"
    max_velocity = [2.1, 2.1]

    # Create the subplots
    _, axes = plt.subplots(2, len(current_transects), figsize=(15, 10))
    add_letters(axes)
    plt.subplots_adjust(
        left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.9, hspace=0.3
    )

    # Plot the transects
    for i in range(len(current_transects)):
        velocity_subplot(
            current_transects[i], elevation_transects[i], axes[0, i], max_velocity[i]
        )
        secondary_subplot(current_transects[i], elevation_transects[i], axes[1, i])

    axes[0][0].set_title("Northern jet transect: current velocity")
    axes[0][1].set_title("Southern jet transect: current velocity")
    axes[1][0].set_title("Northern jet transect: divergence and vertical current")
    axes[1][1].set_title("Southern jet transect: divergence and vertical current")


def plot_MARS2D_and_MARS3D_profiles(
    MARS2D,
    MARS3D,
    bathymetry,
    figsize,
    levels,
    points,
    legend_location="upper right",
    xoffset=0,
    yoffset=0,
):
    def plot_vertical(ax, DS, da_name, title, points, xlabel, ylabel):
        ax.scatter(
            DS[da_name].isel(GroundRange=points[0][0], CrossRange=points[0][1]),
            -DS.isel(GroundRange=points[0][0], CrossRange=points[0][1])["level"],
            s=3,
            label="Point A",
        )
        ax.scatter(
            DS[da_name].isel(GroundRange=points[1][0], CrossRange=points[1][1]),
            -DS.isel(GroundRange=points[1][0], CrossRange=points[1][1])["level"],
            s=3,
            label="Point B",
        )
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="lower left")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3)

    top_axes = [None, None, None]
    bottom_axes = [None, None, None]

    for i in range(3):
        top_axes[i] = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())

    bottom_axes[0] = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    bottom_axes[1] = fig.add_subplot(gs[1, 1])
    bottom_axes[2] = fig.add_subplot(gs[1, 2])

    extent = [
        MARS2D.longitude.min().values + xoffset,
        MARS2D.longitude.max().values - xoffset,
        MARS2D.latitude.min().values + yoffset,
        MARS2D.latitude.max().values - yoffset,
    ]

    depth = -bathymetry["elevation"]

    # TOP ROW
    cmaps = ["YlGn", Bathymetrycmap, Bathymetrycmap]

    # Plot bathymetry
    for ax, cmap in zip(top_axes, cmaps):
        splot.contours(
            depth,
            ax=ax,
            extent=extent,
            vmin=40,
            vmax=120,
            level_step=20,
            legend_title="Elevation",
            linewidths=0.8,
            legend_location=legend_location,
            cmap=cmap,
        )

    # Plot current
    gl1 = splot.quiver_with_background(
        MARS2D,
        ax=top_axes[0],
        selection="Current",
        title="MARS2D horizontal surface current",
        extent=extent,
        coarsen_arrows=True,
        vmax=3.2,
    )
    # Plot divergence
    gl2 = splot.single(
        MARS2D["CurrentDivergence"],
        ax=top_axes[1],
        extent=extent,
        title="MARS2D surface current divergence",
        cbar_label="Divergence/f",
        vmax=20,
    )
    # Plot vertical current
    gl3 = splot.single(
        MARS2D["CurrentW"],
        ax=top_axes[2],
        extent=extent,
        title="MARS2D vertical surface current",
        cbar_label="Current velocity [$ms^{-1}$]",
        vmax=0.2,
    )

    # Remove left labels from the middle and right plot
    gl1.bottom_labels = False
    gl2.bottom_labels = False
    gl3.bottom_labels = False
    gl2.left_labels = False
    gl3.left_labels = False

    # BOTTOM ROW
    quiver_kwargs = {
        "vmax": 3.2,
        "scale": 8,
        "headwidth": 5,
        "headlength": 5,
    }

    # Plot bathymetry
    splot.contours(
        depth,
        ax=bottom_axes[0],
        extent=extent,
        vmin=40,
        vmax=120,
        level_step=20,
        legend_title="Elevation",
        linewidths=0.8,
        legend_location=legend_location,
        cmap="YlGn",
    )

    # Plot current
    splot.quiver_with_background(
        MARS3D.sel(level=levels[0]),
        ax=bottom_axes[0],
        selection="Current",
        title=f"MARS3D current at depth of {-levels[0]}$h$",
        extent=extent,
        coarsen_arrows=False,
        **quiver_kwargs,
    )

    for i in range(2):
        bottom_axes[0].plot(
            MARS3D["longitude"].isel(GroundRange=points[i][0], CrossRange=points[i][1]),
            MARS3D["latitude"].isel(GroundRange=points[i][0], CrossRange=points[i][1]),
            marker="*",
            color="white",
            markersize=7,
            transform=ax.projection,
        )
    for i in range(2):
        top_axes[0].plot(
            MARS3D["longitude"].isel(GroundRange=points[i][0], CrossRange=points[i][1]),
            MARS3D["latitude"].isel(GroundRange=points[i][0], CrossRange=points[i][1]),
            marker="*",
            color="white",
            markersize=7,
            transform=ax.projection,
        )

    bottom_axes[0].text(
        MARS3D["longitude"].isel(GroundRange=points[0][0], CrossRange=points[0][1])
        - 0.002,
        MARS3D["latitude"].isel(GroundRange=points[0][0], CrossRange=points[0][1])
        + 0.0035,
        "A",
        fontsize=10,
        color="white",
        transform=ax.projection,
    )
    bottom_axes[0].text(
        MARS3D["longitude"].isel(GroundRange=points[1][0], CrossRange=points[1][1])
        - 0.00175,
        MARS3D["latitude"].isel(GroundRange=points[1][0], CrossRange=points[1][1])
        + 0.0035,
        "B",
        fontsize=10,
        color="white",
        transform=ax.projection,
    )
    top_axes[0].text(
        MARS3D["longitude"].isel(GroundRange=points[0][0], CrossRange=points[0][1])
        - 0.002,
        MARS3D["latitude"].isel(GroundRange=points[0][0], CrossRange=points[0][1])
        + 0.0035,
        "A",
        fontsize=10,
        color="white",
        transform=ax.projection,
    )
    top_axes[0].text(
        MARS3D["longitude"].isel(GroundRange=points[1][0], CrossRange=points[1][1])
        - 0.00175,
        MARS3D["latitude"].isel(GroundRange=points[1][0], CrossRange=points[1][1])
        + 0.0035,
        "B",
        fontsize=10,
        color="white",
        transform=ax.projection,
    )

    plot_vertical(
        bottom_axes[1],
        MARS3D,
        "CurrentVelocity",
        title="MARS3D horizontal current velocity vs depth",
        points=points,
        xlabel="Horizontal current velocity [m/s]",
        ylabel="Depth as fraction of total depth $h$",
    )
    plot_vertical(
        bottom_axes[2],
        MARS3D,
        "CurrentDirection",
        title="MARS3D horizontal current direction vs depth",
        points=points,
        xlabel="Direction [$°$]",
        ylabel="Depth as fraction of total depth $h$",
    )
    bottom_axes[1].set_xlim(0, 3)
    bottom_axes[2].set_xlim(0, 360)

    plt.subplots_adjust(
        left=0.1, right=0.9, top=0.92, bottom=0.1, hspace=0.2, wspace=0.25
    )
    axes = np.concatenate((top_axes, bottom_axes))
    add_letters(axes)

    return axes
