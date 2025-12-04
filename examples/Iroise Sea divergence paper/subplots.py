"""
This module contains functions to plot data on a map

Functions
---------
quiver_with_background
    Plots quiver plot with a background of the same data
single
    Plots a single data array on a map
contours
    Plots a contour plot of a DataArray on a map
histogram_with_stats
    Create a histogram of a DataArray
    Add mean, median, standard deviation, and skewness to the plot
"""

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
from numbers import Number
from oscarplus.processing.filtering import downscale
from warnings import warn
from matplotlib.colors import LinearSegmentedColormap


coolwarm_extended = LinearSegmentedColormap.from_list(
    "coolwarm_extended",
    [[0, "xkcd:light sky blue"], [0.25, "#3a4cc0"], [1, "xkcd:brick red"]],
)


def __plot_basics(ax, extent, projection, title, coastlines=True):
    """
    Adds coast,title and gridlines to the plot, sets extent
    Gridlines are added only for PlateCarree projection
    """
    if coastlines:
        if projection == ccrs.PlateCarree():
            ax.add_feature(
                cfeature.GSHHSFeature(scale="full", facecolor="beige", edgecolor="tan"),
                linewidth=2,
                zorder=3,
            )
        else:
            warn("Coastlines are only available for PlateCarree projection")

    if extent is not None:
        if projection == ccrs.PlateCarree():
            ax.set_extent(extent)
        else:
            warn("Extent is only available for PlateCarree projection")
    if title is not None:
        ax.set_title(title)
    if projection == ccrs.PlateCarree():
        gl = ax.gridlines(crs=projection, draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
    return gl


def quiver_with_background(
    L2,
    ax,
    title,
    selection,
    vmax=None,
    projection=ccrs.PlateCarree(),
    extent=None,
    cmap="default",
    vmin=0,
    add_max_value_to_ticks=True,
    coastlines=True,
    coarsen_arrows=False,
    add_cbar=True,
    **kwargs,
):
    """
    Plots quiver plot with a background of the same data
    Plots either current, ocean surface wind or earth surface wind from L2 OSCAR data
    """

    if selection == "Current":
        background_selection = selection + "Velocity"
    elif selection == "OceanSurfaceWind" or selection == "EarthRelativeWind":
        background_selection = selection + "Speed"
    else:
        raise ValueError(
            "selection must be either 'Current',"
            "'OceanSurfaceWind' or 'EarthRelativeWind'"
        )
    # sets custom colormap if none provided
    if cmap == "default":
        cmap = coolwarm_extended
    # downsaple arrows for better visibility
    if coarsen_arrows:
        # if coarsen_arrows is a number > 1, it will be used as the downscale factor
        # if coarsen_arrows is True, the downscale factor will be 2
        if coarsen_arrows > 1:
            L2_arrows = downscale(L2, coarsen_arrows)
        else:
            L2_arrows = downscale(L2, 2)
    else:
        L2_arrows = L2

    background_kwargs = {"vmax": vmax}
    # plot background and set colorbar ticks depending on selection
    background = L2[background_selection].plot(
        x="longitude",
        y="latitude",
        zorder=1,
        cmap=cmap,
        add_colorbar=False,
        robust=False,
        vmin=vmin,
        ax=ax,
        **background_kwargs,
    )

    if add_max_value_to_ticks and vmax is not None:
        if vmax > 10:
            step = 2
        elif vmax > 5:
            step = 1
        else:
            step = 0.5
        ticks = np.append(np.arange(0, vmax, step), vmax)
        cbar_kwargs = {"ticks": ticks}
    else:
        cbar_kwargs = {}

    # find names of u, v parameters
    u = selection + "U"
    v = selection + "V"

    # plot colorbar
    if add_cbar is True:
        cbar = plt.colorbar(
            background,
            ax=ax,
            shrink=0.8,
            pad=0.075,
            location="right",
            format="%.1f",
            **cbar_kwargs,
        )
        cbar.set_label(selection + " velocity [$ms^{-1}$]")

    # plot arrows
    L2_arrows.plot.quiver(
        x="longitude",
        y="latitude",
        u=u,
        v=v,
        pivot="mid",
        zorder=2,
        ax=ax,
        add_guide=False,
        **kwargs,
    )

    gl = __plot_basics(ax, extent, projection, title, coastlines=coastlines)
    return gl


def single(
    DA,
    ax,
    add_cbar=True,
    extent=None,
    title=None,
    projection=ccrs.PlateCarree(),
    coastlines=True,
    cbar_label=None,
    **kwargs,
):
    """
    Plots a single data array on a map
    """

    if "cmap" in kwargs and kwargs["cmap"] == "default":
        kwargs["cmap"] = coolwarm_extended

    single_plot = DA.plot(
        x="longitude",
        y="latitude",
        transform=projection,
        ax=ax,
        add_colorbar=False,
        **kwargs,
    )

    if title is not None:
        ax.set_title(title)  # adds subfigure title
    if extent is not None:
        ax.set_extent(extent)
    # plot colorbar
    if add_cbar:
        cbar = plt.colorbar(
            single_plot,
            ax=ax,
            shrink=0.8,
            pad=0.075,
            location="right",
        )
        if cbar_label is not None:
            cbar.set_label(cbar_label)
    gl = __plot_basics(ax, extent, projection, title, coastlines=coastlines)
    return gl


def contours(
    DA,
    ax,
    extent,
    vmax,
    vmin=0,
    level_step=50,
    cmap="Greens_r",
    legend_title=None,
    legend_location="upper right",
    **kwargs,
):
    """
    Plots a contour plot of a DataArray on a map
    """

    contour_levels = np.arange(vmin, vmax, level_step)
    cs = DA.plot.contour(
        extent=extent,
        levels=contour_levels,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cmap=cmap,
        **kwargs,
    )
    # Manually set the axis limits to match the extent
    ax.set_xlim(extent[0], extent[1])  # Set x-axis limits
    ax.set_ylim(extent[2], extent[3])  # Set y-axis limits

    # Apply contour labels
    legend_handles = []
    for level in contour_levels:
        legend_handles.append(
            mpatches.Patch(color=cs.cmap(cs.norm(level)), label=f"{level}")
        )

    # Add the legend to the plot
    ax.legend(
        handles=legend_handles,
        title=legend_title,
        loc=legend_location,
        fontsize=6,
        title_fontsize=6,
    )


def histogram_with_stats(DA, DA_name, ax, variable, **kwargs):
    """
    Create a histogram of a DataArray
    Add mean, median, standard deviation, and skewness to the plot
    """
    if "bin_number" in kwargs and "bin_width" in kwargs:
        raise ValueError("bin_number and bin_width cannot be used together")

    flat_DA = DA.values.flatten()

    hist_kwargs = {}
    if "bin_number" in kwargs:
        hist_kwargs["bins"] = kwargs["bin_number"]
    elif "bin_width" in kwargs:
        bins = np.concatenate(
            (
                np.arange(
                    0,
                    np.nanmin(flat_DA) - kwargs["bin_width"],
                    -kwargs["bin_width"],
                )[::-1][:-1],
                np.arange(
                    0, np.nanmax(flat_DA) + kwargs["bin_width"], kwargs["bin_width"]
                ),
            )
        )
        hist_kwargs["bins"] = bins

    # Plot the histogram
    ax.hist(flat_DA, histtype="bar", density=True, **hist_kwargs)

    mean_value = DA.mean()
    median_value = DA.median()
    skewness = scipy.stats.skew(flat_DA, nan_policy="omit")
    standard_deviation = DA.std()

    # Label x-axis, y-axis, and add title
    ax.set_xlabel(f"OSCAR surface current {variable}/f")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {DA_name}")

    # Add mean and median lines
    ax.axvline(mean_value, color="r", linestyle="--", label=f"Mean: {mean_value:.2f}")
    ax.axvline(
        median_value, color="y", linestyle="--", label=f"Median: {median_value:.2f}"
    )

    ax.plot([], [], " ")
    ax.plot([], [], " ")

    # Add legend with mean, median, standard deviation, and skewness
    ax.legend(
        loc="upper right",
        fontsize="small",
        labels=[
            f"Mean: {mean_value:.2f}",
            f"Median: {median_value:.2f}",
            f"std: {standard_deviation:.2f}",
            f"Skewness: {skewness:.2f}",
        ],
    )

    if "xlim" in kwargs:
        if isinstance(kwargs["xlim"], Number):
            ax.set_xlim(-kwargs["xlim"], kwargs["xlim"])
        elif len(kwargs["xlim"]) == 2:
            ax.set_xlim(kwargs["xlim"])
        else:
            raise ValueError("xlim must be a number or a list of length 2")
    if "ylim" in kwargs:
        if isinstance(kwargs["ylim"], Number):
            ax.set_ylim(0, kwargs["ylim"])
        elif len(kwargs["ylim"]) == 2:
            ax.set_ylim(kwargs["ylim"])
        else:
            raise ValueError("ylim must be a list of length 1 or 2")
