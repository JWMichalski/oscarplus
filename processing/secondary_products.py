"""
OSCAR+ processing secondary_products module
===========================================
This module contains functions to calculate secondary products from the L2_AR_MF dataset

Functions
---------
- calculate_secondary_products:
    Calculate secondary products from the L2_AR_MF dataset
- calculate_upwelling_SWT:
    Find upwelling using shallow water theory
"""

from oscarplus.tools.utils import get_resolution, align_with_track
from oscarplus.tools import calc
from numpy import sin, deg2rad


def __drop_rotated_components(DS):
    """
    Drop the rotated components from the dataset

    Parameters
    ----------
    DS : ``xarray.DataSet``
        Dataset from which to drop the rotated components.
    """
    if "CurrentU_rot" and "CurrentV_rot" in DS:
        DS = DS.drop(
            [
                "CurrentU_rot",
                "CurrentV_rot",
            ]
        )
    if "EarthRelativeWindU_rot" and "EarthRelativeWindV_rot" in DS:
        DS = DS.drop(
            [
                "EarthRelativeWindU_rot",
                "EarthRelativeWindV_rot",
            ]
        )


def calculate_secondary_products(DS, resolution=None):
    """
    Calculate secondary products from the L2_AR_MF dataset

    Parameters
    ----------
    DS : ``xarray.DataSet``
        Dataset for which to calculate the secondary products.
        Must contain the following coordinates: 'latitude', 'longitude'
        and the following variables: 'CurrentU', 'CurrentV'.
        Optional variables to calculate secondary products:
        'EarthRelativeWindU', 'EarthRelativeWindV'.
    resolution : ``int``
        Integer, containing the resolution of the data in meters.
        Default is None.
        If None, DS must contain the Resolution attribute formatted as "YYYxYYYm".
    """
    def calculate_secondary_product(product):
        # Calculate the divergence
        if "CurrentU_rot" in DS:
            DS[f"Current{product}"] = (
                calc.wrap_numpy_2D_vector_calc(
                    DS["CurrentU_rot"], DS["CurrentV_rot"], product
                )
                / resolution
            )  # Divide by resolution to convert to 1/s
        if "EarthRelativeWindU_rot" in DS:
            DS[f"EarthRelativeWind{product}"] = (
                calc.wrap_numpy_2D_vector_calc(
                    DS["EarthRelativeWindU_rot"],
                    DS["EarthRelativeWindV_rot"],
                    product,
                )
                / resolution
            )

    def calculate_secondary_product_div_by_f(product):
        calculate_secondary_product(product)
        if f"Current{product}" in DS:
            DS[f"Current{product}"] = DS[f"Current{product}"] / f
        if f"EarthRelativeWind{product}" in DS:
            DS[f"EarthRelativeWind{product}"] = DS[f"EarthRelativeWind{product}"] / f

    if resolution is None:
        resolution = get_resolution(DS)

    align_with_track(DS)

    f = (
        2
        * 7.2921e-5
        * sin(
            deg2rad((DS["latitude"].max() + DS["latitude"].min()) / 2)
        )  # take an average of the latitudes
    )  # coriolis frequency

    calculate_secondary_product_div_by_f("Divergence")
    calculate_secondary_product_div_by_f("Curl")
    calculate_secondary_product("ShearRate")
    calculate_secondary_product("StrainRate")
    calculate_secondary_product("KineticEnergyDensity")
    calculate_secondary_product("Enstrophy")

    __drop_rotated_components(DS)


def calculate_upwelling_SWT(DS, depth):
    """
    Find upwelling using shallow water theory

    Parameters
    ----------
    DS : ``xarray.DataSet``
        Dataset with coordinates: 'latitude', 'longitude'
        and variables: 'CurrentU_rot', 'CurrentV_rot'
    depth : ``xarray.DataArray``
        Depth data. Coordinates: 'latitude', 'longitude'
        Must increase with depth
    """
    # Interpolate depth to DS using longitude and latitude
    interpolated_depth = depth.interp(
        latitude=DS.latitude, longitude=DS.longitude, method="linear"
    )
    align_with_track(DS)
    hU = interpolated_depth * DS["CurrentU_rot"]
    hV = interpolated_depth * DS["CurrentV_rot"]
    DS["CurrentW"] = calc.wrap_numpy_2D_vector_calc(
        hU, hV, "Divergence"
    ) / get_resolution(DS)
    __drop_rotated_components(DS)
