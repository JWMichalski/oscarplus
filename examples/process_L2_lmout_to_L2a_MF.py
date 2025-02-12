"""
This script is used to process one L2 lmout file to L2a MF file.
Generates NetCDF files for each step of the process.
"""

import oscarplus as op
import os
import xarray as xr
import seastar as ss
import numpy as np
import warnings

###################
# INPUT PARAMETERS#
###################
# select file
date = "20220517"
track = "2"
gmf = "mouche12kp20_rsv20"
resolution = "200x200m"

# input parameters for simulataneous spatial ambiguity removal removal
weight = 1.5  # weight (windcurrent ratio) for the wind cost
iteration_number = 2  # number of passes

# filename base for saving files
filenamebase = f"{date}_Track_{track}_{resolution}_{gmf}"

##########
# SET UP #
##########

# load files in
L1c, L1c_path = op.tools.readers.read_OSCAR(date, track, gmf, "L1c", resolution)
lmout, L2_path = op.tools.readers.read_OSCAR(date, track, gmf, "L2 lmout", resolution)
csv_path = os.path.join(
    op.tools.readers.get_data_dirs()["wind"],
    "OSCAR Iroise Sea track times_reanalysis.csv",
)

# create mask
mask = op.processing.level1.compute_beam_land_mask(L1c, dilation=2)

##########################
# PROCESS L2 lmout TO L2 #
##########################
print("Processing L2 lmout to L2")

# Build geophysical dataset
winddirection, windspeed = op.tools.readers.get_wind_data(date, track, csv_path)
geo = op.processing.level1.build_geo_dataset(L1c, windspeed, winddirection)

# remove ambiguities for a start point
ambiguity = {
    "name": "closest_truth",
    "method": "wind",
    "truth": geo,
}
L2_sol = ss.retrieval.ambiguity_removal.solve_ambiguity(lmout, ambiguity)

# clean up datasets
lmout = xr.where(mask, lmout, np.nan)
L2_truth = op.processing.level2.prepare_dataset(L2_sol)
lmout_prepped = op.processing.level2.prepare_dataset(lmout)
L2_truth = xr.where(mask, L2_truth, np.nan)

# add attributes to L2_truth
if "DateTaken" not in L2_truth.attrs:
    L2_truth.attrs["DateTaken"] = date
if "Track" not in L2_truth.attrs:
    L2_truth.attrs["Track"] = track
if "GMF" not in L2_truth.attrs:
    L2_truth.attrs["GMF"] = gmf
if "Level" not in L2_truth.attrs:
    L2_truth.attrs["Level"] = "L2 lmout"
if "Resolution" not in L2_truth.attrs:
    L2_truth.attrs["Resolution"] = f"{resolution}"

# spatial ambiguity removal with wind and windcurrent methods
L2 = op.ambiguity_removal.spatial_selection.solve_ambiguity_spatial_selection(
    lmout_prepped,
    L2_truth,
    op.ambiguity_removal.cost_functions.calculate_Euclidian_distance_to_neighbours,
    iteration_number=iteration_number,
    windcurrentratio=weight,
    inplace=False,
)

L2 = op.tools.utils.cut_NaNs(L2)

# clean up datasets
L2 = L2.drop_vars(["x_reduce", "Observables", "x_variables"])
L2.attrs["AmbiguityRemoval"] = (
    f"simultaneous ambiguity removal, windcurrentratio = {weight}, 2 passes"
)
L2.attrs["Level"] = "L2"

# save NetCDF files
SSAR_windcurrent_NetCDFpath = os.path.join(
    op.tools.readers.get_data_dirs()["OSCAR"],
    f"Iroise Sea {resolution} L2",
)
os.makedirs(SSAR_windcurrent_NetCDFpath, exist_ok=True)
L2.to_netcdf(
    os.path.join(
        SSAR_windcurrent_NetCDFpath,
        f"{filenamebase}_L2.nc",
    )
)

#######################
# PROCESS L2 TO L2  MF#
#######################
print("Processing L2 to L2 MF")

# median filtering
L2_MF = op.processing.filtering.component_median_windcurrent(L2)

# update attributes
L2_MF.attrs["Filter"] = "magnitude/component median"
L2_MF.attrs["Level"] = "L2 MF"

# save NetCDF file
MF_NetCDF_path = os.path.join(
    op.tools.readers.get_data_dirs()["OSCAR"], f"Iroise Sea {resolution} L2 MF"
)
os.makedirs(MF_NetCDF_path, exist_ok=True)
L2_MF.to_netcdf(
    os.path.join(
        MF_NetCDF_path,
        f"{filenamebase}_MF.nc",
    )
)

##########################
# PROCESS L2 MF TO L2a MF#
##########################
print("Processing L2 MF to L2a MF")

# calculate secondary products
L2_MF = op.processing.level2.remove_unreliable_cells(L1c, L2_MF)
L2a_MF = L2_MF.copy()
op.processing.secondary_products.calculate_secondary_products(L2a_MF)

# update attributes
L2a_MF.attrs["Level"] = "L2a MF"

# save NetCDF file
L2a_MF_NetCDFpath = os.path.join(
    op.tools.readers.get_data_dirs()["OSCAR"],
    f"Iroise Sea {resolution} L2a MF",
)
os.makedirs(L2a_MF_NetCDFpath, exist_ok=True)
L2a_MF.to_netcdf(
    os.path.join(
        op.tools.readers.get_data_dirs()["OSCAR"],
        "Iroise Sea 200x200m L2a MF",
        f"{filenamebase}_MF.nc",
    )
)

# check if all attributes are present
if not all(
    attr in L2a_MF.attrs
    for attr in [
        "DateTaken",
        "Track",
        "GMF",
        "Level",
        "Resolution",
        "AmbiguityRemoval",
        "Filter",
    ]
):
    print(L2a_MF.attrs)
    warnings.warn("One or more of the attributes are missing in L2a_MF.attrs.")
