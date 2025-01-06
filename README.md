# OSCARplus
Welcome to OSCARplus python module software repository, used for processing of Ocean Surface Current Airborne Radar demonstrator (OSCAR) data, along with Model for Applications at Regional Scale (MARS2D/3D) and bathymetry.

## 1. Installation
### 1.1 Download
1. Download the latest release from the "Releases" section on the right side of the project page and unzip it.
2. In your home directory, create a folder structure as follows:
   ```
   oscarplus_modules/oscarplus/
   ```
3. Move the contents of `/oscarplus-v[RELEASE VERSION].zip/oscarplus-v[RELEASE VERSION]/` folder into the `oscarplus_modules/oscarplus/` folder.
4. This library depends on the Seastar Project library available here: [Seastar Project library](https://github.com/ACHMartin/seastar_project).
   - Download the `v2023.10.3` release.
   - Move the folder /seastar_project-2023.10.3.zip/seastar_project-2023.10.3/seastar/ to /oscarplus_modules/.

After following these steps your directory structure should look like this:
```
oscarplus_modules/
├── oscarplus/
└── seastar/
```
### 1.2 Create a Python Environment
To install the required Python packages, you can use `conda`. In anaconda prompt run (replace /PATH/TO/ with the path to your oscarplus_modules directory):
```
conda env create -f /PATH/TO/oscarplus_modules/oscarplus/environment.yaml
conda activate oscarplus
```
### 1.3 Add modules to the Python Path
To ensure the code can recognize the modules, add them to your Python path. Run the following command in the anaconda prompt:
```
conda develop /PATH/TO/oscarplus_modules/
```
### 1.4 Add datapaths (optional)
The recommended method for providing data to this module is by specifying paths in a data_dir.txt file.
Alternatively, you can directly pass paths as arguments when calling reader functions.
#### Using data_dir.txt
1. Open the data_dir.txt file.
2. Replace the placeholder /PATH/TO/xxx with the actual paths to your data directories.
   - At a minimum, include the directory containing OSCAR data.
   - Paths to other datasets are optional but can be added if available.
#### OSCAR Data Directory Structure
The directory containing OSCAR data must have subdirectories named according to the following format:
##### Level L1b
```Iroise Sea L1b```
##### Level L1c and higher
```Iroise Sea RRRxRRRm LEVEL```

Here:
```RRR``` represents the resolution of the data in meters (e.g., 200).
```LEVEL``` indicates the processing level (described below).

An example directory structure might look like this:
```
OSCAR/
├── Iroise Sea L1b/
├── Iroise Sea 200x200m L1c/
├── Iroise Sea 200x200m L2/
├── Iroise Sea 200x200m L2 lmout/
└── Iroise Sea 200x200m L2 MF/
```
Inside each directory, the module expects data files following naming scheme:
```YYYYMMDD_Track_AA_OSCAR_RRRxRRR_GMF_STATE.nc```

Where:
```YYYYMMDD``` represents the date,
```AA``` represents the track number,
```OSCAR``` should be included only for level L1b and L1c,
```RRR``` represents the resolution (only for level L1c and higher)
```GMF``` represents the geophysical model function used for L1c to L2 lmout processing (only for level L2 lmout and higher),
```STATE``` represents the state of the data (L1b/c for levels L1b/c, lmout for L2 lmout, L2 for L2 and MF for higher).

Example file names:
```
L1b: 20220522_Track_1_OSCAR_L1b.nc
L1c: 20220522_Track_1_OSCAR_200x200m_L1c.nc
L2 lmout: 20220522_Track_11_200x200m_mouche12_lmout.nc
L2: 20220522_Track_11_200x200m_mouche12_L2.nc
```
#### Recognized Processing Levels
The following processing levels are supported:
- L1b/L1c: Data before the inversion
- L2 lmout: Data containing ambiguities
- L2: Data with ambiguities removed
- L2 MF: L2 data processed with median filtering
- L2a MF: L2 MF data augmented with derivative products (e.g., curl, divergence, etc.)

Kernel restart is required to recognize the changes.
## 3. License
Copyright 2024 Jakub Michalski

Licensed under the MIT License, (the "License"); you may not 
use this file except in compliance with the License. You may obtain a copy of 
the License from this repository (LICENSE).
