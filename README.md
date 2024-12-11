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
## 2. License
Copyright 2024 Jakub Michalski

Licensed under the MIT License, (the "License"); you may not 
use this file except in compliance with the License. You may obtain a copy of 
the License from this repository (LICENSE).
