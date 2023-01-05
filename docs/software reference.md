---
author: Samuel B Powell
copyright: 2022, Samuel B Powell
contact: samuel.powell@uq.edu.au
---
# Software Reference

The software consists of a suite of Python scripts, each for a specific task.
The various scripts are coordinated by the [SCons](https://scons.org) build tool,
controlled by a [yaml](https://yaml.org) settings file.
This ensures that scripts are re-run when any parameters of the system change, including updated settings or changes to the scripts.
Intermediate results are stored in [H5 files](https://www.hdfgroup.org/solutions/hdf5/), a standardized data format with broad support.
The H5 files are self-documenting with internal metadata, to help prevent data-wrangling mishaps.

## Brief description

  - `SConstruct` is the SCons build script, written in Python, which reads the settings files and describes the various task workflows and dependencies.
  - [`image_stats.py`](software - image_stats) reads all of the calibration images and compiles the image statistics into a `* samples.h5` file.
  - [`monochromator.py`](software - monochromator) reads the monochromator irradiance spectral measurements and compiles them into a single `* monochromator.h5` file.
  - [`camera_response.py`](software - camera_response) reads the samples and monochromator files and infers the camera's spectral response, stored into a `* response.h5` file.
  - [`plots.py`](software - plots) produces a PDF of plots of the response.
  - `collate_responses.py` collates the spectral response data from multiple cameras into a single excel spreadsheet.
  - [`util.py`](software - util) contains various functionality common to the scripts, such as reading the settings and a helper class for dealing with the H5 files.
  - `environment.yml` lists the software dependencies, see more below.

## Installation

Run these commands in your operating system's terminal. On Windows, Microsoft's "Windows Terminal" is a good choice (https://www.microsoft.com/store/productId/9N0DX20HK701).

### 1. Install `conda` and, optionally, `mamba`
This suite of scripts depends on a variety of other software packages, all of which are distributed in the [Anaconda Python/R data science platform](https://anaconda.com).
A minimal (and free) installation is available through the [Miniforge project](https://github.com/conda-forge/miniforge).
Installing the `mamba` tool is highly recommended, as it is much faster than Anaconda's default package management tool `conda`.

If you're starting from scratch, the fastest way is to install Mambaforge: [https://github.com/conda-forge/miniforge#mambaforge](https://github.com/conda-forge/miniforge#mambaforge)

```{note}
If you aren't using the Anaconda Console, make sure to run `conda init` in your terminal after installation. You may have to navigate to the installation directory for this to work on Windows, but you'll only need to do it once!
```

If you already have `conda` installed, then open a terminal and run:

```
conda install mamba
```

### 2. Install dependencies
Installing the scripts' dependencies is straightforward with `mamba`.
Open a terminal in the script directory and run:

```
mamba env create
```

`mamba` will read the `environment.yml` file and create the `camera-cal` [environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), installing the listed software packages.
The default install location will be within your home folder, so administrator privileges are unnecessary.

## Running the scripts

### 1. Activate the environment
First activate the environment by running

```
conda activate camera-cal
```

This will give you access to all of the installed software.

### 2. Run the scripts
Launch the scripts by running `scons`:

```
scons
```

`scons` will read the `SConstruct` script, scan the directory for settings files, and then run the scripts with the parameters from those files.

### 3. Examine the results
The results will be written out to files corresponding to each settings `.yml` file.
For example, the `Nikon D810 UV 60mm macro.yml` file provided as an example produces:
  - `Nikon D810 UV 60mm macro plots.pdf` which contains plots of the camera's spectral response and the measurements used to generate it.
  - `Nikon D810 UV 60mm macro response.h5` which contains all of the numerical data of the spectral response:
    - `response_mean`, `response_std`: Which contain the spectral response and its noise, respectively. Both have shape `(number of spectral bins)x(number of channels)`, and several metadata attributes including `n`, the number of samples used to estimate the noise; and `R2`, the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) of the regression.
    - `wavelength`: Which contains the center wavelength of each spectral bin of `response_mean` and `response_std`.
    - The `samples` group contains the sample measurements used to generate the response.
      - `samples/mean`, `samples/std`: Contain the mean and standard deviation of the pixel values from each measurement image. Both have shape `(number of sample images)x(number of channels)`, and attribute `n`, the number of pixels averaged.
      - `samples/nominal_wavelength`: The nominal wavelength of each sample image.
      - `samples/estimate_mean`, `samples/estimate_std`: The mean and standard deviation of the estimated pixel values based on multiplying the illuminant by the calculated spectral response. These values are used to compute `R2`.


## Reference

```{toctree}
:maxdepth: 2
software - camera_response
software - image_stats
software - monochromator
software - plots
software - response_to_excel
software - util
```

## HDF5 resources
 - Cross-platform file viewer [HDFView](https://www.hdfgroup.org/downloads/hdfview/).
 - Python has support via [h5py](https://www.h5py.org/).
 - Matlab has [built-in support](https://www.mathworks.com/help/matlab/hdf5-files.html).
 - R has support via [rhdf5](https://bioconductor.org/packages/release/bioc/html/rhdf5.html).
