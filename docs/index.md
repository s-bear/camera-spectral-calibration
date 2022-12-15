---
author: Samuel B Powell
contact: samuel.powell@uq.edu.au
copyright: 2022, Samuel B Powell
---

# Camera Spectral Calibration

```{toctree}
:maxdepth: 2
:caption: Contents
01-process
02-maths
03-errors
04-nikon-results
05-software
```

## Introduction

This document describes a rigorous approach to estimating the relative spectral sensitivities of a digital camera.
Our process maintains a chain of calibration, from certified calibrated light sources through to the digital camera, to ensure consistent results.
Each step of the process is guided by mathematical models of the instruments, including consideration of measurement noise.
Our software is designed around individual scripts for each part of the analysis, coordinated by the Scons build system to promote reproducible results.
All results and intermediate data are stored in self-documenting standard HDF5 files, which are supported by all popular scientific computing packages.

Briefly, spectral calibration of an instrument requires measuring a set of known spectra and inferring the instrument's sensitivity from those measurements.
For a digital camera, this means photographing a set of known spetral irradiances as produced by, for example, a colour chart or a monochromator.
Here we use a monochromator as our specific application requires calibration in the ultraviolet (UV) spectrum, which is not typically covered by a colour chart.
The general idea is the same in either case and we present some results based on the X-Rite ColorChecker[^note-xrite] chart's spectral reflectances.

[^note-xrite]: Now available from [Calibrite](https://calibrite.com).

Of course, maintaining a chain of calibration means that we must calibrate the monochromator (or colour chart), which requires measuring its output with a calibrated optical spectrometer.
The complete chain of calibration is shown in {numref}`chain-of-calibration-intro`.
The details of the spectrometer calibration should be covered by its manual, so we treat it minimally here.
The monochromator and camera [calibration procedures](01-process) are covered first, followed by the [mathematical details](02-maths).
Finally, we present an [error analysis](03-errors) based on a synthetic camera response, and [results](04-nikon-results) from calibrating a Nikon D810 camera.

```{figure} chain-of-calibration.svg
:name: chain-of-calibration-intro
:alt: Flowchart showing the chain of calibration

Arrows indicate which instrument is used to calibrate the next in the chain of calibration.
We start with two externally certified lamps (indicated by the dashed arrows): a gas discharge lamp that produces emission lines at known wavelengths, and a broad-spectrum halogen lamp with a known spectral irradiance.
These are used to calibrate an optical spectrometer, which is used to calibrate the monochromator, which is used to calibrate the camera.
```

```{warning}
Note that we are only aiming for a *relative* spectral calibration.
Specifically, we wish to know the camera's sensitivity ratio between its colour channels within a spectral range (e.g., "the blue channel is 100 times more sensitive than red at 450-455 nm") and between spectral ranges within a single colour channel (e.g., "the red channel is 100 times more sensitive at 650-655 nm than at 450-455 nm").
Performing an absolute, or photometric, spectral calibration requires more sophisticated equipment and more care while performing the measurements, though the mathematical treatment would be similar to what we present here.
```
