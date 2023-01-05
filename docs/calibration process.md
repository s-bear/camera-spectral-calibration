---
author: Samuel B Powell
copyright: 2022, Samuel B Powell
contact: samuel.powell@uq.edu.au
---
# Calibration Process

As stated in the [introduction](index), spectral calibration requires measuring a set of known spectral irradiances.
One of the most versatile instruments for producing consistent and controllable narrow-band light is a monochromator, which we will be using here.
We will show a comparision of the monochromator's performance relative to a standard colour chart in [a later section](error analysis).
Of course, the monochromator must be calibrated by measuring its output with an optical spectrometer, which must be calibrated against known light sources---a gas discharge lamp with well-defined emission lines at known wavelengths, and a deuterium-halogen lamp with a known, broad-band spectral irradiance.
See {numref}`chain-of-calibration-process`.

```{figure} chain-of-calibration.svg
:name: chain-of-calibration-process
:alt: alt-text

The calibration chain starts with two already-calibrated lamps: a gas discharge lamp that produces emission lines at known wavelengths and a broad-spectrum halogen lamp with a known spectral irradiance. These are used to calibrate an optical spectrometer, which is used to calibrate a monochromator, which is used to calibrate the camera.
```

## Calibrating the spectrometer

Digital optical spectrometers split incident light by wavelength across an array of photodetectors (often a linear CCD).
Each photodetector element serves as a bin, counting the incident photons within a narrow wavelength band during the device’s integration time. These devices require two stages of calibration: first, we must determine which wavelengths each bin is capturing (i.e., wavelength calibration), and second, we must determine how the photodetectors’ counts correspond to physical units (i.e., irradiance calibration). Wavelength calibration is performed by measuring the output of a gas discharge lamp with well-known emission lines, which are used to identify which wavelengths align with which bins. Once the wavelength range of each bin is understood, we measure the output of a broad-spectrum lamp with known spectral irradiance. We then compare our uncalibrated spectrometer’s measurement to the known spectrum and determine a set of coefficients to map our measurement onto the true value. 

In practice, these procedures tend to be included in a spectrometer’s operating software and are straightforward to execute. A spectrometer’s wavelength alignment typically changes slowly, if at all, and thus wavelength calibration only needs to be performed rarely. The device’s spectral sensitivity depends on the spectral transmission of the light-collecting optics as well as the photodetector array’s optoelectronic characteristics—which may depend on the electronic configuration of the device as well as the ambient temperature. During field work we recalibrate our spectrometers’ irradiance response daily, if not more often.

## Calibrating the monochromator

Once the spectrometer is calibrated, it is used to calibrate the monochromator. Monochromators are essentially spectrometers in reverse: a broad-spectrum light source is split by wavelength, and a system of rotating mirrors is used to focus a narrow band of wavelengths out of the device’s exit slit and typically into a fibre optic. The calibration procedure is to set the nominal wavelength of the monochromator to each wavelength of interest (e.g., we use every 5 nm from 300 nm to 800 nm) and measure its output with the spectrometer. These measurements give us the actual peak wavelength of each output, the bandwidth, and may reveal any sidebands or light leakage. One could use this information to adjust the monochromator—correct any wavelength misalignments and adjust the light intensity to emit a desired photon flux at each wavelength—but such is largely unnecessary here. Digital cameras can tolerate a wide range of intensities by varying shutter speeds with no ill effects on the calibration procedure. Minor deviations in peak wavelength are also inconsequential.

Monochromators require regular recalibration as their lamps’ spectral irradiance change with age. With use, metal from the lamp’s filament will slowly vaporize and deposit on the interior surface of the glass bulb, changing its effective spectrum. LEDs also change spectrum with age, though due to other mechanisms. We consider it best practice to recalibrate every 50 hours of use, following the guidelines for recertification of officially certified calibrated lamps.

## Calibrating the camera

We are now ready to calibrate a camera! The camera is set to photograph the monochromator output, preferably filling most of the frame. Because we are only calibrating the spectral response of the camera, it does not matter if the image is in focus—it is perfectly acceptable to put the monochromator’s output very close to the camera’s lens. It is also reasonable to calibrate a camera with no lens—though this means that any lens’s spectral transmission will need to be measured separately. The monochromator is set to each wavelength of interest in turn, and the camera is used to photograph the output. For the best signal-to-noise ratio (SNR), we want the pixels to be well-exposed, but none over-exposed. We target a peak at about 75% on the image histogram. The camera should be set to record RAW images in fully manual mode, including shutter speed, aperture, ISO sensitivity/gain, and white balance. While we could technically use automatic shutter speed or automatic aperture modes, in our experience most cameras do not adjust well to a black frame with a single pure colour dot—exposure bracketing is a more reliable way to ensure at least one good exposure in this scenario. 

Once all of the images are recorded, we begin the data processing. Each image is converted to a standard format and cropped to show just the monochromator output. We select the highest SNR image from each exposure bracket, normalize by the exposure time, and record its mean pixel value. We analyse the entire spectral image stack to determine the fixed-pattern noise (e.g., dust or hot/dead pixels) and subtract it to determine each image’s temporal noise component (read noise and shot noise). We then use the samples along with the monochromator spectral data to infer the camera’s spectral sensitivities as a linear regression problem, described in detail below. 
