---
author: Samuel B Powell
copyright: 2022, Samuel B Powell
contact: samuel.powell@uq.edu.au
---
# Mathematical Details
## Spectrometer Model
A typical digital optical spectrometer's response may be modelled as
```{math}
:label: spec-model
f_i = d_i + k \tau A \int_{\lambda_{i-1}}^{\lambda_i} g(\lambda) E(\lambda) d\lambda.
```
Where...

$f_i \in \RR \units{counts} \textup{ for } i \in \{1, \dots, N\}$
: is the $i$'th spectral sample measurement.
  This measurement requires non-trivial calibration and signal processing to yield a useful value.

$d_i \in \RR \units{counts}$
: is the photodetector's dark response.
  This is a random variate which captures the effects of thermal and electronic noise with non-zero mean.
  Its distribution will vary with temperature, integartion time, and the electronic configuration of the photodetector.
  Most spectrometers include optically masked photodetectors to automatically estimate this parameter at the time of measurement, but this assumes the dark response will be uniform across all photodetectors.
  Another strategy for estimating the dark response is to measure a "dark frame" by blocking the spectrometer's input just before taking the actual measurement.

$k \in \RR, \tau \in \RR \units{s}, A \in \RR \units{cm^2}$
: are the electronic gain (unitless), integration time, and photodiode area, respectively.

$\lambda_i \in \RR \units{nm} \textup{ for } i \in \{0,\dots,N\}$
: is the $i$'th sample bin's right edge.
  We assume the bins don't overlap and there is no gap between them, so $\lambda_{i-1}$ is the bin's left edge.
  We also assume the bin widths are approximately uniform and that the sample rate (reciprocal bin width) exceeds the Nyquist rate of the incident irradiance (ie. no aliasing).
  The centre wavelength of each bin is $\lambda_{c,i} = (\lambda_{i-1} + \lambda_{i})/2$.

$g(\lambda) \in \RR \units{counts/\mu J}$
: is the spectrometer's spectral sensitivity.
  We assume $g(\lambda)$ varies smoothly and slowly, so that we can reasonably approximate it with a constant value over each spectral bin.

$E(\lambda) \in \RR \units{\mu W/cm^2/nm}$
: is the incident spectral irradiance, which we assume is constant over the duration of the integration time.

```{note}
We have neglected the rounding error inherent to digital measurements.
Approximating a real value with a digital one introduces noise on the order of half of a least-significant digit, which should be negligible relative to other measurement noise.
```

```{caution}
We have also assumed that the photodetector response is linear.
In practice, linearity should not be expected and $f_i$ should be linearized with some transfer function before applying this model.
This should be handled by the spectrometer's software, but it may be wise to double-check by e.g. measuring the same irradiance with a series of different integration times.
This, of course, pulls the spectrometer's internal clock into the chain of calibration and maybe we should be calibrating it too!
The calibration rabbit-hole is infinite!
```

### Estimating the incident irradiance

To estimate $E(\lambda)$ using {eq}`spec-model`, we assume that $g(\lambda)$ is constant across the bin which allows us to factor it out of the integral.
Then we rearrange to yield:
```{math}
:label: spec_estimate
\hat{E}_i = \frac{f_i - \hat{d}_i}{k \tau A \hat{g}_i} \approx \int_{\lambda_{i-1}}^{\lambda_i} E(\lambda) d\lambda.
```
Where...

$\hat{E}_i \in \RR \units{\mu W/cm^2/nm}$
: is the estimated spectral irradiance at wavelength $\lambda_{c,i}$.

$\hat{d}_i \in \RR \units{counts}$
: is the estimated dark offset, from either the optically masked photodetectors or from a previously measured "dark frame.

$\hat{g}_i \in \RR \units{counts/\mu J}$
: is the estimated spectral response, as produced by the spectrometer's calibration procedure.


### Resampling spectral measurements
Spectrometers often have non-uniformly spaced bin centres, which is not ideal.
A common post-processing step is to resample the data to uniform spacing, or in other words, estimate the $\hat{E}_i$ that would have been measured if the bins were centered on different wavelengths.

Let $\lambda'_i$ for $i \in \{0,\dots,N\}$ be the new bin edges.
The cumulative sum of $\hat{E}_i$ will approximate the integral of $E(\lambda)$ from the beginning of the spectrometer's range.
```{math}
:label: rebin_cumsum
\hat{F}_i = \sum_{j=1}^{i}\hat{E}_j \approx \int_{\lambda_0}^{\lambda_i} E(\lambda) d\lambda.
```
Interpolating the cumulative sum at the new bin edges will approximate the integral from the orignal left-most edge, $\lambda_0$, to the new bin's right edge.
```{math}
:label: rebin_interp
\hat{F}'_i = \underset{\lambda'_i}{\mathrm{interp}} \left\{ (\lambda_i, \hat{F}_i) : i \in 0,\dots, N \right\} \approx \int_{\lambda_0}^{\lambda'_i} E(\lambda) d\lambda.
```
Differencing adjacent samples of $\hat{F}'_i$ yields the spectral irradiance integrated over the new bins.
```{math}
:label: rebin_diff
\hat{E}'_i = \hat{F}'_i - \hat{F}'_{i-1} \approx \int_{\lambda'_{i-1}}^{\lambda'_i} E(\lambda) d\lambda.
```

The accuracy of this method depends on how well the interpolation function can estimate the original underlying spectral irradiance.
If the original sampling rate (reciprocal bin width) exceeds the Nyquist rate of the irradiance, then an appropriate interpolation filter will result in minimal resampling error.

## Camera Model

The spectral response of a typical digital camera's pixel may be modelled as:
```{math}
:label: pixel-model
f = d + k \tau A \int_{\lambda_{min}}^{\lambda_{max}} g(\lambda) E(\lambda) d\lambda.
```

$f \in \RR^c \units{counts}$
: is the pixel value, with dimensionality $c$ for the number of colour channels.

$d \in \RR^c \units{counts}$
: is the pixel's dark response.
  This is a random variate with non-zero mean that captures the effects of thermal and electronic noise.
  It will vary with temperature, gain, and integration time.
  Many modern cameras have features that will automatically compensate for it (e.g. Nikon's "Long Exposure Noise Reduction").
  Otherwise, it can be estimated by capturing "dark frames" by capping the lens and taking a photo using the same settings.

$k \in \RR^{c\times c}, \tau \in \RR \units{s}, A \in \RR \units{cm^2}$
: are the electronic gain / colour matrix (unitless), integration time, and pixel area, respectively.

$\lambda_{min} \le \lambda \le \lambda_{max} \in \RR \units{nm}$
: is the sensitive wavelength range of the camera.

$E(\lambda) \in \RR \units{\mu W/cm^2/nm}$
: is the incident spectral irradiance.

$g(\lambda) \in \RR^c \units{counts/\mu J}$
: is the pixel's spectral sensitivity, with $c$ colour channels.

```{caution}
Again, we have assumed a linear response.
This is not true in general and is explicitly untrue for most common image and video formats.
In practice, it is necessary to use the camera's raw image format and render them into 16-bit TIFFs with a linear profile.
Such images will not "look good" but they will allow error-free mathematical manipulation!
```

### Image sampling and temporal noise estimation
Cameras typically record images with 2D arrays of pixels, and each pixel will have slightly different characteristics.
Image noise can be divided into two components: fixed-pattern noise, due to variations in pixel characteristics, and temporal noise, due to electronic fluctuations over time.
Expanding the pixel model to a model of the whole image yields:
```{math}
:label: image-model
f_{ixy} = r_{ixy} + d_{xy} + k_{xy} \tau A \int_{\lambda_{min}}^{\lambda_{max}} g_{xy}(\lambda) E_{ixy}(\lambda) d\lambda
```

$i \in \{1,\dots,M\}$
: is the image index, indicating variation in time.

$x,y \in \{1,\dots,W\},\{1,\dots,H\}$
: is the pixel location in the 2D image array.

$r_{ixy} \in \RR^c ~ N(0, \Sigma), \Sigma \in \mathrm{diag}(\RR^c)$
: is the temporal noise component, which in this model is zero-mean, independent, and identically distributed (IID) across pixels, but each colour channel may have a different noise power.

The variations in $d_{xy}$, $k_{xy}$, and $g_{xy}$ across the image sensor are known as "fixed-pattern noise" and are apparent when imaging a spatially uniform light source.
Calibrating a camera's fixed-pattern noise is known as "flat fielding" (because it involves photographing a flat light field) and is beyond the scope of this document.
We do, however, want to estimate the *temporal* noise of the pixels, which requires factoring out any spatial variance.

Our dataset consists of $M$ images of the monochromator output, each at a different wavelength $\lambda_i$.
To estimate the temporal noise of the pixels, we assume that all of the pixels have the same spectral response and factor the incident irradiance into a relative spectral distribution, which we assume is constant across the image, and a total irradiance, which varies spatially.
This allows us to factor the spectral response integral out of the image model and replace it with a nominal pixel value $p_i=\tau A \int g(\lambda) E_i(\lambda) d\lambda$.
```{math}
:label: image-model-noise
f_{ixy} = r_{ixy} + d_{xy} + k_{xy} E_{xy} p_i
```

Now we take the spatial average over the image. As the temporal noise is zero-mean and IID, it cancels out.
```{math}
:label: image-model-noise-mean
\overline{f}_i = \frac{1}{WH}\sum_{x,y} f_{ixy} = \overline{d} + \overline{k E} p_i
```

As both the image model {eq}`image-model-noise` and spatial average {eq}`image-model-noise-mean` are linear with respect to $p_i$, we can use a linear regression on each pixel to find a transform that will remove any spatial variations.
```{math}
:label: image-model-noise-regression
\hat{f}_{ixy} = a_{xy} + b_{xy} \overline{f}_i \approx f_{ixy}; \mathrm{s.t.}\underset{a_{xy},b_{xy}}{\mathrm{min}} \| \hat{f}_{ixy} - f_{ixy} \|_2^2
```

The residuals give the temporal noise sample for each pixel:
```{math}
:label: image-model-noise-residual
N_{ixy} = f_{ixy} - \hat{f}_{ixy}
```

```{note}
Flat fielding the image would employ a similar sequence of operations, but requires that $E_{xy}$ be constant across the image. Then $a_{xy}$ and $b_{xy}$ can be used to compensate for variations in $d_{xy}$ and $k_{xy}$. Correcting for variations in the spectral sensitivy across pixels ($g_{xy}) seems insane.
```

### Estimating the camera's spectral response
While the camera's pixel response {eq}`pixel-model` follows the same form as the spectrometer {eq}`spec-model`, we can't make the same assumption that the spectral sensitivity, $g(\lambda)$, is constant over the integral and factor it out.
In some cases it's possible to "un-mix" such an integral using deconvolution techniques, but the application would not be straightforward here.
Instead, we move into the discrete (sampled) domain and infer the camera's spectral response as a regression problem.
```{math}
:label: cam-matrix-model
\begin{align}
f_i &= d_i + k_i \tau_i A \sum_{j=1}^{N} E_{i,j} g_j \\
\hat{f}_i &= \frac{f_i - d_i}{k_i \tau_i A} = \sum_{j=1}^{N} E_{i,j} g_j \\
\mat{\hat{F}} &= \mat{E G}
\end{align}
```

$\mat{\hat{F}} = (\hat{f}_1,\dots,\hat{f}_i,\dots,\hat{f}_M )^\trans \in \RR^{M\times c} \units{counts/s/cm^2}$
: is the matrix of (normalized) measured pixel values.
  The matrix has $M$ rows corresponding to measurements with $c$ columns for the colour channels.

$\mat{G} = (g_1,\dots,g_j,\dots,g_N )^\trans \in \RR^{N\times c} \units{counts/\mu J}$
: is the pixel's discrete spectral sensitivity matrix, with $N$ rows for the spectral samples and $c$ columns for the colour channels.

$\mat{E} = \left( E_{1,1}, \ddots, E_{i,j}, \ddots, E_{M,N} \right) \in \RR^{N\times M} \units{\mu W/cm^2/nm}$
: is the incident spectral irradiance matrix, with $M$ rows for the measurements and $N$ columns for the spectral samples.

Solving {eq}`cam-matrix-model` for $\mat{G}$ appears to be a straightforward linear regression
```{math}
:label: cam-regression-naive
\mat{\hat{G}} = \underset{\mat{G}}{\mathrm{argmin}} \| \mat{\hat{F}} - \mat{E G} \|_2^2
```
but has a few significant caveats due to the nature of the measurements.

We use a monochromator to generate spectral irradiances with smooth, narrow bands that we sweep over the sensitive range of the camera.
We (somewhat arbitrarily) set our monochromator output bandwidth to 10 nm full-width at half-maximum (FWHM), and sweep the output in wavelength by 5 nm steps, corresponding to the Nyquist rate for the 10 nm bandwidth.
This is significantly greater than our spectrometer's sampling period of 1 nm[^note-sampling], leading to an under-determined system when we try to solve {eq}`cam-matrix-model` for the spectral sensitivity matrix $\mat{G}$.
That is, there are infinitely many $\mat{G}$'s that could explain our measurements and we must introduce additional information to the system, in the form of a regularization term, to select the "right" $\mat{G}$.
Here, we use a generalized ridge regression[^note-ridge] to impose a smoothness constraint on the $\mat{G}$ corresponding to the Nyquist rate of the system.
```{math}
:label: cam-regression-ridge
\mat{\hat{G}} = \underset{\mat{G}}{\mathrm{argmin}} \| \mat{\hat{F}}-\mat{EG} \|_2^2 + \|\mat{\Gamma} \mat{G} \|_2^2
```

$\mat{\Gamma} \in \RR^{N\times N}$
: is the Tikhonov matrix.
  $\mat{\Gamma} = \alpha \mat{I}$ yields the standard ridge regression, which penalises solutions with large L-2 norm and selects lower "energy" solutions.
  Setting $\mat{\Gamma} = \mat{I}\ast h$, where $\ast$ is convolution and $h$ is a high-pass filter kernel[^note-convolution], penalises solutions with content above $h$'s frequency cut-off and thus selects lower-frequency, smooth solutions.

[^note-sampling]: We could maybe address this issue by resampling the monochromator output spectra down to the 5 nm sample spacing?
[^note-ridge]: Also known as linear regression with [Tikhonov regularization](https://en.wikipedia.org/wiki/Ridge_regression#Tikhonov_regularization).
[^note-convolution]: Here, $h \in \RR^{1 \times n}$ is a row vector and $(\mat{I} \ast h)\_{i,j}$ $= h\_{i-j+\lceil n/2\rceil}$, or zero if the index is out of bounds.

This regression has a closed-form solution given by
```{math}
:label: cam-ridge-solution
\mat{\hat{G}} = \left(\mat{E}^\trans \mat{E} + \mat{\Gamma}^\trans \mat{\Gamma} \right)^{-1} \mat{E}^\trans \mat{\hat{F}}
```
or it may be solved by minimization techniques if the system becomes too large, or constraints such as a non-negative solution are desired.
