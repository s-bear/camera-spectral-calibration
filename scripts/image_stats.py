#!/usr/bin/env python3

# Copyright 2022, Samuel B Powell
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
This script loads a spectral image series, specified by `util.Settings`,
calculates the mean and standard deviation for the highest SNR image in each spectral bracket,
and saves the results to an H5 file.
"""

from __future__ import annotations

import argparse
from util import Settings, H5Storage, QuietPrint
import numpy as np
import imageio.v2 as io
import piexif as exif
from numba import jit
from pathlib import Path


from typing import Tuple,Literal
from numpy.typing import ArrayLike

def _parse_args(args,**kw) -> argparse.Namespace:
    """Parse command-line arguments for `main`."""
    p = argparse.ArgumentParser()
    p.add_argument('-i','--input',type=Path,help='input settings yaml file')
    p.add_argument('-o','--output',type=Path,help='output h5 file')
    p.add_argument('--quiet',action='store_true',help='suppress messages')
    p.add_argument('--max',default=0.995,type=float,help='Max pixel value (0 - 1.0), overexposure threshold')
    return p.parse_args(args,argparse.Namespace(**kw))

def main(args=None,**kw):
    """Compute stats for a spectral image stack.

    - Load the image stack using `load_images` and take the mean.
    - Subtract the fixed-pattern noise (dust spots, etc) relative to the mean using `fpn`--the
      remaining noise is treated as temporal image noise.
    - Finally, the statistics are saved using `CameraSamples`.
    
    If called without parameters, parses arguments from the command line.
    Arguments may be passed as a list of strings or as keyword arguments. E.g.::

        main(['-i','settings.yml','--quiet'])
        # or
        main(input='settings.yml',quiet=True)

    Parameters
    ----------
    input : ``-i``, ``--input``, Path
        input settings yaml file
    output : ``-o``, ``--output``, Path
        output h5 file
    quiet : ``--quiet``, bool
        suppress status messages
    max : ``--max``, float, default 0.995
        set overexposure threshold from (0 - 1.0)
    """
    args = _parse_args(args,**kw)
    with QuietPrint(args.quiet):
        settings = Settings(args.input)
        
        nom_wl, samples, times = load_images(settings, args.max)
        
        print('Estimating fixed pattern and temporal noise... ',end='')


        samples_n = samples[0,:,:,0].size #we're going to average over this many pixels
        samples_mean = np.mean(samples,(1,2)) #(n_wl, channels) -- the value we expect each sample to have (they should be uniform)
        fpn_offset, fpn_bias, fpn_residuals = fpn(samples, samples_mean)
        #the residuals must be due to temporal noise
        samples_noise = np.std(fpn_residuals,(1,2)) #(n_wl, channels)
        print('DONE')

        out_path = Path(args.output).with_suffix('.h5')
        print(f'Saving "{out_path}"... ',end='')
        cs = CameraSamples()
        cs.nom_wl = nom_wl
        cs.wl_units = 'nm'
        cs.mean = samples_mean
        cs.std = samples_noise
        cs.units = 'counts/s'
        cs.n = samples_n
        cs.exp_time = times
        cs.exp_time_units = 's'
        cs.camera = settings.camera
        cs.lens = settings.lens
        cs.settings = settings.settings
        cs.label = settings.label
        cs.ND = settings.nd
        cs.save(out_path)
        print('DONE')

class CameraSamples(H5Storage):
    """Load and save camera sample data.

    Parameters
    ----------
    path
        ``None`` or path to h5 file to load.
    """
    
    _h5members = [
        (None,None,['camera','lens','settings','label']),
        ('nom_wl','nominal_wavelength',[('wl_units','units')]),
        ('exp_time','exposure_time',[('exp_time_units','units')]),
        ('mean','samples',['units','n'],['nominal_wavelength',('channel',None)]),
        ('std','samples_std',['units','n'],['nominal_wavelength',('channel',None)]),
        'ND'
    ]

    def __init__(self, path : Path|None = None):
        self.nom_wl : np.ndarray = None #: Nominal wavelengths for each sample image. Shape (N,).
        self.wl_units : str = None #: Units of `nom_wl`

        self.mean : np.ndarray = None #: Mean of each sample image. Shape (N,C) -- number of samples, colour channels.
        self.std : np.ndarray = None #: Standard deviation of each sample image. Shape (N,C).
        self.n : int = None #: Number of pixels averaged for each sample image.
        self.units : str = None #: Units of `mean` and `std`.

        self.exp_time : np.ndarray = None #: Exposure time of each sample image. Shape (N,).
        self.exp_time_units : str = None #: Units of `exp_time`

        self.ND : float = None #: Neutral density filter value.

        self.camera : str = None #: Camera name/description.
        self.lens : str = None #: Lens name/description.
        self.settings : str = None #: Camera & lens settings used.
        self.label : str = None #: Short name to identify the dataset.
        self.scale : float = 1.0 #: Scale applied to `mean` and `std`. 
        
        if path is not None:
            self.load(path)
    
    def load(self, path : Path):
        """Load data from ``path``.

        Load data from H5 file at ``path``. Missing items will be set to ``None``.
        """
        with self._open(path,'r') as f:
            self._load(f, warn_missing=True)
        self.scale = 1.0

    def save(self, path : Path):
        """Save data to ``path``.

        Save data to h5 file at ``path``. Existing files will be overwritten.
        """
        with self._open(path, 'w') as f:
            self._save(f)

    def normalize(self,mode : Literal['peak','total'] = 'peak'):
        """Normalize peak or total value to 1 based on ``mode``."""
        if mode == 'peak':
            new_scale = np.nanmax(self.mean)
        elif mode == 'total':
            new_scale = np.trapz(self.mean, self.wl,axis=1).max()
        else:
            raise ValueError("mode must be one of ('peak', 'total').")
        self.mean /= new_scale
        self.std /= new_scale
        self.scale *= new_scale

    def denormalize(self):
        """Remove scaling from `normalize`."""
        if self.scale != 1.0:
            self.mean *= self.scale
            self.std *= self.scale
            self.scale = 1.0

def snr(img : ArrayLike) -> float:
    """Compute the SNR of a multi-channel image.

    Computes the signal to noise ratio (SNR) of an N-channel image,
    based on the covariance between colour channels. This captures
    possible correlations between the channels rather than treat
    them as independent random variables. ::

        noise = sqrt(sum(cov(img.reshape((-1,N)).T)))/N
        snr = mean(img)/noise

    Parameters
    ----------
    img : array_like[(..., N)]
        N-channel image.

    Returns
    -------
    snr: float
        The signal to noise ratio of the image
    """
    c = img.shape[-1] #number of channels
    m = np.mean(img)
    cov = np.cov(img.reshape((-1,c)).T)
    return m*c / np.sqrt(np.sum(cov))

def load_images(settings : Settings, overexp : float = 0.995) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Load a spectral image stack.

    The images are specified by `util.Settings` and cropped by ``settings.roi``.
    Images with any pixel values over ``overexp`` are rejected, the image with
    the best SNR from each wavelength bracket is kept and normalized by
    its exposure time.

    This function prints status messages. Use `util.QuietPrint` to suppress them.

    Parameters
    ----------
    settings
        A `util.Settings` instance describing the image stack.
    overexp
        The overexposure threshold.

    Returns
    -------
    wavelengths : (N) ndarray
        The nominal wavelength of each sample.
    samples : (N,H,W,C) ndarray
        The sample image, normalized by the exposure time.
    exptimes : (N) ndarray
        The exposure time of each sample.
    """
    images = settings.images
    roi = settings.roi
    
    def _ratio(r):
        return r[0]/r[1]

    wavelens = []
    samples = []
    times = []
    for wl in sorted(images.keys()):
        idx = images[wl]
        print(f'Loading {wl}',end='')
        best_snr, best_image, best_t = 0, None, 0
        for i in idx:
            ipath = settings.image_file(i,'.tiff')
            #load the photo
            img = io.imread(ipath)
            #crop to roi
            img = img[roi]

            #check for over-exposure
            thresh = overexp*np.iinfo(img.dtype).max
            if np.any(img > thresh):
                print('x',end='')
                continue

            #get exposure time
            meta = exif.load(ipath)
            t = _ratio(meta['Exif'][exif.ExifIFD.ExposureTime])

            #normalize by exposure time
            img = img/t

            #compute signal-to-noise ratio
            img_snr = snr(img)

            if img_snr > best_snr:
                best_snr = img_snr
                best_image = img
                best_t = t
            
            print('.',end='')

        if best_image is not None:
            #keep the best image
            wavelens.append(wl)
            samples.append(best_image)
            times.append(best_t)
        
        print(' DONE')
    wavelens = np.array(wavelens)
    times = np.array(times)
    samples = np.array(samples) #(n_wl, height, width, channels)
    return wavelens, samples, times

# if this were a simpler problem we could just use np.linalg.lstsq,
# but it doesn't scale well to running on every pixel in the image separately
# so we use numba's jit to accelerate these functions
# the jit'ed functions are called by a plain python function so that
# autodoc can handle them correctly.

@jit
def _linear_regression(x,y):
    #  y = a + b*x
    #simple linear regression:
    #  b = cov(x, y)/var(x); a = mean(y) - b*mean(x)
    n = x.shape[0]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)
    xy_cov = np.sum((x-x_mean)*(y-y_mean))/n
    b = xy_cov/x_var
    a = y_mean - b*x_mean
    #r2 = (xy_cov**2)/(x_var*y_var)
    return a,b

@jit
def _fpn(img_stack, stimulus):
    #these images are supposed to be uniform, so any deviation is considered noise
    # we want to do a linear regression on each pixel
    y = img_stack
    x = stimulus
    a = np.empty(y.shape[1:])
    b = np.empty_like(a)
    residuals = np.empty_like(y)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                a_ijk,b_ijk = _linear_regression(x[:,k],y[:,i,j,k])
                a[i,j,k] = a_ijk
                b[i,j,k] = b_ijk
                estimate = a_ijk + b_ijk*x[:,k]
                residuals[:,i,j,k] = y[:,i,j,k] - estimate
    return a, b, residuals

def fpn(img_stack : ArrayLike, stimulus : ArrayLike) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Compute the fixed-pattern noise of an image stack.
    
    Compute the fixed-pattern noise and residuals of an image stack, assuming a linear
    pixel response. For each pixel ``i,j`` we fit a linear model::

        img_stack[:,i,j] = offset[i,j] + bias[i,j]*stimulus[:]
    
    The fixed pattern noise is described by ``offset`` and ``bias``. The residuals
    cover any remaining noise (e.g. temporal).

    Parameters
    ----------
    img_stack : array_like[(N, H, W, C)]
        Stack of N images.
    stimulus : array_like[(N,C)]
        Expected pixel values

    Returns
    -------
    offset : ndarray[(H,W,C)]
    bias : ndarray[(H,W,C)]
    residuals : ndarray[(N, H, W, C)]
        ``residuals = img_stack - (offset + bias*stimulus)``
    """
    return _fpn(img_stack, stimulus)

if __name__ == '__main__':
    main()
    