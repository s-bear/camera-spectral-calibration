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
This script collects Ocean Optics spectrometer measurements of a spectral light
source (ie. monochromator), processes them and saves into a single H5 file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np

from util import H5Storage, QuietPrint

from numpy.typing import NDArray
from typing import Literal

def _parse_args(args,**kw):
    """Parse command-line arguments for `main`."""
    p = argparse.ArgumentParser()
    p.add_argument('-r','--range',nargs=2,default=[250,850],help='(min, max) of output bin centers, inclusive')
    p.add_argument('-s','--step',default=1.0,help='output bin width. Note: will be adjusted if the range is not evenly divisible')
    p.add_argument('-i','--input',type=Path,help='path to directory containing input files')
    p.add_argument('-o','--output',type=Path,help='Output h5 file path')
    p.add_argument('--quiet',action='store_true',help='Suppress output messages')
    return p.parse_args(args,argparse.Namespace(**kw))

def main(args=None,**kw):
    """Collate and rebin monochromator output spectral irradiances.

    - Scans the input directory for Ocean Optics spectrometer files named ``{wavelength} ND{-density}.txt``
    - Loads them with the `OOSpectrum` class
    - The spectra are rebinned according to ``range`` and ``step``.
    - Noise levels are estimated from the electronic dark pixels
    - The data are saved using the `MonochromatorSpectra` class.
    
    If called without parameters, parses arguments from the command line.
    Arguments may be passed as a list of strings or as keyword arguments. E.g.::

        main(['-i','data/path/','--quiet'])
        # or
        main(input='data/path/',quiet=True)

    Parameters
    ----------
    range : ``-r``, ``--range``, [float, float], default (250.0, 850.0)
        (min, max) of output bin centers, inclusive.
    step : ``-s``, ``--step``, float, default 1.0
        output bin width. Note: the range will be divided into ``1 + int((max - min)/step)`` bins.
    input : ``-i``, ``--input``, Path
        path to directory containing spectra files (OceanOptics spectrometer text output files)
    output : ``-o``, ``--output``, Path
        output h5 file path
    quiet : ``--quiet``, bool
        suppress status message printing
    """
    args = _parse_args(args,**kw)
    with QuietPrint(args.quiet):
        print(f'Scanning for files in {args.input}... ',end='')
        #each filename is '{wavelength} ND{-density}.txt'
        # we can extract those with a regular expression
        spec_file_re = re.compile('(?P<wavelength>[\\d\\.]+) ND(?P<density>-[\\d\\.]+).txt')
        spec_files = []
        for f in args.input.iterdir():
            m = spec_file_re.match(f.name)
            if not m: continue
            wl = float(m.group('wavelength'))
            nd = float(m.group('density'))
            spec_files.append((wl,nd,f))
        spec_files = sorted(spec_files) #sort by wavelength, then density, then filename
        print('DONE')
        
        wl_min, wl_max = args.range
        wl_step = args.step
        rebin_wls = np.linspace(wl_min, wl_max, 1+int((wl_max-wl_min)/wl_step))

        print('Processing... ',end='')
        nominal_wls = []
        nds = []
        wls = None 
        irrad = []
        irrad_std = []
        raw_wls = None
        raw_irrad = []
        
        for wl, nd, path in spec_files:
            print(f'{wl} ',end='')

            data = OOSpectrum(path)
            data.rebin(rebin_wls, nd=nd, correct_dark=True, compute_noise=True)
            
            nominal_wls.append(wl)
            nds.append(nd)
            irrad.append(data.irrad)
            irrad_std.append(data.irrad_std)
            raw_irrad.append(data.raw_irrad)    
        print('DONE')

        out_file = args.output.with_suffix('.h5')
        print(f'Saving {out_file}... ',end='')
        mono = MonochromatorSpectra()
        mono.spectrometer = data.spectrometer #TODO: assumes they're all the same
        mono.date = data.date #TODO: assumes they're all the same
        mono.nom_wl = np.array(nominal_wls)
        mono.wl = data.wls #TODO: assumes they're all the same
        mono.wl_units = data.wl_units #TODO: assumes they're all the same
        mono.mean = np.array(irrad)
        mono.std = np.array(irrad_std)
        mono.n = data.count #TODO: assumes they're all the same
        mono.units = data.irrad_units

        mono.raw_dark_pixels = data.dark_pixels #TODO: assumes they're all the same
        mono.raw_optical_density = nds
        mono.raw_wl = data.raw_wls #TODO: assumes they're all the same
        mono.raw_wl_units = data.wl_units #TODO: assumes they're all the same
        mono.raw_irrad = np.array(raw_irrad)
        mono.raw_irrad_units = data.irrad_units
        mono.raw_n = data.count #TODO: assumes they're all the same
        mono.raw_edc = data.edc #TODO: assumes they're all the same
        mono.raw_nlc = data.nlc #TODO: assumes they're all the same
        
        mono.save(out_file)
        print('DONE')

class MonochromatorSpectra(H5Storage):
    """Class for loading and saving monochromator output spectral irradiances.

    Parameters
    ----------
    path
        ``None`` or path to h5 file to load.
    load_raw
        if ``True``, loads raw spectra in addition to processed spectra.
    """

    _h5members = [(None,None,['spectrometer','date']), #file attributes
        ('nom_wl','nominal_wavelength',[('wl_units','units')]),
        ('wl','wavelength',[('wl_units','units')]),
        ('mean','irradiance',['units','n'],['nominal_wavelength','wavelength']),
        ('std','irradiance_std',['units','n'],['nominal_wavelength','wavelength'])
    ]

    _h5members_raw = [
        ('raw_wl','raw/wavelength',[('raw_wl_units','units')]),
        ('raw_irrad','raw/irradiance',
         [('raw_n','n'),('raw_dark_pixels','dark_pixels'),('raw_irrad_units','units'),('raw_edc','e_dark_correction'),('raw_nlc','nonlinearity_correction')],
         ['nominal_wavelength',('wavelength','raw/wavelength')]),
        ('raw_optical_density','raw/optical_density')
    ]

    def __init__(self, path : Path|None = None, load_raw : bool = False):
        self.spectrometer : str = None #: Spectrometer model and serial number
        self.date : str = None #: Date string of measurements
        
        self.nom_wl : NDArray = None #: Nominal peak wavelength for each spectrum. Shape (N).
        self.wl : NDArray = None #: Wavelengths for each sample in the spectrum. Shape (M).
        self.wl_units : str = None #: Units of `nom_wl` and `wl`.
        
        self.mean : NDArray = None #: Mean spectrum at each nominal peak wavelength. Shape (N,M).
        self.std : NDArray = None #: Standard deviation of spectrum at east nominal peak wavelength. Shape (N,M).
        self.units : str = None #: Units of `mean` and `std`.
        self.n : int = None #: Number of spectra averaged for each nominal peak wavelength.
        self.scale : float = 1.0 #: Scale of `mean` and `std`
        
        self.raw_dark_pixels : tuple[int,int] = None #: (begin,end) indices of optically masked dark pixels. ``dark = raw_irrad[:, begin:end]``
        self.raw_optical_density : NDArray = None #: Optical density attenuating each spectrum at each nominal peak wavelength. Shape (N).
        self.raw_wl : NDArray = None #: Wavelengths for each sample in the raw spectrum. Shape (K).
        self.raw_wl_units : str = None #: Units of `raw_wl`.
        self.raw_n : int = None #: Number of spectra averaged (within the spectrometer software) for each nominal peak wavelength.
        self.raw_irrad : NDArray = None #: Raw spectral measurements. Shape (N, K).
        self.raw_irrad_units : str = None #: Units of `raw_irrad`
        self.raw_edc : bool = None #: True if the spectrometer's internal dark correction was applied.
        self.raw_nlc : bool = None #: True if the spectrometer's internal non-linearity correction was applied.

        if path is not None:
            self.load(path, load_raw)

    def load(self, path : Path, load_raw : bool=False):
        """Load data from ``path``.

        Load data from H5 file at ``path``. Missing items will be set to ``None``.
        ``self.raw_*`` items will only be loaded if ``load_raw`` is ``True``.
        """
        with self._open(path,'r') as f:
            self._load(f, warn_missing=True)
            if load_raw:
                self._load(f, self._h5members_raw, warn_missing=True)
        self.scale = 1.0

    def save(self, path : Path):
        """Save data to ``path``.

        Save data to h5 file at ``path``. Existing files will be overwritten.
        Raw data will only be stored if `raw_wl` and `raw_irrad` are not ``None``.
        """
        #TODO: how should save deal with scale
        with self._open(path,'w') as f:
            self._save(f)
            if self.raw_wl is not None and self.raw_irrad is not None:
                self._save(f, self._h5members_raw)

    def normalize(self,mode : Literal['peak','total']='peak'):
        """Normalize peak or total value to 1. mode is 'peak' or 'total'"""
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

class OOSpectrum:
    """Load and parse OceanOptics spectral data.

    Parameters
    ----------
    path
        Path to Ocean Optics spectrometer data file to load.
    """
    def __init__(self, path : Path):
        #the file opens with 'key: value' metadata lines first
        kvre = re.compile('([^:]*?):\\s*(.*)') #regex for 'key: value'
        
        self.metadata : dict[str,str] = {} #: metadata loaded from the file header section
        with open(path,'r') as f:
            for i,line in enumerate(f):
                if line.startswith('>>>>>Begin Spectral Data<<<<<'): break
                m = kvre.match(line)
                if not m: continue
                key,value = m.groups()
                self.metadata[key] = value
            metadata_rows = i+1
        
        #use numpy's loadtxt to get the rest of the data
        raw_wls, raw_irrad = np.loadtxt(path, skiprows=metadata_rows, unpack=True)
        self.raw_wls : NDArray = raw_wls #: raw wavelengths (not resampled). shape (K)
        self.raw_irrad : NDArray = raw_irrad #: raw irradiance data (not resampled). shape (K)
        self.wls : NDArray = None #: wavelengths (after resampling by `rebin`). shape (M)
        self.irrad : NDArray = None #: irradiance (after resampling by `rebin`). shape (M)
        self.irrad_std : NDArray = None #: irradiance noise estimate. shape (M)
        self.wl_units : str = 'nm' #: units for `raw_wls` and `wls`
        self.irrad_units : str = 'uW/cm2/nm' #: units for `irrad` and `raw_irrad`
        self.spectrometer : str = self.metadata['Spectrometer'] #: spectrometer model and serial
        self.date : str = self.metadata['Date'] #: date string of measurement
        self.time : float = float(self.metadata['Integration Time (sec)']) #: integration time
        self.count : int = int(self.metadata['Scans to average']) #: number of spectra averaged per sample
        self.edc : bool = self.metadata['Electric dark correction enabled'] == 'true' #: True if the spectrometer's internal dark correction has been applied.
        self.nlc : bool = self.metadata['Nonlinearity correction enabled'] == 'true' #: True if the spectrometer's internal nonlinearity correction has been applied
        self.boxcar : int = int(self.metadata['Boxcar width']) #: number of pixels averaged by the spectrometer's internal boxcar filter.
        
        self.dark_pixels : tuple[int,int] = (0,190) #: range of dark pixels -- hardcoded for the Ocean Optics USB4000

    def rebin(self, new_bins, nd = None, correct_dark=True, compute_noise=False):
        """rebin spectral data using `rebin_samples` and estimate noise from dark pixels.
        
        resampled data are stored in `wls`, `irrad` and `irrad_std`.

        Parameters
        ----------
        new_bins
            As per `rebin_samples`
        nd
            if not ``None``, the irradiance is divided by ``10**nd`` to account for
            a neutral density filter's transmittance.
        correct_dark
            if ``True`` use the optically masked dark pixels to correct the irradiance
        compute_noise
            if ``True`` use the optically masked dark pixels to estimate the temporal noise
            (N.B. does not model shot noise!)
        """
        dark_mean = 0
        dark_var = 0
        raw_wls = self.raw_wls
        raw_irrad = self.raw_irrad
        
        if correct_dark:
            dark_mean = np.mean(self.raw_irrad[slice(*self.dark_pixels)])
            raw_wls = self.raw_wls[self.dark_pixels[1]:]
            raw_irrad = self.raw_irrad[self.dark_pixels[1]:] - dark_mean    
        
        if nd:
            transmittance = 10**nd
            raw_irrad = raw_irrad/transmittance
        
        wls, irrad = rebin_samples(raw_wls, raw_irrad, new_bins, True, True)
        irrad_std = None

        if compute_noise:
            # we assume that the total noise variance is (read_var) + (shot_var)
            # read_var can be estimated by taking the variance of the dark pixels
            # shot_var is (electron count)/(averages)
            #   unfortunately the ocean optics file format does not include both the
            #   uncalibrated counts and the calibrated irradiances. It isn't possible
            #   to just use the irradiances as they have been scaled by the transmittance &
            #   calibration coefficients. The shot noise variance would need to be
            #   scaled by the square of the coefficient (var(c*X) = (c**2)*var(X))
            #   In the end, we just ignore the shot noise and accept that the noise
            #   will be underestimated.
            read_var = np.var(self.raw_irrad[slice(*self.dark_pixels)])
            shot_var = 0 # self.raw_counts*self.calibration**2/self.count

            #count how many bins went into each new bin
            bin_count = rebin_samples(raw_wls, np.ones_like(raw_wls), new_bins)[1]
            
            #scale the noise and go back to standard deviation units
            irrad_std = np.sqrt((read_var + shot_var) / bin_count)

        #store the results
        self.wls = wls
        self.irrad = irrad
        self.irrad_std = irrad_std



## Spectral rebinning ##

def bin_edges(centers,axis=0):
    """Compute edges midway between center points along axis."""
    centers = np.asanyarray(centers)
    edges_shape = list(centers.shape)
    edges_shape[axis] += 1
    edges = np.empty_like(centers,shape=edges_shape)
    #move axes (produces a view, leaving original memory layout)
    centers = np.moveaxis(centers, axis, 0)
    edges = np.moveaxis(edges, axis, 0)
    #edges are half-way between centers
    edges[1:-1] = (centers[:-1] + centers[1:])/2
    #first and last edge are symmetrical about centers
    edges[0] = centers[0] - 0.5*(centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5*(centers[-1] - centers[-2])
    return np.moveaxis(edges,0,axis)

def rebin_samples(centers, samples, new_bins=None, input_density=False, output_density=False):
    """Rebin binned samples.
    
    - Assumes each sample is the integral of f(x) over each bin's limits.
    - Assumes the bin edges are evenly spaced between the bin centers.

    New sample values are taken by interpolating the cumulative sum of the
    samples at new bin limits.
    
    Parameters
    ----------
    centers: array_like, 1D
    samples: array_like, same shape as bin_centers
    new_bins: None, scalar, or array_like
        if None: uses the same centers
        if scalar: uses arange(centers[0], centers[-1], new_bins)
        if array_like: new centers
    input_density: boolean
        if True, indicates that the input is normalized by the bin widths.
          e.g. for a photospectrometer the input's units are photons/nm rather than just photons
    output_density: boolean
        if True, indicates that the ouput should be normalized by the bin widths
            
    Returns
    -------
    new_centers : ndarray
    new_samples : ndarray
    """    
    centers, samples = np.asarray(centers), np.asarray(samples)
    #edges and widths of bins
    edges = bin_edges(centers)
    widths = 1
    
    if input_density:
        widths = np.diff(edges)
    
    if new_bins is None:
        if input_density == output_density:
            #no change
            return centers, samples
        else:
            new_centers = centers
            new_edges = edges
    else: #new_bins is not None
        if np.isscalar(new_bins):
            new_centers = np.arange(centers[0],centers[-1],new_bins)
        else:
            new_centers = np.asarray(new_bins)
        new_edges = bin_edges(new_centers)
    new_widths = 1
    if output_density:
        new_widths = np.diff(new_edges)
    
    
    if new_bins is not None:
        # Assume samples[i] = (integral of f(x) dx from edges[i] to edges[i+1]) / widths[i]
        # We want new_samples[i] = (integral of f(x) dx from new_edges[i] to new_edges[i+1]) / new_widths[i]
        # Let F[i] = integral of f(x) dx from edges[0] to edges[i]
        # Then, samples[i]*widths[i] = F[i+1] - F[i]
        #       samples[i+1]*widths[i+1] = F[i+2] - F[i+1]
        # Observe F[0] = integral of f(x) dx from edges[0] to edges[0] = 0
        # Thus samples[0]*widths[0] = F[1] - F[0] = F[1]
        #      samples[1]*widths[1] = F[2] - F[1] = F[2] - samples[0]*widths[0]
        #      samples[2]*widths[2] = F[3] - F[2] = F[3] - (samples[0]*widths[0] + samples[1]*widths[1])
        # Thus F[i], i > 0 = sum(samples[j]*widths[j]) for j = 0 to i-1 (inclusive)
        
        # That is, F[i] is the integral of the underlying function from edges[0] to edges[i]
        # and is calculated by taking the cumulative sum of the samples multiplied by the bin widths
        
        # We estimate the same function for the new bins by interpolating F
        

        #we use the cumulative sum of samples[i]*width[i] to estimate 
        # the integral of f(x) from edges[0] to edges[i].
        F = np.empty(edges.shape)
        F[0] = 0 #integral from edges[0] to edges[0] is 0
        np.cumsum(samples*widths, out=F[1:]) 
        
        #then interpolate to get the same at the new bin edges
        new_F = np.interp(new_edges, edges, F)
        #and take the difference to get the new binning
        new_samples = np.diff(new_F)/new_widths
    else:
        #we're not rebinning, just changing the density weightings
        new_samples = samples*widths/new_widths
    
    return new_centers, new_samples


if __name__ == '__main__':
    main()
