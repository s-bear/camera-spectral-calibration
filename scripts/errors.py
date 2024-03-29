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
This script estimates the reconstruction errors of camera_response.py
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import stats

from util import QuietPrint, H5Storage
from monochromator import MonochromatorSpectra, bin_edges
from image_stats import CameraSamples
import camera_response
from camera_response import CameraResponse
import matplotlib.pyplot as plt

def _parse_args(args, **kw):
    """Parse and return command-line arguments for `main`."""
    p = argparse.ArgumentParser(
        description='Compute spectral response reconstruction errors for a given illuminant source',
        epilog='Additional arguments (--bootstrap, --method, etc) are passed to camera_response'
        )
    p.add_argument('-s','--source',help='illuminant source spectral irradiance file')
    p.add_argument('-o','--output',help='output plots path')
    p.add_argument('-b','--bandwidth_range', nargs=3, type=float, default=[5.,26.,1.], metavar='START STOP STEP',
        help='Bandwidth range to analyze, as range(START,STOP,STEP), ie. not inclusive of STOP.')
    p.add_argument('-w','--wavelength_range', nargs=2, type=float, default=[300.0,800.0],metavar='FIRST LAST',
        help='Wavelength range to analyze, inclusive. Sampling will match the source\'s.')
    p.add_argument('--quiet', action='store_true',help='suppress messages')
    
    # we don't want to forward -m or -i to camera_response because we'll be overriding them
    # so we add them here and suppress them
    p.add_argument('-m','--monochromator',help=argparse.SUPPRESS)
    p.add_argument('-i','--input',help=argparse.SUPPRESS)

    #use parse_known_args, which returns (args, unknowns) so that we can forward the unparsed args to camera_response
    return p.parse_known_args(args, argparse.Namespace(**kw))

def main(args=None, **kw):
    """Perform error analysis using synthetic camera response data.
    
    1. Generate a synthetic camera spectral response
    2. Generate samples from that response
    3. Run camera_response.main() on those samples
    4. Perform error analysis

    Parameters
    ----------
    source : ``-s``, ``--source``, Path
        illuminant source spectral irradiance file
    bandwidth_range : ``-b``, ``--bandwidth_range``, (START, STOP, STEP)
        Bandwidth range to analyze, as ``range(START,STOP,STEP)``.
    wavelength_range : ``-w``, ``--wavelength_range``, (FIRST, LAST)
        Wavelength range to analyze, inclusive. Sampling will match the source. 
    quiet : ``--quiet``, bool
        suppress status messages

    Additional arguments are passed on to camera_response.main()
    """
    args, camera_response_args = _parse_args(args,**kw)
    
    #sanitize kw so we don't get any accidental repeated arguments when calling camera_response.main below
    for arg in ['input','output','monochromator','quiet']:
        kw.pop(arg,None)

    with QuietPrint(args.quiet):
        print(f'Loading {args.source}... ',end='')
        irrad = MonochromatorSpectra(args.source)
        print('DONE')

        print('Generating spectral responses... ',end='')
        wl = irrad.wl
        wl_min, wl_max = args.wavelength_range
        wl_mask = (wl >= wl_min)*(wl <= wl_max)
        peaks = wl[wl_mask]
        bw_start, bw_stop, bw_step = args.bandwidth_range
        widths = np.arange(bw_start, bw_stop, bw_step)

        all_responses = make_response(wl[None,:,None], peaks[None,None,:], widths[:,None,None], axis=1) #shape: (widths, wls, peaks)
        print('DONE')

        print('Generating samples... ',end='')
        #use einsum for repeated matrix multiplies
        # irrad.mean is (N_nom, N_wl); resps is (N_bw, N_wl, N_peaks)
        # we want to do ``for i in range(N_bw): all_samples[i] = irrad.mean @ resps[i]``
        all_samples = np.einsum('ij,...jk', irrad.mean, all_responses) #shape: (widths, nom_wl, peaks)
        print('DONE')

        samples = CameraSamples()
        samples.nom_wl = irrad.nom_wl
        samples.wl_units = irrad.wl_units
        samples.n = 1
        samples.std = np.zeros_like(all_samples[0]) #no sample noise -- any noise in the output will be due to the illuminant
        samples.units = 'counts'
        samples.exp_time = 0
        samples.exp_time_units = 's'
        samples.ND = 0
        samples.camera = 'synthetic data'
        samples.lens = 'N/A'
        samples.settings = 'N/A'
        samples.label = 'synthetic data'

        print('Running camera_response for each bandwidth',end='')
        est_response = CameraResponse()
        rms_error = []
        rms_noise = []
        for samples_mean, true_response in zip(all_samples, all_responses):
            # run camera_response
            samples.mean = samples_mean
            camera_response.main(camera_response_args, input=samples,monochromator=irrad,output=est_response,quiet=True, **kw)
            # get error between the true response and estimated response
            rms_error.append(np.sqrt(np.mean((true_response - est_response.mean)**2,axis=0)))
            rms_noise.append(np.sqrt(np.mean(est_response.std**2,axis=0)))
            print('.',end='')
        rms_error = np.array(rms_error) #shape: (widths, peaks)
        rms_noise = np.array(rms_noise)
        print(' DONE')

        print('Saving... ',end='')
        rerr = ReconstructionStats()
        rerr.source = args.source
        rerr.wl = peaks
        rerr.bw = widths
        rerr.rms_error = rms_error
        rerr.rms_noise = rms_noise
        rerr.save(args.output)
        print(' DONE')
        
def make_response(wl, peak, width, offset=0.05, axis=0):
    # find the edges between each wavelength bin
    x = bin_edges(wl,axis)
    s = width/2.3548200450309493 # FWHM = 2.355*sigma
    y = np.diff(stats.norm.cdf(x, loc=peak, scale=s),axis=axis)
    return offset + y*((1-offset)/np.max(y))

class ReconstructionStats(H5Storage):
    _h5members = [
        (None,None,['source']),
        ('wl','wavelength',[('wl_units','units')]),
        ('bw','bandwidth',[('wl_units','units')]),
        ('rms_error','rms_error',[],['bw','wl']),
        ('rms_noise','rms_noise',[],['bw','wl']),
    ]

    def __init__(self, path=None):
        self.source = None
        self.wl = None
        self.wl_units = 'nm'
        self.bw = None
        self.rms_error = None
        self.rms_noise = None

        if path is not None:
            self.load(path)

    def load(self, path):
        with self._open(path,'r') as f:
            self._load(f, warn_missing=True)

    def save(self, path):
        with self._open(path,'w') as f:
            self._save(f)


if __name__ == '__main__':
    main()
