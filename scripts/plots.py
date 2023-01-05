#!/usr/bin/env python3

"""
Make plots of spectral response data.
"""

from __future__ import annotations

"""
Copyright 2022, Samuel B Powell

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import argparse
from util import QuietPrint
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from monochromator import MonochromatorSpectra
from camera_response import CameraResponse, find_peaks

def _parse_args(args,**kw):
    """Parse and return command-line arguments for `main`."""
    p = argparse.ArgumentParser()
    p.add_argument('-r','--response',help='input response file')
    p.add_argument('-m','--monochromator',help='input monochromator spectra file')
    p.add_argument('-o','--output',type=Path, help='output pdf path')
    p.add_argument('--style',default='ggplot',help='matplotlib style name, or path to style file')
    p.add_argument('--logy',action='store_true',help='use logarithmic y scale for monochromator and response')
    p.add_argument('--dpi',default=300,type=int,help='set dpi')
    p.add_argument('--figsize',nargs=2,type=float,default=[8.3,5.8],help='set page size (default A5 landscape)')
    p.add_argument('--margins',nargs=4,type=float,default=[0.5,0.5,1.0,0.75],help='page margins (left, right, top, bottom) in inches')
    p.add_argument('--title-pos',type=float,default=0.75,help='title position from top of page, in inches. ignores margins.')
    p.add_argument('--quiet',action='store_true',help='suppress messages')
    return p.parse_args(args,argparse.Namespace(**kw))

def main(args=None,**kw):
    """Make pdf of figures for a spectral response.

    - Load data
    - Plot monochromator spectral irradiance data
    - Plot camera spectral response
    - Plot camera spectral image stack measurements vs estimated values from the spectral response
    - Save the figures to a pdf

    If called without parameters, parses arguments from the command line.
    Arguments may be passed as a list of strings or as keyword arguments. E.g.::

        main(['-r','response.h5','--quiet'])
        # or
        main(response='response.h5',quiet=True)

    Parameters
    ----------
    response : ``'-r'``, ``'--response'``, Path
        path to spectral response file
    monochromator : ``'-m'``, ``'--monochromator'``, Path
        path to monochromator irradiance file
    output : ``'-o'``, ``'--output'``, Path
        path to pdf plots file
    style: '--style', matplotlib style name
        Matplotlib style name or path to style file. default ``ggplot``
    logy: '--logy', bool
        Use log-y scale
    dpi: '--dpi'
        Figure dpi
    figsize: '--figsize', [width, height]
        Set figure (page) size in inches. Default [8.3,5.8] is A5 landscape
    margins: '--margins', [left, right, top, bottom]
        Set page margins, in inches. Default [0.5, 0.5, 1.0, 0.75]
    title_pos: '--title-pos', float
        Title position from the top of the page, in inches. Ignores margins. Default 0.75
    quiet: '--quiet'
        Suppress progress messages
    """
    args = _parse_args(args,**kw)
    with QuietPrint(args.quiet):
        style_params = {
            'figure.figsize': args.figsize,
            'figure.constrained_layout.use': True,
            'figure.dpi': args.dpi,
            'savefig.dpi': args.dpi,
        }
        plt.style.use(args.style)
        plt.style.use(style_params)

        title_y = 1 - args.title_pos/args.figsize[1] #in figure coordinates

        figs = []

        if args.response:
            print(f'Plotting {args.response}... loading... ',end='')
            resp = CameraResponse(args.response)
            resp.normalize('peak')

            print('plotting response... ',end='')
            fig,ax = plt.subplots()
            set_margins(fig, args.margins)
            fig.suptitle('Camera response',y=title_y)
            plot_response(resp.wl, resp.mean, resp.std, resp.n, logy=args.logy, ax=ax)
            set_text(ax, f'Wavelength ({resp.wl_units})', 'Normalized spectral response')
            figs.append(('fig-resp',fig))

            print('plotting samples... ',end='')
            fig, ax = plt.subplots()
            set_margins(fig, args.margins)
            fig.suptitle('Calibration measurements and estimates',y=title_y)
            rgb_scale = np.nanmax(resp.samples_mean)
            rgb_mean, rgb_std = resp.samples_mean/rgb_scale, resp.samples_std/rgb_scale
            est_mean, est_std = resp.estimates_mean/rgb_scale, resp.estimates_std/rgb_scale
            plot_sample_estimates(resp.samples_wl, rgb_mean, rgb_std, est_mean, est_std,ax=ax)
            set_text(ax, f'Nominal wavelength ({resp.wl_units})','Normalized pixel values')
            figs.append(('fig-meas', fig))
            print('DONE')

        if args.monochromator:
            print(f'Plotting {args.monochromator}... loading... ',end='')
            irrad = MonochromatorSpectra(args.monochromator)
            irrad.normalize('peak')

            print('plotting... ',end='')
            fig, ax = plt.subplots()
            set_margins(fig, args.margins)
            fig.suptitle('Monochromator output irradiance',y=title_y)
            plot_irradiances(irrad.wl, irrad.mean, irrad.std, logy=args.logy, ax=ax)
            set_text(ax, f'Wavelength ({irrad.wl_units})', 'Normalized spectral irradiance')
            figs.append(('fig-irrad',fig))
            print('DONE')

        print('Saving figures... ',end='')
        #workaround for constrained_layout bug (fixed in mpl 3.6.2?)
        plt.rcParams['figure.constrained_layout.use'] = False
        with PdfPages(args.output) as pdf:
            for name, fig in figs:
                pdf.savefig(fig)
        print('DONE')

def set_margins(fig, margins):
    """Set figure margins in inches, (left,right,top,bottom)."""
    #unpack
    left, right, top, bottom = margins
    width, height = fig.get_size_inches()
    #convert margins to figure coordinates
    left, right = left/width, 1-right/width
    bottom, top = bottom/height, 1-top/height
            
    #set layout rect
    engine = fig.get_layout_engine()
    if isinstance(engine, plt.matplotlib.layout_engine.TightLayoutEngine):
        #the tight layout engine uses (left, bottom, right, top)
        rect = (left, bottom, right, top)
    elif isinstance(engine, plt.matplotlib.layout_engine.ConstrainedLayoutEngine):
        #the contrained layout engine uses (left, bottom, width, height)
        rect = (left, bottom, right-left, top-bottom)
    else:
        raise RuntimeError("Can't adjust margins")
    engine.set(rect=rect)
    engine.execute(fig)
    
def set_text(ax, xlabel=None, ylabel=None, title=None):
    """Set axis title and labels."""
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)

def srgb_spectrum(wl, fit=[(0.366,435.,39.3),(1.00,610.,86.0),(0.913,539,90.3),(1.00,453,71.2)]):
    """Estimate sRGB for each wavelength, based on gaussian approximation."""
    #we use the CDF to get the integral of the gaussian over each wavelength bin
    d = wl[1]-wl[0] #assume wl has uniform spacing
    edges = np.linspace(wl[0]-d/2,wl[-1]+d/2,len(wl)+1) #edges halfway between each wl
    norm_fwhm = 2*np.sqrt(2*np.log(2)) #fwhm of a gaussian curve with unit standard deviation
    cdf = lambda s,x,w: s*stats.norm.cdf(edges,loc=x,scale=w/norm_fwhm)/stats.norm.pdf(0,scale=w/norm_fwhm)
    r0,r1,g,b = fit
    red = cdf(*r0) + cdf(*r1)
    green = cdf(*g)
    blue = cdf(*b)
    rgb = np.diff(np.stack((red,green,blue),-1),axis=0)
    return rgb/rgb.max() #TODO: figure out why it wasn't normalized from the maths

def plot_irradiances(wl, irrad, irrad_std=0, likelihood=0.95, logy=False, ax=None):
    """Plot monochromator irradiances."""
    if ax is None: ax = plt.gca()
    wl_peak = find_peaks(wl, irrad)
    srgb = srgb_spectrum(wl_peak)
    ax.set_prop_cycle('color',srgb)
    ax.plot(wl, irrad.T)
    thresh = 10**(np.ceil(np.log10(irrad.max())-4)) #4 orders of magnitude below max
    if np.max(irrad_std) > thresh:
        irrad_low, irrad_hi = stats.norm.interval(likelihood, loc=irrad, scale=irrad_std)
        for i in range(irrad_low.shape[0]):
            ax.fill_between(wl, irrad_low[i], irrad_hi[i],alpha=0.5)
    if logy:
        ax.set_yscale('symlog',linthresh=thresh,linscale=0.2,subs=np.arange(2,10))
    return ax

def plot_sample_estimates(wl, rgb_mean, rgb_std, est_mean, est_std, likelihood=0.95, ax=None):
    """Plot camera samples and their regression estimates."""
    if ax is None: ax = plt.gca()
    ax.set_prop_cycle('color',['darkred','darkgreen','darkblue','red','green','blue'])
    rgb_lines = ax.plot(wl,rgb_mean)
    est_lines = ax.plot(wl,est_mean,':')
    #here we want to show the magnitude of the pixel noise, so we show the
    # 95% likelihood interval of the normal distribution with the standard deviation.
    #(if we wanted a confidence interval for the mean, we'd use the std error and the t distribution)
    rgb_low, rgb_hi = stats.norm.interval(likelihood, loc=rgb_mean, scale=rgb_std)
    rgb_fills = []
    for i in range(rgb_low.shape[-1]):
        rgb_fills.append(ax.fill_between(wl, rgb_low[:,i], rgb_hi[:,i],alpha=0.5))
    est_low, est_hi = stats.norm.interval(likelihood,loc=est_mean, scale=est_std)
    est_fills = []
    for i in range(est_low.shape[-1]):
        est_fills.append(ax.fill_between(wl, est_low[:,i], est_hi[:,i],alpha=0.5))

    ax.legend([ tuple(rgb_lines) + tuple(rgb_fills), tuple(est_lines) + tuple(est_fills)],
              [f'Measurements (mean, {likelihood:.0%})',f'Estimates (mean, {likelihood:.0%})'],
              handler_map={tuple:plt.matplotlib.legend_handler.HandlerTuple(len(rgb_lines),pad=0)}
              )
    return ax

def plot_response(wl, resp_mean, resp_std, resp_n, confidence=0.95, logy=False, ax = None):
    """Plot estimated camera spectral response."""
    if ax is None: ax = plt.gca()
    ax.set_prop_cycle('color',['red','green','blue'])
    resp_lines = ax.plot(wl, resp_mean)
    #here we want to show the confidence interval around our estimate of the mean response
    resp_sem = resp_std/np.sqrt(resp_n) #standard error of mean
    resp_low, resp_hi = stats.t.interval(confidence,resp_n-1, loc=resp_mean, scale=resp_sem)
    resp_fills = []
    for i in range(resp_low.shape[-1]):
        resp_fills.append(ax.fill_between(wl, resp_low[:,i], resp_hi[:,i], alpha=0.5))
    if logy:
        thresh = 10**(np.ceil(np.log10(resp_mean.max()))-4) #4 orders of magnitude below max
        ax.set_yscale('symlog',linthresh=thresh,linscale=0.2,subs=np.arange(2,10))
    ax.legend([tuple(resp_lines) + tuple(resp_fills)],[f'Estimates (mean, {confidence:.0%})'],
              handler_map={tuple:plt.matplotlib.legend_handler.HandlerTuple(len(resp_lines),pad=0)})
    return ax

if __name__=='__main__':
    main()
