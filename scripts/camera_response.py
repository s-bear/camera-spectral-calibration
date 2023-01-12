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
This script estimates a camera's spectral sensitivities from a set of samples
(photos of a spectral light source) and spectral irradiance measurements of
the light source. Results are saved in an H5 file.
"""

from __future__ import annotations

import argparse
import numpy as np
from numba import jit

from numpy.typing import NDArray
from typing import Literal, Callable

from util import H5Storage, QuietPrint
from monochromator import MonochromatorSpectra
from image_stats import CameraSamples

import scipy.stats as stats
from scipy.optimize import minimize, minimize_scalar

from scipy.signal import convolve, firwin
from sklearn import linear_model, metrics

#ignore warnings from minimize_scalar
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

def _parse_args(args, **kw):
    """Parse and return command-line arguments for `main`."""
    p = argparse.ArgumentParser()
    p.add_argument('-m','--monochromator',help='monochromator spectral irradiance file')
    p.add_argument('-i','--input',help='input samples file')
    p.add_argument('-o','--output',help='output response file')
    p.add_argument('--bootstrap',default=100,help='Number of bootstrapping iterations for estimating response noise.')
    p.add_argument('--method',default='ridge-smooth',choices=['ols','ridge','ridge-smooth'],help='Choose regression method')
    p.add_argument('-nn','--non-negative',action='store_true',help='use non-negative constraint in regression')
    p.add_argument('-r2','--target-r2',default=0.999,type=float,help='target coefficient of determination. Controls bias-variance trade-off and helps prevent over-fitting of ridge regression.')
    p.add_argument('-b','--bandwidth',help='ridge-smooth bandwidth. default will be the max of monochromator irradiance bandwidth or input sample spacing')
    p.add_argument('--quiet',action='store_true',help='suppress messages')
    return p.parse_args(args, argparse.Namespace(**kw))

def main(args=None,**kw):
    """Compute camera spectral sensitivity from spectral image stack samples.

    If called without parameters, parses arguments from the command line.
    Arguments may be passed as a list of strings or as keyword arguments. E.g.::

        main(['-i','samples.h5','--quiet'])
        # or
        main(input='samples.h5',quiet=True)

    Parameters
    ----------
    monochromator :``-m``, ``--monochromator``, Path or MonochromatorSpectra
        monochromator spectral irradiance file.
    input : ``-i``, ``--input``, Path or CameraSamples
        input samples h5 file
    output : ``-o``, ``--output``, Path or CameraResponse
        output response h5 file
    bootstrap : ``--bootstrap``, int, default 100
        Number of bootstrapping iterations
    method : ``--method``, str, one of ['ols', 'ridge', 'ridge-smooth']
        Regression method
    non_negative: ``--non-negative``, bool, default False
        If True, use non-negative constrained regression.
    target_r2: ``-r2``, ``--target-r2``, float, default 0.999
        Target coefficient of determination. Controls bias-variance tradeoff (alpha) of ridge regression, helps prevent overfitting.
    bandwidth: ``-b``, ``--bandwidth``, float
        Bandwidth for ``ridge-smooth``. Default will be max of monochromator irradiance bandwidth or input sample spacing.
    quiet : ``--quiet``, bool
        suppress status messages
    """
    args = _parse_args(args,**kw)
    with QuietPrint(args.quiet):
        #Load data, compute spectral response, save results.
        if isinstance(args.monochromator, MonochromatorSpectra):
            irrad = args.monochromator
        else:
            print(f'Loading {args.monochromator}... ',end='')
            irrad = MonochromatorSpectra(args.monochromator)
            print('DONE')

        if isinstance(args.input, CameraSamples):
            rgb = args.input
        else:
            print(f'Loading {args.input}... ',end='')
            rgb = CameraSamples(args.input)
            print('DONE')

        print(f'Estimating spectral response ({args.method})... ',end='')
        #normalize inputs
        irrad.normalize('total')
        rgb.normalize()

        #we need the intersection of the monochromator's nominal wavelengths and the samples'
        nom_wl, irrad_idx, rgb_idx = np.intersect1d(irrad.nom_wl, rgb.nom_wl, return_indices=True)
        #crop the data down to just those samples:
        irrad.mean = irrad.mean[irrad_idx]
        irrad.std = irrad.std[irrad_idx]
        irrad.nom_wl = nom_wl

        rgb.mean = rgb.mean[rgb_idx]
        rgb.std = rgb.std[rgb_idx]
        rgb.nom_wl = nom_wl
        
        if args.method == 'ols':
            alpha = 0.0
            print('bootstrapping... ',end='')
            res = bootstrap(ols, irrad.mean, irrad.std, rgb.mean, rgb.std, args.bootstrap,non_negative=args.non_negative)

        elif args.method == 'ridge':
            print('alpha... ',end='')
            #we want to find the regression with the lowest variance that still satisfies
            # our accuracy requirements (r2 >= r2_min)
            #first we want to find the alpha that gives us our minimum accuracy
            kw={'return_estimate':True,'non_negative':args.non_negative} #args to pass to regressions
            f = lambda alpha: (args.target_r2 - metrics.r2_score(rgb.mean, ridge(irrad.mean, rgb.mean, alpha, **kw)[1]))**2
            m = minimize_scalar(f, bounds=(1e-8,1e8),method='bounded')
            alpha = m.x
            kw['alpha'] = alpha
            print('bootstrapping... ',end='')
            res = bootstrap(ridge, irrad.mean, irrad.std, rgb.mean, rgb.std, args.bootstrap, **kw)

        if args.method == 'ridge-smooth':
            print('bandwidth... ',end='')
            #determine the monochromator irradiance bandwidth -- which acts as a point-spread function
            irrad_bw = np.median(fwhm(irrad.wl, irrad.mean)) #TODO: we assume there isn't much variation
            #determine the sample spacing
            irrad_peak_wl = find_peaks(irrad.wl, irrad.mean)
            cam_bw = 2*np.median(np.diff(irrad_peak_wl)) #TODO: we assume there isn't much variation

            if args.bandwidth is None:
                # the actual bandwidth is the larger
                bw = np.max((irrad_bw, cam_bw))
            else:
                bw = args.bandwidth

            #Smoothness Constraint Filter
            #  This filter is used to *penalize* the regression result
            #  so to enforce smoothness, we want a high-pass filter
            filt_hipass = firwin(15, 1/bw,fs=1/np.median(np.diff(irrad.wl)),pass_zero=False)


            print('alpha... ',end='')
            kw={'beta':1.0,'filt':filt_hipass,'return_estimate':True,'non_negative':args.non_negative}
            f = lambda alpha: (args.target_r2 - metrics.r2_score(rgb.mean, ridge_smooth(irrad.mean, rgb.mean, alpha, **kw)[1]))**2
            m = minimize_scalar(f,bounds=(1e-8,1e8),method='bounded')
            alpha = m.x
            kw['alpha'] = alpha
            print('bootstrapping... ',end='')    
            res = bootstrap(ridge_smooth, irrad.mean, irrad.std, rgb.mean, rgb.std, args.bootstrap, **kw)
        
        resp_mean, resp_std, est_rgb_mean, est_rgb_std, R2 = res
        #denormalize the response
        resp_scale = rgb.scale / irrad.scale
        resp_mean *= resp_scale
        resp_std *= resp_scale
        resp_scale = 1.0
        resp_units = f'({rgb.units})/({irrad.units})'

        #denormalize the samples and estimate before saving
        est_rgb_mean *= rgb.scale
        est_rgb_std *= rgb.scale
        rgb.denormalize()
        irrad.denormalize()
        print('DONE')

        if isinstance(args.output, CameraResponse):
            resp = args.output
        else:
            resp = CameraResponse()
        resp.camera = rgb.camera
        resp.lens = rgb.lens
        resp.settings = rgb.settings
        resp.label = rgb.label
        resp.wl = irrad.wl
        resp.wl_units = irrad.wl_units
        resp.mean = resp_mean
        resp.std = resp_std
        resp.units = resp_units
        resp.scale = resp_scale
        resp.n = args.bootstrap
        resp.R2 = R2
        resp.alpha = alpha
        resp.method = args.method
        resp.samples_wl = rgb.nom_wl
        resp.samples_units = rgb.units
        resp.samples_n = rgb.n
        resp.samples_mean = rgb.mean
        resp.samples_std = rgb.std
        resp.estimates_mean = est_rgb_mean
        resp.estimates_std = est_rgb_std
        resp.estimates_n = args.bootstrap

        if not isinstance(args.output, CameraResponse):
            print('Saving results... ',end='')
            resp.save(args.output)
            print('DONE')

class CameraResponse(H5Storage):
    """Load and save camera spectral response data.

    Parameters
    ----------
    path
        ``None`` or path to h5 file to load.
    """
    
    _h5members = [
        (None,None,['camera','lens','settings','label']),
        ('wl','wavelength',[('wl_units','units')]),
        ('mean','response_mean',['units','scale','n','R2','alpha','method'],['wavelength',('channel',None)]),
        ('std','response_std',['units','scale','n']),
        ('samples_wl','samples/nominal_wavelength',[('wl_units','units')]),
        ('samples_mean','samples/mean',[('samples_units','units'),('samples_n','n')],[('nominal_wavelength','samples/nominal_wavelength'),('channel',None)]),
        ('samples_std','samples/std',[('samples_units','units'),('samples_n','n')],[('nominal_wavelength','samples/nominal_wavelength'),('channel',None)]),
        ('estimates_mean','samples/estimate_mean',[('samples_units','units'),('estimates_n','n')],[('nominal_wavelength','samples/nominal_wavelength'),('channel',None)]),
        ('estimates_std','samples/estimate_std',[('samples_units','units'),('estimates_n','n')],[('nominal_wavelength','samples/nominal_wavelength'),('channel',None)]),
    ]

    def __init__(self, path=None):
        self.camera : str = None #: Camera name/description.
        self.lens : str = None #: Lens name/description.
        self.settings : str = None #: Camera and lens settings used.
        self.label : str = None #: Short name to identify dataset.
        self.wl : NDArray = None #: Spectral response wavelength. Shape (N).
        self.wl_units : str = None #: Units of `wl`
        self.mean : NDArray = None #: Mean spectral response. Shape (N,C)
        self.std : NDArray = None #: Standard deviation of spectral response. Shape (N,C)
        self.units : str = None #: Units of `mean` and `std`.
        self.scale : float = 1.0 #: Scale applied to `mean` and `std`.
        self.n : int = None #: number of bootstrap iterations used to compute `mean` and `std`.
        self.R2 : float = None #: Coefficient of determination of `estimates_mean` relative to `samples_mean`.
        self.alpha : float = None #: Bias-variance tradeoff coefficient of the regression method.
        self.method : str = None #: Regression method used to compute `mean` and `std`
        self.samples_wl : NDArray = None #: Nominal wavelength of `samples_mean`, `samples_std`, `estimates_mean`, and `estimates_std`. Shape (M).
        self.samples_units : str = None #: Units of `samples_mean`, `samples_std`, `estimates_mean`, and `estimates_std`.
        self.samples_n : int = None #: Number of pixels averaged for each element of `samples_mean` and `samples_std`.
        self.samples_mean : NDArray = None #: Mean pixel value for each sample image.
        self.samples_std : NDArray = None #: Pixel standard deviation for each sample image.
        self.estimates_mean : NDArray = None #: Mean of estimated pixel values across bootstrapping iterations
        self.estimates_std : NDArray = None #: Std of estimated pixel values across bootstrapping iterations
        self.estimates_n : int = None #: number of bootstrap iterations for `estimates_mean` and `estimates_std`.
        if path:
            self.load(path)
    
    def normalize(self, mode : Literal['peak','total'] = 'peak'):
        """Normalize peak or total response to 1. ``mode`` is ``'peak'`` or ``'total'``"""
        mode = str(mode).lower()
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

    def load(self,path : Path):
        """Load data from ``path``."""
        with self._open(path,'r') as f:
            self._load(f, warn_missing=True)

    def save(self,path : Path):
        """Save data to ``path``. Overwrites existing files."""
        with self._open(path,'w') as f:
            self._save(f)

### Signal analysis ###

def fwhm(x : ArrayLike, y : ArrayLike, axis : int = -1) -> np.ndarray:
    """Measure FWHM along the given axis.

    Measure the full-width at half-maximum (FWHM) along the given axis.
    Measures the difference in ``x`` where ``y`` is half of its maximum along ``axis``

    Parameters
    ----------
    x
        Data coordinate, must have same shape as ``y``.
    y
        Data value, must have same shape as ``x``.
    axis
        Dimension to measure width along.

    Returns
    -------
    width : ndarray
        Shape as ``x`` and ``y`` but less one dimension for ``axis``.
    """
    x,y = np.broadcast_arrays(x,y)
    #vectorize x
    x = np.moveaxis(x, axis, -1)
    x_shape = x.shape
    x = x.reshape((-1, x.shape[-1]))
    #vectorize y
    y = np.moveaxis(y, axis, -1)
    y = y.reshape((-1, y.shape[-1]))
    #normalize and find coordinates of high points
    y_norm = y/y.max(1,keepdims=True)
    r,c = np.nonzero(y_norm >= 0.5)
    #find rising and falling edge indexes
    i_rise = np.unique(r, True)[1] #index of rising edge, into r,c
    i_fall = r.size-1-np.unique(-r[::-1],True)[1][::-1] #index of falling edge, into r,c
    r_rise, c_rise_0, c_rise_1 = r[i_rise], c[i_rise]-1, c[i_rise] #index of rising edge, into x,y (just after)
    r_fall, c_fall_0, c_fall_1 = r[i_fall], c[i_fall], c[i_fall]+1 #index of falling edge, into x,y (just before)
    #find rising and falling edge x value
    f = lambda x0,y0,x1,y1: x0 + (x1-x0)*(0.5 - y0)/(y1-y0) #find halfway point
    x_rise = f(x[r_rise, c_rise_0], y_norm[r_rise, c_rise_0], x[r_rise, c_rise_1], y_norm[r_rise, c_rise_1])
    x_fall = f(x[r_fall, c_fall_0], y_norm[r_fall, c_fall_0], x[r_fall, c_fall_1], y_norm[r_fall, c_fall_1])
    #caluclate width
    w = x_fall - x_rise
    #reshape and return
    return w.reshape(x_shape[:-1])

def find_peaks(x : ArrayLike, y : ArrayLike, axis : int = -1, threshold : float = 0.75) -> np.ndarray:
    """Find ``x`` of peak ``y`` along an axis. Only works for monomodal data.

    Finds the centroid of ``y[y > threshold*np.nanmax(y, axis)]``
    
    Parameters
    ----------
    x
        data coordinate. Must have same shape as ``y``.
    y
        Data value. Must have same shape as ``x``
    axis
        Dimension to integrate along for centroid
    threshold
        relative threshold of ``y`` to consider for peaks.

    Returns
    -------
    peaks : ndarray
        ``x`` coordinate of peaks in ``y`` along ``axis``. Shape as ``x`` but less one dimension for ``axis``.
    """
    y_masked = y*(y >= threshold*np.nanmax(y,axis,keepdims=True))
    peak_x = np.trapz(x*y_masked, x, axis=axis)/np.trapz(y_masked,x,axis=axis)
    return peak_x

### Regression functions ###

def ols(X : ArrayLike, y : ArrayLike, return_estimate : bool = False, non_negative : bool = False) -> np.ndarray | Tuple[np.ndarray,np.ndarray]:
    """Ordinary least squares regression.
    
    Solves :math:`\\min_w||X w - y||_2^2`.

    Parameters
    ---------
    X : array_like[(n_samples, n_features)]
        Independent variables, input, feature vectors, or training data.
    y : array_like[(n_samples, n_targets)]
        Dependent variables, output, or target values.
    return_estimate
        if ``True``, also return ``estimate = X @ w``.
    non_negative
        if ``True``, constrain ``w`` to non-negative values.
    
    Returns
    -------
    w : ndarray[(n_features, n_targets)]
        Feature weights
    estimate : ndarray[(n_samples, n_targets)]
        Estimate of ``y`` for computing regression quality metrics such as R2. ``estimate = X @ w``
    """
    model = linear_model.LinearRegression(fit_intercept=False,copy_X=False,positive=non_negative)
    model.fit(X,y)
    res = model.predict(np.eye(X.shape[-1]))
    if return_estimate:
        res = res, model.predict(X)
    return res

@jit
def _ridge_loss_general(X,y,w,M):
    """Generalize ridge regression loss and gradient."""
    w = w.reshape((X.shape[1], y.shape[1])) #unvectorize
    residuals = X @ w - y # Note: negative of the usual form -- saves a negation op in the gradient
    Mw = M @ w
    loss = 0.5*residuals.T @ residuals + 0.5*Mw.T @ Mw
    grad = X.T @ residuals + M.T @ Mw
    return np.sum(loss), grad.reshape((-1)) #vectorize

@jit
def _ridge_solve_pinv(X,y,M):
    return np.linalg.pinv(X.T @ X + M.T @ M) @ X.T @ y

def ridge_general(X : ArrayLike, y : ArrayLike, M : ArrayLike, return_estimate:bool=False, non_negative:bool=False) -> np.ndarray | Tuple[np.ndarray,np.ndarray]:
    """Generalized ridge regression with regularization (Tikhonov) matrix M.
    
    Solves :math:`\\min_w||X w - y||_2^2 + ||M w||_2^2` using :math:`w = (X^\\trans X + M^\\trans M)^{-1} X^\\trans y`

    Non-negative solutions are found by iterative minimization.

    Parameters
    ---------
    X : array_like[(n_samples, n_features)]
        Independent variables, input, feature vectors, or training data.
    y : array_like[(n_samples, n_targets)]
        Dependent variables, output, or target values.
    M : array_like[(n_regularizer, n_features)]
        Regularization (Tikhonov) matrix.
    return_estimate
        if ``True``, also return ``estimate = X @ w``.
    non_negative
        if ``True``, constrain ``w`` to non-negative values.
    
    Returns
    -------
    w : ndarray[(n_features, n_targets)]
        Feature weights
    estimate : ndarray[(n_samples, n_targets)]
        Estimate of ``y`` for computing regression quality metrics such as R2. ``estimate = X @ w``
    """
    if non_negative:
        f = lambda w: _ridge_loss_general(X, y, w, M)
        w0 = np.zeros((X.shape[1], y.shape[1])).reshape((-1))
        bounds = [(0, np.inf)]*w0.size
        m = minimize(f, w0, bounds=bounds, jac=True) #TODO: deal with minimize results
        m.x = m.x.reshape((X.shape[1], y.shape[1]))
        res = m.x
    else:
        res = _ridge_solve_pinv(X,y,M)
    if return_estimate:
        res = res, X @ res
    return res

def ridge(X : ArrayLike, y : ArrayLike, alpha : float = 1.0, return_estimate : bool = False, non_negative:bool=False)  -> np.ndarray | Tuple[np.ndarray,np.ndarray]:
    """Ridge regression.
    
    Solves :math:`\\min_w||X w - y||_2^2 + \\alpha||w||_2^2` using `ridge_general`.

    Parameters
    ---------
    X : array_like[(n_samples, n_features)]
        Independent variables, input, feature vectors, or training data.
    y : array_like[(n_samples, n_targets)]
        Dependent variables, output, or target values.
    alpha : float
        Regularization strength.
    return_estimate
        if ``True``, also return ``estimate = X @ w``.
    non_negative
        if ``True``, constrain ``w`` to non-negative values.
    
    Returns
    -------
    w : ndarray[(n_features, n_targets)]
        Feature weights
    estimate : ndarray[(n_samples, n_targets)]
        Estimate of ``y`` for computing regression quality metrics such as R2. ``estimate = X @ w``
    """
    M = np.sqrt(alpha)*np.eye(X.shape[1])
    return ridge_general(X,y,M,return_estimate,non_negative)

def ridge_smooth(X : ArrayLike, y : ArrayLike, alpha : float =1.0, beta : float =1.0, filt : ArrayLike=[1,-1], return_estimate:bool=False, non_negative:bool=False) -> np.ndarray | Tuple[np.ndarray,np.ndarray]:
    """Ridge regression with a smoothness constraint.
    
    Solves :math:`\\min_w||X w - y||_2^2 + \\alpha||w||_2^2 + \\beta||h*w||_2^2` using `ridge_general`.
    In this case, :math:`h*w` represents the convolution of ``filt`` with ``w`` along the ``n_features`` axis.

    Conceptually, applying a high-pass filter to :math:`w` as a regularization term will penalize
    solutions with high-frequency content, resulting in a smooth solution.

    Parameters
    ---------
    X : array_like[(n_samples, n_features)]
        Independent variables, input, feature vectors, or training data.
    y : array_like[(n_samples, n_targets)]
        Dependent variables, output, or target values.
    alpha : float
        Regularization strength.
    beta : float
        Filter strength
    filt : array_like[(n_taps,)]
        Finite impulse response filter. Should be a high-pass filter to ensure a smooth filter.
    return_estimate
        if ``True``, also return ``estimate = X @ w``.
    non_negative
        if ``True``, constrain ``w`` to non-negative values.
    
    Returns
    -------
    w : ndarray[(n_features, n_targets)]
        Feature weights
    estimate : ndarray[(n_samples, n_targets)]
        Estimate of ``y`` for computing regression quality metrics such as R2. ``estimate = X @ w``
    """
    #build the penalty matrix for general ridge regression
    filt = np.asarray(filt)
    D = convolve(np.eye(X.shape[1]), filt[None], 'same') #each row is the filter, centered on the diagonal
    M = np.sqrt(alpha)*np.eye(X.shape[1]) + np.sqrt(beta)*D #the penalty matrix
    return ridge_general(X, y, M, return_estimate, non_negative)

def bootstrap(R : Callable[[ArrayLike,ArrayLike,...],Tuple[np.ndarray,np.ndarray]], X_mean : ArrayLike, X_std : ArrayLike, y_mean : ArrayLike, y_std : ArrayLike, n:int=100, rng:np.random.Generator|None=None, **kw) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    """Perform a bootstrapped regression
    
    The regression ``R(X,y,return_estimate=True,**kw)`` will be perfomed ``n`` times, 
    each time sampling ``X = rng.normal(X_mean, X_std)`` and ``y = rng.normal(y_mean, y_std)``.

    Parameters
    ----------
    R
        The regression method, such as `ols`, `ridge`, `ridge_smooth`. It must accept the keyword argument 'return_estimate'.
    X_mean : array_like[(n_samples, n_features)]
        Mean of ``X`` distribution.
    X_std : array_like[(n_samples, n_features)]
        Standard deviation of ``X`` distribution.
    y_mean : array_like[(n_samples, n_targets)]
        Mean of ``y`` distribution.
    y_std : array_like[(n_samples, n_targets)]
        Standard deviation of ``y`` distribution.
    n
        Number of bootstrap iterations
    rng
        a numpy random number generator. Defaults to ``numpy.random.default_rng()``.
    **kw
        Any extra keyword arguments are passed to ``R``

    Returns
    -------
    w_mean : ndarray[(n_features, n_targets)]
        Mean of the feature weights ``w`` over all bootstrap iterations
    w_std : ndarray[(n_features, n_targets)]
        Standard deviation of the feature weights ``w`` over all bootstrap iterations
    estimate_mean : ndarray[(n_samples, n_targets)]
        Mean of ``estimate = X @ w`` over all bootstrap iterations
    estimate_std : ndarray[(n_samples, n_targets)]
        Standard deviation of ``estimate`` over all bootstrap iterations.
    R2
        Coefficient of variation between ``y_mean`` and ``estimate_mean``
    """
    if n > 1:
        if rng is None: rng = np.random.default_rng()
        X = rng.normal(X_mean[None],X_std[None],(n,)+X_mean.shape)
        y = rng.normal(y_mean[None],y_std[None],(n,)+y_mean.shape)
    else:
        n = 1
        X,y = X_mean,y_mean
    #perform the regression for each replicate
    weights, estimates = None,None
    kw = kw.copy()
    kw['return_estimate'] = True
    for i in range(n):
        w, y_est = R(X[i],y[i],**kw)
        if weights is None: weights = np.empty((n,)+w.shape)
        if estimates is None: estimates = np.empty((n,)+y_est.shape)
        weights[i] = w
        estimates[i] = y_est
    
    w_mean, w_std = np.mean(weights,0), np.std(weights,0)
    est_mean, est_std = np.mean(estimates,0), np.std(estimates,0)
    
    R2 = metrics.r2_score(y_mean, est_mean) # coefficient of variation
    return w_mean, w_std, est_mean, est_std, R2

if __name__ == '__main__':
    main()
