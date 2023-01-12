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
The ``util`` module contains several useful shared classes and methods for the
camera spectral calibration suite.
"""

from __future__ import annotations

import warnings, builtins
import numpy as np
from numpy.typing import ArrayLike
import h5py, yaml, re
from pathlib import Path
from typing import Literal

class QuietPrint:
    """Context manager class for suppressing ``print()``

    Replaces ``builtins.print`` with either `print_quiet` or `print_flush`
    inside the context. Nesting is supported::

        with QuietPrint(True):
            print('This will not print')
            with QuietPrint(False):
                print('This will print with flush=True')
            print('This will not print')
        print('This will print normally')

    Parameters
    ----------
    quiet : bool
        If ``True``, ``builtins.print`` is monkey-patched with `print_quiet`
        within the context, otherwise it uses `print_flush`.

    """

    print_original = builtins.print
    
    @staticmethod
    def print_quiet(*args, **kw):
        """Don't print."""
        pass
    
    @staticmethod
    def print_flush(*args, **kw):
        """Print with flush=True by default."""
        kw.setdefault('flush',True)
        QuietPrint.print_original(*args,**kw)

    def __init__(self, quiet=False):
        self.quiet = quiet
        self._print = None

    def __enter__(self):
        self._print = builtins.print
        if self.quiet:
            builtins.print = QuietPrint.print_quiet
        else:
            builtins.print = QuietPrint.print_flush

    def __exit__(self, etype, evalue, tb):
        builtins.print = self._print
        return False #we don't handle any exception

class Settings:
    """Camera Spectral Calibration task settings.

    This class loads image stack, file paths, and other data related to the spectral calibration workflow
    from a yaml file. The yaml file should be formatted as::

        camera: Camera model name
        lens: Lens model name
        settings: Camera & lens settings (e.g. ISO & Aperture)
        label: Short name for dataset
        ND: 0.0 # ND filter value (or 0.0 if unused)
        crop: [x, y, w, h] # raw->tiff crop settings
        dcraw: -w # dcraw command-line settings
        roi: [x, y, w, h] # region of interest (in cropped image) for collecting stats
        non_negative: True # set True to use non-negative regression for spectral response
        source_spectra: monochromator data directory
        image_dir: image file directory
        image_name: DSC_{:04}.NEF #image filename pattern. the {} will be replaced with the image number
        images:
            300: [1, 2, 3]
            305: [4, 5, 6]
            ...
            wavelength: [image numbers]
    
    The first line must be ``camera: ...`` for `check_valid` to accept the file, but the other items
    may appear in any order. Omitted items will be set to ``None``.
    
    Parameters
    ----------
    path
        Path to the settings yaml file.
    """
    
    def __init__(self, path):
        # get the absolute path to the settings file
        # the other paths will be relative to this
        path = Path(path).absolute()

        #parse the file
        with open(path,'r') as f:
            self.data = yaml.safe_load(f)
        
        # metadata
        # we use data.get instead of data[] because it returns None for missing items
        self.camera : str = self.data.get('camera') #: Camera name/description.
        self.lens : str = self.data.get('lens') #: Lens name/description.
        self.settings : str = self.data.get('settings') #: Camera & lens settings.
        self.label : str = self.data.get('label') #: Short name to identify the dataset.
        self.nd : float = self.data.get('ND') #: Neutral density filter value.
        self.crop_xywh : list[int] = self.data.get('crop') #: ``dcraw`` crop settings [x,y,w,h]
        self.dcraw_args : str = self.data.get('dcraw') #: ``dcraw`` extra arguments
        self.roi : tuple[slice,slice] = self.data.get('roi') #: region of interest slices for image statistics
        if self.roi is None:
            self.roi = slice(None),slice(None)
        else:
            x,y,w,h = self.roi
            self.roi = slice(y,y+h),slice(x,x+w)
        self.non_negative : bool = bool(self.data.get('non_negative')) #: True to use non-negative regression methods
        # Directory containing all of the images
        self._image_dir = Path(self.data.get('image_dir'))

        if not self._image_dir.is_absolute():
            # make image_dir relative to this file's path
            self._image_dir = path.parent / self._image_dir
        self.image_dir : str = str(self._image_dir) #: Path to image files

        self.image_name : str = self.data.get('image_name') #: Image filename pattern
        self.images : dict[float, list[int]] = self.data.get('images') #: mapping wavelength to lists of image numbers
        
        # File for image stats, in the image dir
        self.samples_file : str = str(self._image_dir.with_suffix('.h5')) #: path to image samples file

        spectra_path = path.parent / self.data.get('source_spectra')
        self.spectra_dir : str = str(spectra_path) #: illuminant spectra data directory
        self.spectra_file : str = str(spectra_path.with_suffix('.h5')) #: illuminant spectra data file
        
        stem = path.with_suffix('')
        self.response_file : str = f'{stem} response.h5' #: spectral response data file path
        self.plots_file : str = f'{stem} plots.pdf' #: spectral response plots file path
        self.excel_file : str = f'{stem} response.xlsx' #: spectral response excel file path
    
    @staticmethod
    def check_valid(path : str) -> bool:
        """Check for a (possibly) valid settings yaml file, without loading the whole thing."""
        with open(path, 'r') as f:
            #search for the first non-comment line
            line = None
            while not line or line.strip().startswith('#'):
                line = f.readline()
        try:
            yaml.safe_load(line)['camera']
            return True
        except:
            return False    

    def image_file(self, i : int, suffix : str|None =None) -> str:
        """Get path (as string) to i'th image file."""
        img_path = self._image_dir / self.image_name.format(i)
        if suffix: img_path = img_path.with_suffix(suffix)
        return str(img_path)

class H5Storage:
    """Base class for data storage backed by H5 files.
    
    This class provides several methods that can be used by derived classes
    to semi-automatically load and save data from H5 files.

    E.g. in the derived class::

        class Foo(H5Storage):
            _h5members = [('x'),('y')]
    
            def __init__(self):
                self.x = numpy.arange(100)

            def save(self):
                with self._open('file.h5','w') as f:
                    self._save(f)

    The ``members`` argument to `_load` and `_save` is either ``None`` or an iterable where
    each item is ``(name, h5_name, attrs, dims, func)``. 
    If the ``members`` argument is ``None``, it defaults to ``self._h5members``.

    ``h5_name``, ``attrs``, ``dims``, and ``func`` are optional, but must be specified in order.
    If not specified, they will default to ``None``.
    if ``h5_name`` is ``None`` it will default to ``name``. If both
    ``name`` and ``h5_name`` are ``None``, ``h5_name`` will be set to ``'/'``, representing
    the file's root (e.g. to set file attributes).
    If ``func`` is ``None`` it will default to ``numpy.array``.
    
    Data will be loaded as ``self.name = func(file[h5_name])`` and stored as
    ``file.create_dataset(h5_name,data=self.name)``. If the dataset already exists, then it
    will be overwritten (requiring that the shape matches!), if the same data was previously
    written to a different dataset, then a hardlink will be created rather than copy the data.

    ``attrs`` is either ``None`` or an iterable of ``(attr_name, h5_attr_name)``
    Attributes are loaded as ``self.attr_name = file[h5_name].attrs[h5_attr_name]``
    and stored as ``file[h5_name].attrs[h5_attr_name] = self.attr_name``.
    
    To access file or group attributes, set ``name`` to ``None`` and ``h5_name`` to either
    the group name or ``'/'`` to access file attributes.

    ``dims`` is either ``None`` or an iterable of ``h5_scale_name`` or ``(dim_label, h5_scale_name)``.
    If only ``h5_scale_name`` is given, it will also be used as the label. After all datasets have
    been written to the file, all datasets referred to by ``h5_scale_name`` will have ``make_scale()``
    called on them and then attached as dimension scales to the associated datasets.
    If the ``h5_scale_name`` dataset does not exist a `H5Storage.Warning` will be issued.
    """
    class Warning(UserWarning):
        """Warning class derived from ``UserWarning``, emitted by `_load` and `_save`."""
        pass

    def _warn(self, message : str):
        warnings.warn(self.Warning(message),stacklevel=2)

    class _Item:
        def __init__(self,name : str, h5_name : str = None, attrs=(),dims=(),func=np.array):
            if h5_name is None:
                if name is not None: h5_name = name # name,None -> name,name
                else: h5_name = '/' # None,None -> None,'/' for the file root
            self.name, self.h5_name = name, h5_name
            self.attrs, self.dims, self.func = attrs, dims, func
    
    def _item(self,*args,**kw):    
        if isinstance(args[0],(tuple,list)):
            return self._Item(*args[0])
        elif isinstance(args[0], self._Item):
            return args[0]
        return self._Item(*args,**kw)

    def _open(self, path : str, mode : Literal['r','r+','w','w-','x','a']) -> h5py.File:
        """Open and return an h5py File.
        
        :meta public:
        """
        return h5py.File(path,mode)

    def _load(self, file : h5py.File, members : list[tuple[str,...]]|None = None, warn_missing : bool =False):
        """Load the specified members from file.
        
        :meta public:

        Parameters
        ----------
        file 
            an open ``h5py.File``.
        members
            list of class members to load.
        warn_missing
            if ``True``, issue a warning for missing items rather than raise an exception.
        """
        if members is None:
            members = self._h5members

        loaded = dict()

        for item in members:
            #sanitize args:
            item = self._item(item)
            
            #access object
            h5_obj = file.get(item.h5_name) #returns None if the object isn't found
            
            if item.name:
                if item.name in loaded:
                    self._warn(f'Skipping `{item.name}`, already loaded from `{loaded[item.name]}`.')
                else:
                    #load data
                    if h5_obj:
                        setattr(self, item.name, item.func(h5_obj))
                        loaded[item.name] = item.h5_name
                    elif warn_missing:
                        setattr(self, item.name, None)
                        self._warn(f'H5 item not found: `{item.h5_name}`. Setting `{item.name} = None`.')
                    else:
                        raise LookupError(f'H5 item not found: `{item.h5_name}`.')

            for attr in item.attrs:
                attr = self._item(attr)
                if attr.name in loaded:
                    self._warn(f'Skipping `{attr.name}`, already loaded from `{loaded[attr.name]}`.')
                    continue
                #load attributes
                h5_attr_obj = None
                if h5_obj:
                    h5_attr_obj = h5_obj.attrs.get(attr.h5_name)
                if h5_attr_obj:
                    setattr(self, attr.name, h5_attr_obj)
                    loaded[attr.name] = f'{item.h5_name}:{attr.h5_name}'
                elif warn_missing:
                    setattr(self, attr.name, None)
                    self._warn(f'H5 item attribute not found: `{item.h5_name}:{attr.h5_name}`. Settings `{item.name} = None`.')
                else:
                    raise LookupError(f'H5 item attribute not found: `{item.h5_name}:{attr.h5_name}`.')

    def _save(self, file : h5py.File, members : list[tuple[str,...]]|None = None):
        """Save the specified members to file.
        
        :meta public:

        Parameters
        ----------
        file
            an open ``h5py.File`` file.
        members
            list of class members to store.
        """
        if members is None:
            members = self._h5members
        dim_scales = set()
        dim_attachments = []
        stored = dict()
        for item in members:
            #sanitize args
            item = self._item(item)
            
            h5_obj = file.get(item.h5_name) #returns None if the object isn't found

            if item.name:
                #store data
                py_obj = getattr(self,item.name)
                if not h5_obj and item.name in stored:
                    #we've written this dataset to the file already
                    #make a hard-link to the previously stored dataset
                    file[item.h5_name] = file[stored[item.name]]
                    h5_obj = file[item.h5_name]
                elif h5_obj:
                    if item.name in stored:
                        self._warn(f'H5 item `{item.h5_name}` already exists. Overwriting with `{item.name}`')
                    #the target dataset is in the file already,
                    #try to overwite the existing dataset
                    h5_obj = file[item.h5_name]
                    h5_obj[:] = py_obj
                    stored[item.name] = item.h5_name
                else:
                    #make new dataset
                    h5_obj = file.create_dataset(item.h5_name, data=py_obj)
                    stored[item.name] = item.h5_name
                #set dimension metadata
                for d, dim_info in zip(h5_obj.dims, item.dims):
                    if dim_info is None: continue
                    if isinstance(dim_info,(tuple,list)):
                        dim_label, dim_scale = dim_info
                    else:
                        dim_label, dim_scale = dim_info, dim_info
                    if dim_label:
                        d.label = dim_label
                    if dim_scale:
                        #deal with these later so that we don't have to worry about creation order
                        dim_scales.add(dim_scale)
                        dim_attachments.append((item.h5_name, d, dim_scale))
            elif not h5_obj:
                #create group
                h5_obj = file.create_group(item.h5_name)
            
            for attr in item.attrs:
                attr = self._item(attr)
                #store attributes
                py_obj = getattr(self, attr.name)
                h5_obj.attrs[attr.h5_name] = py_obj

        for ds in dim_scales:
            if ds in file:
                file[ds].make_scale()
        for h5_name, d, ds in dim_attachments:
            if ds in file:
                d.attach_scale(file[ds])
            else:
                self._warn(f'Cannot attach dataset `{ds}` to dimension scale `{d.label}` of `{h5_name}`: does not exist!')
