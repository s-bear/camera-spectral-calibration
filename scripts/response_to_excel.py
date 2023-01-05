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
Collate camera spectral response data into Excel format.
"""

from __future__ import annotations

import argparse
from util import QuietPrint
from camera_response import CameraResponse
from pathlib import Path
import numpy as np
import h5py
import xlsxwriter as xl

def _parse_args(args,**kw) -> argparse.Namespace:
    """Parse command-line arguments for `main`."""
    p = argparse.ArgumentParser()
    p.add_argument('-o','--output',help='output Excel file')
    p.add_argument('--quiet',action='store_true',help='suppress messages')
    p.add_argument('input',metavar='FILE',type=Path, nargs='+',help='Input h5 files')
    return p.parse_args(args,argparse.Namespace(**kw))

def main(args=None,**kw):
    """Convert spectral response h5 file to Excel format.

    If called without parameters, parses arguments from the command line.
    Arguments may be passed as a list of strings or as keyword arguments. E.g.::

        main(['-o', 'results.xlsx','response1.h5','response2.h5'])
        # or
        main(output='results.xlsx',input=['response1.h5','response2.h5'])

    Parameters
    ----------
    output : ``-o``, ``--output``, Path
        output Excel file
    input : list of Paths
        List of spectral response files to process. Each is converted to a single worksheet of the output file.
    quiet : ``--quiet``, bool
        suppress status messages
    """
    args = _parse_args(args,**kw)
    with QuietPrint(args.quiet):

        response_files = args.input
        xlpath = args.output
        
        with xl.Workbook(xlpath) as wb:
            for rfile in response_files:
                print(f'Loading {rfile}... ',end='')
                try:
                    data = CameraResponse(rfile)
                except Exception as err:
                    print(f'ERROR: {err}')
                    continue
                print('OK')

                print('Writing worksheet... ',end='')
                try:
                    ws = wb.add_worksheet(data.label)
                except (xl.exceptions.DuplicateWorksheetName, xl.exceptions.InvalidWorksheetName):
                    ws = wb.add_worksheet()
                ws_name = ws.get_name()
                ws.write_row('A1',['Camera:',data.camera])
                ws.write_row('A2',['Lens:',data.lens])
                ws.write_row('A3',['Settings:',data.settings])
                ws.write_row('A4',['Units:','wavelength:',data.wl_units,'response:',data.units])
                num_channels = data.mean.shape[1]
                if num_channels == 3:
                    channel_names = ['Red','Green','Blue']
                else:
                    channel_names = [f'Channel {i+1}' for i in range(num_channels)]
                ws.write_row('A5',['Wavelength'] + channel_names + [f'std({c})' for c in channel_names])
                ws.write_column('A6',data.wl)
                r,c = xl.utility.xl_cell_to_rowcol('B6') #get row, col index numbers
                r_end = r + data.wl.shape[0]-1
                for i in range(num_channels):
                    ws.write_column(r,c,data.mean[:,i])
                    c += 1
                for i in range(num_channels):
                    ws.write_column(r,c,data.std[:,i])
                    c += 1
                chart = wb.add_chart({'type':'scatter','subtype':'straight'})
                for i,c in enumerate(channel_names):
                    chart.add_series({'name':[ws_name,r-1,i+1],
                                     'categories':[ws_name,r,0,r_end,0],
                                     'values':[ws_name,r,i+1,r_end,i+1]})
                chart.set_title({'name':f'{data.camera}, {data.lens}'})
                chart.set_x_axis({'name':f'Wavelength {data.wl_units}','min':data.wl[0],'max':data.wl[-1]})
                chart.set_y_axis({'name':'Normalized spectral response','min':0})
                ws.insert_chart(1, 10, chart)

                print('DONE')


if __name__ == '__main__':
    main()