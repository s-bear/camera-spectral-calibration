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
Run camera spectral response workflow.

Run this script by executing "scons" (Windows) or "scons-3" (Linux) in this directory.

To install dependencies for this suite of scripts (see environment.yml):
    conda env create
(Get conda from https://anaconda.org/ or https://docs.conda.io/en/latest/miniconda.html)

You must activate the environment in your terminal before running scons
    conda activate camera-cal

This script:
1. Searches for yml files describing camera calibration datasets
2. Uses dcraw to convert and crop the photos listed in the yml files to TIFF
3. Calculates the mean and std (less the fixed pattern noise) of each TIFF 
   using 'collate_image_stats.py'
4. Collates monochromator output spectral irradiances using 'collate_spectra.py'
5. Estimates the camera's spectral response using 'camera_response.py'
6. Collates all of the response files into an Excel spreadsheet with 'collate_responses.py'.
"""

import os
from pathlib import Path
from scripts.util import Settings

#tell SCons to print which file we're working on
Progress('$TARGET\r',overwrite=True)

#set up scons environment. use tools=['python'] to save time (~10 sec) searching for
#  the default build tools.
env = Environment(ENV=os.environ,tools=['python'])

#check if files have changed using first timestamps, then MD5 sums
env.Decider('MD5-timestamp')

#windows doesn't include the 3
if env['PLATFORM'] == 'win32':
    env.SetDefault(PYTHON='python')
else:
    env.SetDefault(PYTHON='python3')

# dcraw_emu builder for RAW -> TIFF conversion
#   -o 0 : use camera colorspace
#   -q 0 : use linear interpolation
#   -4 : 16-bit linear
#   -T : TIFF output
env.SetDefault(DCRAW='dcraw_emu')
env.SetDefault(DCRAW_OPTS='-o 0 -q 0 -4 -T') #matches mica toolbox
dcraw = '$DCRAW $DCRAW_OPTS -Z $TARGET $SOURCE'

# exiftool to copy exif information from raw -> tiff
env.SetDefault(EXIFTOOL='exiftool')
env.SetDefault(EXIFTOOL_OPTS='-f -q -overwrite_original')
exiftool = '$EXIFTOOL -tagsFromFile $SOURCE $EXIFTOOL_OPTS $TARGET'

env['BUILDERS']['DCRaw'] = Builder(action=[dcraw,exiftool],suffix='.tiff')

#settings for each camera/lens combo are defined by yaml files
# the Settings class for loading the yaml files is used by other scripts too
# so it's in util.py

def parse_settings(target, settings_path):
    """parse the yaml file and add build tasks to SCons"""
    settings = Settings(settings_path)
    
    ### Image conversion ###

    #only process images if the image directory exists.
    #this lets us distribute test datasets without e.g. 10 GB of image files.
    if Path(settings.image_dir).is_dir():
        dcraw_crop = '-B {} {} {} {}'.format(*settings.crop_xywh)

        tiffs = []
        for wl in settings.images:
            for i in settings.images[wl]:
                raw_img = settings.image_file(i)
                tiffs += env.DCRaw(raw_img, DCRAW_OPTS=f'$DCRAW_OPTS {settings.dcraw_args} {dcraw_crop}',
                                   EXIFTOOL_OPTS=f'$EXIFTOOL_OPTS -title="{wl}"')

        ### collect stats from TIFFs ###
        samples = env.Command(target=settings.samples_file, source=['scripts/image_stats.py',str(settings_path)],
                              action='$PYTHON $SOURCE -i ${SOURCES[1]} -o $TARGET')
        env.Depends(samples, tiffs)

    ### Collate monochromator spectra ###
    
    #only process monochromator data if the spectra dir exists.
    #this lets us distribute test datasets without all of the spectra files.
    if Path(settings.spectra_dir).is_dir():
        spectra = env.Command(target=settings.spectra_file, source=['scripts/monochromator.py',settings.spectra_dir],
                                     action='$PYTHON $SOURCE -i ${SOURCES[1]} -o $TARGET')
        env.Depends(spectra, Glob(f'{settings.spectra_dir}/*'))

    ### Compute camera response from samples ###
    response = env.Command(target=settings.response_file, source=['scripts/camera_response.py',settings.samples_file, settings.spectra_file],
                action='$PYTHON $SOURCE $OPTS -i ${SOURCES[1]} -m ${SOURCES[2]} -o $TARGET',
                OPTS='--non-negative' if settings.non_negative else '')

    ### Make plots ###
    plots = env.Command(target=settings.plots_file, source=['scripts/plots.py',settings.spectra_file, settings.response_file],
                action='$PYTHON $SOURCE -m ${SOURCES[1]} -r ${SOURCES[2]} -o $TARGET')

    ### Make excel ###
    excel = env.Command(target=settings.excel_file, source=['scripts/response_to_excel.py',settings.response_file],
                        action='$PYTHON $SOURCE -o $TARGET ${SOURCES[1]}')

    #ensure that the plots and excel files will be built even though nothing depends on them
    env.Alias(target, plots)
    env.Alias(target, excel)

    return response


print('Parsing target files... ',end='',flush='')
if not COMMAND_LINE_TARGETS:
    COMMAND_LINE_TARGETS = Glob('*.yml', strings=True)

response_files = []
for tgt in COMMAND_LINE_TARGETS:
    settings_file = Path(tgt).with_suffix('.yml')
    if Settings.check_valid(settings_file):
        resp_file = parse_settings(tgt, settings_file)
        response_files.append(resp_file)

if not response_files:
    print('No valid targets!',flush=True)
    Exit(2)
xl = env.Command(target='responses.xlsx', source=['scripts/response_to_excel.py'] + response_files,
                 action='$PYTHON $SOURCE -o $TARGET ${SOURCES[1:]}')
print('DONE',flush=True)
