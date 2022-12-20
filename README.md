
# Camera spectral calibration scripts

## Getting started:

Run these commands in your operating system's terminal.

1. Installing `conda`
    * If you're starting fresh, you can install `conda` and `mamba` in one step from the miniforge project: https://github.com/conda-forge/miniforge#mambaforge

    * If you've installed Anaconda before, you may need to use the "Anaconda Powershell" or similar the first time. Run `conda init` to enable access to `conda` on all system terminals.

    * On Windows, Microsoft's Windows Terminal is a good choice (https://www.microsoft.com/store/productId/9N0DX20HK701).

2. Install `mamba`.
    * This isn't strictly necessary: `mamba` has most of the functionality of `conda` but is significantly faster.
    * To install `mamba` open a terminal and run
        ```
        conda install -c conda-forge mamba
        ```
    * Depending on how you initially set up `conda` you may need to run `conda install ...` as an administrator (run the terminal as an administrator). If that's not possible, just use `conda` in place of `mamba` in the next step.

3. Install dependencies. 
    * `mamba` is recommended as it's much faster!
    * Open a terminal and change to the directory containing this file
        ```
        cd ~/path/to/camera-spectral-calibration
        ```
        or on Windows
        ```
        cd 'path\to\camera-spectral-calibration'
        ```
        
    * Create the environment (specified in `environment.yml`):
        ```
        mamba env create
        ```
4. Activate the environment:
    ```
    conda activate camera-cal
    ```

5. Generate the documentation
    ```
    cd docs
    scons
    ```
6. Now read the full documention in `docs/_build/html/index.html`

