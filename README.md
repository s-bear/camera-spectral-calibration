
# Camera spectral calibration scripts

## Getting started:

Run these commands in your operating system's terminal.
If you've installed Anaconda before, you may need to use the "Anaconda Powershell" or similar the first time. Run
```
conda init
```
To enable access to `conda` on all system terminals.

On Windows, Microsoft's Windows Terminal is a good alternative (https://www.microsoft.com/store/productId/9N0DX20HK701).

First, install the execution environment using `conda` or `mamba`.
`mamba` is recommended as it's much faster!

If you already have `conda`, run:

```
conda install -c conda-forge mamba
```
Depending on how you initially installed `conda`, you may need to run `conda install` as an administrator. If that's not possible, simply run the command below using `conda` in place of `mamba`--it will be signicantly slower, but still works.

If you're starting fresh, install from the miniforge project: https://github.com/conda-forge/miniforge#mambaforge

Open a terminal and change to the directory containing this file
```
cd ~/path/to/camera-spectral-calibration
```

Create the environment (specified in `environment.yml`):

```
mamba env create
```

Activate the environment:

```
conda activate camera-cal
```

Generate the documentation
```
cd docs
scons
```

Now read the full documention in `docs/_build/html/index.html`

