
# Camera spectral calibration scripts

## Getting started:

Run these commands in your operating system's terminal.
On Windows, use Microsoft's Windows Terminal (https://www.microsoft.com/store/productId/9N0DX20HK701).

First, install the execution environment using `conda` or `mamba`.
`mamba` is recommended as it's much faster!

If you already have `conda`, run:

```
conda install mamba
```

If you're starting fresh, install from the miniforge project: https://github.com/conda-forge/miniforge#mambaforge

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

