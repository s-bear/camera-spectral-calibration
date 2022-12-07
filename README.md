
# Camera spectral calibration scripts

## Getting started:

Install the execution environment using conda or mamba.
mamba is recommended as it's much faster!

If you already have conda, run:

```
conda install mamba
```

If you're starting fresh, install from the miniforge project: https://github.com/conda-forge/miniforge#mambaforge

Create the environment (from `environment.yml`):

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

