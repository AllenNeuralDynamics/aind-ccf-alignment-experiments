# `aind-ccf-alignment-experiments`

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

Experiments and helper utilities for SmartSPIM-to-CCF image registration
primarily within the Insight Toolkit (ITK) image processing ecosystem.

Includes:
- [Example](notebooks/LocalFetchFromS3.ipynb) describing how to obtain SmartSPIM light sheet data
  made publicly available by the Allen Institute for Neural Dynamics.
- [Example](notebooks/RegisterToCCF.ipynb) demonstrating how to run in-memory registration
  of SmartSPIM light sheet data to the Common Coordinates Framework (CCF) atlas.
- [Command line application](src/aind_ccf_alignment_experiments/postprocess_cli.py) for applying registration transform output
  to SmartSPIM and CCF label volumes for evaluation
- Other experiments and motivating examples describing pipeline mechanisms,
  comparison of point- and voxel-based metrics, and results analysis.

Previous experimental methods and results discussion are available at:

http://aind-kitware-collab.s3-website-us-west-2.amazonaws.com/

## Installation

### To install and run notebooks:

1. Create or active a Python virtual environment
2. Set up Jupyter Notebook:
```sh
(my-venv) > python -m pip install jupyter
```
3. Navigate to the repository and install from local sources:
```sh
(my-venv) path/to/aind-ccf-alignment-experiments > python -m pip install .[notebook]
```
4. Launch Jupyter Notebook, navigate to the `notebooks` directory, and run notebooks:
```sh
(my-venv) path/to/aind-ccf-alignment-experiments > jupyter notebook
```

### To install the basics for registration with ITKElastix:

```sh
(my-venv) path/to/aind-ccf-alignment-experiments > python -m pip install .
```

### To install the basics, plus ANTs registration support:
```sh
(my-venv) path/to/aind-ccf-alignment-experiments > python -m pip install .[ants]
```

## Usage

To use `aind_ccf_alignment_experiments` as a library:

```py
import aind_ccf_alignment_experiments
from aind_ccf_alignment_experiments import point_distance as point_distance
# etc
```

To use `postprocess_cli.py` as a command line application for postprocessing with registration results:

```sh
(my-venv) > aind-postprocess-cli --help
```

## Testing

Basic tests are included to ensure the module imports properly.

```sh
(my-venv) path/to/aind_ccf_alignment_experiment > python -m pip install .[develop]
(my-venv) path/to/aind_ccf_alignment_experiment > pytest
```

## Formatting

Use the `black` auto-formatter tool to format your code before contributing to the repository.

```sh
(my-venv) path/to/aind_ccf_alignment_experiment > python -m pip install .[develop]
(my-venv) path/to/aind_ccf_alignment_experiment > python -m black ./src/*
```

## Contributing

Collaborators are encouraged to follow this process for contributions:
1. Fork the repository under your GitHub user account
2. Install packages for development

```sh
(my-venv) path/to/aind_ccf_alignment_experiment > python -m pip install .[develop,notebook]
```

2. Commit your code changes in a branch on your user fork
3. Create a pull request against the main repository

## Acknowledgements

`aind-ccf-alignment-experiments` is primarily a collaborative effort between:

- The [Allen Institute for Neural Dynamics (AIND)](https://alleninstitute.org/division/neural-dynamics/).
- [Kitware](https://www.kitware.com/)

