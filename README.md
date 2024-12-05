# FastMS

Fast approximations of IC malaria models

# Requirements

For IBM model sampling, you will require R >= 4 with mrc-ide/site installed.

```
install.packages('remotes')
remotes::install_github('mrc-ide/site')
```

# Installation

You can install this package from the root of this directory using pip:

```
pip install -e .
```

# Usage

FastMS is best run from the command line. To see the command line options,
please run:

```
python -m fastms --help
```

# Recreating parameter estimates for malariasimulation

 1. Sample simulations from malariasimulation. You can do this with `python -m fastms sample ibm ...` (use the --help option for specific documentation). I have included a script to do this on a PBS cluster in `scripts/sample.h`.
 2. Process the observational data from Battle et al. (2015). Note: requires malaria site data from 'mrc-ide/site'. The script is provided in `scripts/data_prep.R`.
 3. Train the surrogate model. You can do this with `python -m fastms train ...` (use the --help option for specific documentation)
 4. Perform inference. You can do this with `python -m fastms infer ...` (use the --help option for specific documentation)
 5. Recreate the visualisations and analyses. I have included jupytext notebooks to analyse the posterior estimates in `notebooks/Intrinsic_IBM_inf.md` and the surrogate approximation in `notebooks/Intrinsic_IBM_approx.md`
