# LaCE_pk

Routines to play with Lya power spectrum

## Installation

1. Run `git submodule init && git submodule update` in the `LaCE_manager` repo
2. Set environment variables: `export LACE_MANAGER_REPO=/path/to/repo/LaCE_Manager` and `export LACE_REPO=/path/to/repo/LaCE`. Best to set this in a `.bashrc` or similar.
3. Ensure the python dependencies below are installed
4. `cd LaCE` and run `python3 setup.py install --user`
5. `cd ..` and run `python3 setup.py install --user`


#### Dependencies:
Python version 3.6 or later is necessary due to `CAMB` version dependencies.

The following modules are required:

`numpy`

`scipy`

`matplotlib`

`configobj`

`emcee` version 3.0.2 (not earlier ones, they are significantly different apparently)

`tqdm` to work with emcee progress bar

`corner`

`chainconsumer`

`CAMB` version 1.1.3 or later https://github.com/cmbant/CAMB (only works with Python 3.6 or later as of 14/01/2021)

`GPy` (only works with Python 3.8 or lower, not compatible with 3.9 as of 14/01/2021)

`cProfile`

To setup/run/postprocess simulations:

`configargparse`

`fake_spectra` branch at https://github.com/Chris-Pedersen/fake_spectra which includes temperature rescalings in postprocessing

`validate`

`classylss`

`asciitable`

