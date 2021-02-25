# synchrofit
=====
```synchrofit``` (**synchro**tron **fit**) provides a user-friendly Python package that will model the synchrotron spectrum arising from a radio galaxy. Three models are offered: the Kardashev-Pacholczyk (KP) model (ref), the Jaffe-Perola (JP) model (ref) and the Continuous-Injection (CI) on/off model (CI-off also known as the KGJP model). The expressions for the KP, JP and CI-on models are adapted from Turner, et al (2017b)<sup>**1**</sup>, and the expression for the CI-off model is adapted from Turner, et al (2018)<sup>**2**</sup>:
- <sup>**1**</sup> https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract
- <sup>**2**</sup> https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract

## Credits
Please credit Ross J. Turner and Benjamin Quici if you use this code, or incorporate it into your own workflow. Please acknowledge the use of this code by citing ___ and by providing a link to this repository. 

## Installation
```synchrofit``` is built and tested on python 3.8.5.

You can install via pip using
`pip install synchrofit`

Or you can clone the repository and use `python3 setup.py install` or `pip install .`

## Help
Please read through the README.md for a description of the package as well as workflow and usage examples. If you have found a bug or inconsistency in the code please [submit a ticket](https://github.com/synchrofit/synchrofit/issues). 

## Scripts
- `synchrofit.py` Main code that performs the fitting. 
- `run_synchrofit.sh` Offers a shell-like execution of `synchrofit.py` and contains the relevant configuration options.

## Example workflow

A typical workflow might look like:
   - Supply a list of frequencies, flux densities and flux density uncertainties that describe the spectrum
   - Select either the JP, KP or CI model and fit the spectrum. This will return the following:
       - Optimal fit of the injection index and break frequency, including uncertainties on each
       - The fractional time spent in a remnant phase, if the CI-off model is fit
       - An output model spectrum evaluated over a list of frequencies
   - (optional) Derive the total spectral age (as well as active and inactive ages in the case of the CI-off model). This requires the following:
       - The cosmological redshift of the radio source
       - The magnetic field strength. For a radio galaxy, you may want to consider dynamically estimating this using RAiSE

## Configuring synchrofit.py
At minimum, `synchrofit.py` requires the working directory, input spectrum and the type of model to fit. In the example below, the spectral data is sourced from $work_dir/test_spectra.dat and is fit by the JP model.
```
python3 synchrofit.py \
    --work_dir $work_dir \
    --data "test_spectra.dat" \
    --fit_type "JP" \
```
Alterntively, the spectrum can be supplied manually as follows:
```
python3 synchrofit.py \
    --work_dir $work_dir \
    --freq f1 f2 ... fn \
    --flux s1 s2 ... sn \
    --err_flux es1 es2 ... esn \
    --fit_type "JP" \
```

- `--plot` will plot the input spectrum and the best model fit and write to ${work_dir}/${fit_type}_fit.png. 
- `--write_model` will write the estimated parameters and output model to ${work_dir}/estimated_params_${fit_type}.dat and ${work_dir}/modelspectrum_${fit_type}.dat, respectively. 
- `--age` will determine the spectral ages if a magnetic field strength and redshift is supplied.

Most variables will already have default values set in `synchrofit.py`, and can be changed following the examples above. For a description of each variable use ```synchrofit.py --h```.

## Execution
`synchrofit.py` can either be executed directly from terminal, or by running `./run_synchrofit.sh`. The latter option allows one to store multiple configuration presets. 