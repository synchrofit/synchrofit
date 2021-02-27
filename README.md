# synchrofit
Welcome to ```synchrofit``` (**synchro**tron **fit**ter) -- a user-friendly Python package designed to model a synchrotron spectrum. The goal for this package is to provide a fairly accurate parameterization of a radio spectrum, while requiring little prior knowledge of the source other than its observed spectrum. Accounting for dynamical changes within the radio source, e.g. an evolving magnetic field, are beyond the scope of this code.

## Credits
Please credit Ross J. Turner and Benjamin Quici if you use this code, or incorporate it into your own workflow. Please acknowledge the use of this code by providing a link to this repository (a citation will be available shortly). 

## Installation
```synchrofit``` is built and tested on python 3.8.5.

You can install via pip using
`pip3 install synchrofit`

Or you can clone the repository and use `python3 setup.py install` or `pip3 install .`

## Help
Please read through the README.md for a description of the package as well as workflow and usage examples. If you have found a bug or inconsistency in the code please [submit a ticket](https://github.com/synchrofit/synchrofit/issues). 

## Spectral models
This code offers three models describing the synchrotron spectrum, each implemented within `spectral_models`. <br />
A brief qualitative description of each model is provided below. 

<!-- : the Kardashev-Pacholczyk (KP; [Kardashev 1962](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract)) model, the Jaffe-Perola (JP; [Jaffe & Perola 1973](https://ui.adsabs.harvard.edu/abs/1973A%26A....26..423J/abstract)) model and the Continuous-Injection (CI-on; [Kardashev 1962](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract)) and (CI-off; [Komissarov & Gubanov 1994](https://ui.adsabs.harvard.edu/abs/1994A%26A...285...27K/abstract)) models. The expressions for the KP, JP and CI-on models are adapted from [Turner et al (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract), and the expression for the CI-off model is adapted from [Turner et al (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract).  -->


**The KP and JP models** <br />
The Kardashev-Pacholczyk (KP; [Kardashev (1962)](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract)) and Jaffe-Perola (JP; [Jaffe & Perola (1973)](https://ui.adsabs.harvard.edu/abs/1973A%26A....26..423J/abstract)) model describe the synchrotron spectrum arising from an **impulsively injected** population of electrons -- that is, an entire electron population injected at *t=0* that undergoes radiative losses thereafter. The main constrast between these two models is the occurrence (JP model) or absence (KP model) of electron pitch angle scattering. 

**CI models** <br />
In contrast to the KP and JP models, the Continuous Injection models (CI-on; [Kardashev (1962)](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract)) and (CI-off; [Komissarov & Gubanov (1994)](https://ui.adsabs.harvard.edu/abs/1994A%26A...285...27K/abstract)) describe the synchrotron spectrum arising from a **continuously injected** electron population -- that is, a mixed-age population of electrons with ages ranging anywhere between *t=0* and *t=\tau*. The CI-on model describes sources for which energy injection is currently taking place, whereas the CI-off model extends this by assuming the injection has switched off some time ago. 

**The standard *(KP, KP, CI)* and Tribble *(TKP, TJP, TCI)* forms** <br />
For each model described above, we offer a standard and Tribble form that describe the structure of the magnetic field strength across the source. The **standard** form assumes a **uniform magnetic field strength** across the source. By contrast, the **Tribble** form assumes an **inhomogeneous magnetic field strength**, e.g. a Maxwell-Boltzmann distribution as proposed by [Tribble (1991)](https://ui.adsabs.harvard.edu/abs/1991MNRAS.253..147T/abstract). 

The advantage to the synchrotron spectrum described by the standard form is its independence of the magnetic field strength; [see Equation 9 of Turner et al (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract) demonstrated that the magnetic field strength can be taken out of the integration and simply scale the spectrum. The caveat here is that the assumption of a uniform magnetic field strength is violated within radio lobes. While the Tribble form thus provides a more accurate description of the magnetic field strength structure, the caveat here is that the magnetic field strength must be known in order to parameterize the spectral shape. 

## Main modules
- `spectral_models` contains the KP, JP and CI models used in the fitting. 
- `spectral_fitter` takes an input radio spectrum and applies an adaptive grid model fitting with Bayesian inference in order to estimate the injection index, break frequency and quiescent fraction. By generating a probability distribution over a grid of input parameters, the optimal values for each parameter are estimated by taking the peak of the probability distribution. 
- `spectral_data` constructs a model spectrum using the parameters estimated by `spectral_fitter`.
- `spectral_ages` provides an optional feature to determine the spectral ages using the break frequency and quiescent fraction estimated by `spectral_fitter`. This requires both a magnetic field strength and cosmological redshift to be supplied. Ages are derived using [Equation 4 of Turner, et al (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract). 
- `spectral_plotter` provides an optional feature to plot the input data and fitted model for a visual comparison.

## Usage
### How do I run synchrofit ?
```synchrofit``` is easily ran by executing the following from a terminal: 
`synchrofit --data ${data.dat} --fit_type ${fit_type}`. 
In this example, `${data.dat}` is a `.dat` file containing the input spectrum (see [test_data.dat](https://github.com/synchrofit/synchrofit/test) for example of required format), and `${fit_type}` describes the model to be fit (e.g. KP, JP or CI). 

Alternatively, one can manually supply a spectrum by executing the following 
`synchrofit --freq f1 f2 fn --flux s1 s2 sn --err_flux es1 es2 esn --fit_type ${fit_type}`. 

### Default and custom configurations
```synchrofit``` is setup with a default preset for all parameters related to the model fitting. The current default values seem to provide a good balance between the coarseness of the adaptive grid and the processing speeds. Any of these values can, however, be adjusted by the user based on their requirements. A complete list of arguments accepted by ```synchrofit``` and their descriptions is listed below. 
- `--work_dir` the directory to which fitting outputs and plots are written to. If None, defaults to current working directory. Default = None.
- `--data` name of the data file containing the input spectrum (requires .dat format). Default = None.
- `--freq` list of input frequencies (if `--data` is not specified). Default = None. 
- `--flux` list of input flux densities (if `--data` is not specified). Default = None. 
- `--err_flux` list of input flux density errors (if `--data` is not specified). Default = None. 
- `--freq_unit` input frequency units. Default = Hz.
- `--flux_unit` input flux density units. Default = Jy.
- `--fit_type` Model to fit, e.g. KP, JP or CI. No default. 
- `--n_breaks` Number of increments with which to sample the break frequency range. Default = 31.
- `--n_injects` Number of increments with which to sample the injection index range. Default = 31.
- `--n_remnants` Number of increments with which to sample the remnant ratio range. Default = 31.
- `--n_iterations` 
- `--break_range` Accepted range for the log(break frequency). Default = [8, 11]. 
- `--inject_range` Accepted range for the energy injection index. Default = [2.01, 2.99]. 
- `--remnant_range` Accepted range for the remnant ratio. Default = [0, 1]. 
- `--n_model_freqs` Number of frequencies to evaluate the model at. Default = 100.
- `--mc_length` Number of Monte-Carlo iterations used to estimate the model uncertainty. Default = 500. 
- `--err_model_width` Width of the uncertainty envelope on the model, in units of sigma. Default = 2. 
- `--plot` If True, produce a plot of the observed spectrum and fitted model. Default = False. 
- `--write_model` If True, write the fitting outputs to file. Default = False.
- `--age` If True, determine the spectral age of the source (requires `--bfield` and `--z`). Default = False.
- `--bfield` Magnetic field strength in units of nT. No default. 
- `--z` Cosmological redshift of the source. No default. 

An example of a custom configuration might look as follows:
`synchrofit --data ${data.dat} --fit_type CI --n_breaks 21 --n_injects 21 --n_remnants 21 --n_iterations 5 --break_range 8 10 --inject_range 2.0 3.0 --mc_length 1000`

Consider loading any custom configuration into `run_synchrofit.sh`, which will allow you to save and store these presets for later re-use. To execute this custom setup, simply run `./run_synchrofit.sh` from the terminal.

### What to do in practice
Now that we know how to run ```synchrofit```, below are some suggestions for how you might want to implement this for your radio source. 


**I want to constrain the break frequency**

**I want to measure the spectral age**


**Case 1: I have an integrated radio spectrum, what should I do ?**
In this case, fitting the Continuous Injection models `--fit_type CI` is most applicable, as this takes into consideration the averaging over a mixed-age plasma. By default, `--fit_type CI` will fit the spectrum using a CI-off model. If the radio galaxy is known to be active the spectrum needs to be modelled using the simpler CI-on model. This is done setting `--remnant_range 0`.

**Case 2: I have a spatially-resolved radio spectrum**
In this case, fitting either the KP or JP models is applicable here as we are simply interested in the total spectral age of a small emitting-region.

**Case 3: Is my radio source active or remnant ?**
**#TODO** Perhaps you have a radio source with a curved radio spectrum and would like to ascertain whether the nature of the spectral curvature. 

**Case 4: What if my radio source is ultra-steepa at all frequencies ?** 
