# synchrofit
Welcome to ```synchrofit``` (**synchro**tron **fit**ter) -- a user-friendly Python package designed to model a synchrotron spectrum. The goal for this package is to provide a fairly accurate<sup>[**1**]</sup> parameterization of a radio spectrum, while requiring little prior knowledge of the source other than its observed spectrum. This code is based on the modified synchrotron models presented by [Turner et al (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract) and [Turner et al (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract).<br />

<sup>[**1**]</sup>Accounting for dynamical changes within the radio source, e.g. an evolving magnetic field, is beyond the scope of this code.

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
This code offers three models describing the synchrotron spectrum, each of which comes in a standard and Tribble form. A brief qualitative description of each model is provided below. <br /> 

**The KP and JP models** <br />
The Kardashev-Pacholczyk (KP; [Kardashev (1962)](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract)) and Jaffe-Perola (JP; [Jaffe & Perola (1973)](https://ui.adsabs.harvard.edu/abs/1973A%26A....26..423J/abstract)) model describe the synchrotron spectrum arising from an **impulsively injected** population of electrons -- that is, an entire electron population injected at *t=0* that undergoes radiative losses thereafter. The main constrast between these two models is the occurrence (JP model) or absence (KP model) of electron pitch angle scattering. 

**CI models** <br />
In contrast to the KP and JP models, the Continuous Injection models (CI-on; [Kardashev (1962)](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract)) and (CI-off; [Komissarov & Gubanov (1994)](https://ui.adsabs.harvard.edu/abs/1994A%26A...285...27K/abstract)) describe the synchrotron spectrum arising from a **continuously injected** electron population -- that is, a mixed-age population of electrons with ages ranging anywhere between *t=0* and *t=\tau*. <br /> The CI-on model describes sources for which energy injection is currently taking place, whereas the CI-off model extends this by assuming the injection has switched off some time ago. 

**The standard *(KP, KP, CI)* and Tribble *(TKP, TJP, TCI)* forms** <br />
For each model described above, we offer a standard and Tribble form that describe the structure of the magnetic field strength across the source. The **standard** form assumes a **uniform magnetic field strength** across the source. By contrast, the **Tribble** form assumes an **inhomogeneous magnetic field strength**, e.g. a Maxwell-Boltzmann distribution as proposed by [Tribble (1991)](https://ui.adsabs.harvard.edu/abs/1991MNRAS.253..147T/abstract). 

The advantage to the synchrotron spectrum described by the standard form is its independence of the magnetic field strength; see Equation 9 of [Turner et al (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract) who demonstrate that the magnetic field strength can be taken out of the integration and simply scale the spectrum. The caveat here is that the assumption of a uniform magnetic field strength is violated within radio lobes. While the Tribble form thus provides a more accurate description of the magnetic field strength structure, the caveat here is that the magnetic field strength must be known in order to parameterize the spectral shape. It should be noted that while there is a noticeable difference in the spectrum expected by the standard versus Tribble forms of the JP and KP, the difference in spectral shape between the Tribble-CI and standard-CI spectrum is negligible (see Section 2.3 of [Turner et al (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract)) <br />

**Free parameters** <br />
Each model described above has a number of free parameters that, ultimately, we want to estimate. The JP and KP models have the energy injection index (*s*) and break frequency *(\nu_b)* as free parameters. Additionally to these two, the CI models also have the quiescent fraction *(T)* as a free parameter (i.e. the fractional time spent in a remnant phase). 

## How does synchrofit work? ##
In essence, `synchrofit` takes a spectral model and estimates its free parameters. This is carried out in a number of primary functions described below.<br />

The functions `spectral_models` and `spectral_models_tribble` contain the standard and Tribble forms of the spectral models described above.
```
spectral_models(frequency, luminosity, fit_type, break_frequency, injection_index, remnant_ratio, normalisation, bessel_x, bessel_F)
**Accepts:**
frequency       : 1darray of input frequencies
luminosity      : 1darray of input frequencies
fit_type        : The type of model to fit
break_frequency : The break frequency 
injection_index : The injection index
remnant_ratio   : The remnant ratio
normalisation   : The normalisation factor
bessel_x        : 1darray of values at which to evaluate the Bessel function
bessel_F        : 1darray containing the values of the Bessel functions
**Returns:**
luminosity_predict : 1darray containing the predicted spectrum
normalisation      : Normalisation factor to correctly scale the spectrum
```
`spectral_models_tribble` is setup identical to this, with the one difference being an additional argument required for the magnetic field strength, e.g:
```
spectral_models_tribble(frequency, luminosity, fit_type, bfield, redshift, break_frequency, injection_index, remnant_ratio, normalisation, bessel_x, bessel_F)
bfield : The magnetic field strength.
```
<br />

Model fitting is performed by the `spectral_fitter` function, which uses an adaptive grid model to estimate the peak probable values for each free parameter. The uncertainty on each free parameter is estimated by taking the standard deviation of its marginal distribution. `spectral_fitter` is setup as follows:
```
spectral_fitter(frequency, luminosity, dluminosity, fit_type, n_breaks=31, break_range=[8,11], 
                n_injects=31, inject_range=[2.01,2.99], n_remnants=31, remnant_range=[0,1], 
                n_iterations=3, options=None)
**Accepts**
frequency     : Input frequency (type = 1darray, unit = Hz)
luminosity    : Input flux densities (type = 1darray, unit = Jy)
dluminosity   : Input flux density uncertainties (type = 1darray, unit = Jy)
fit_type      : The type of model to fit (type = str)
n_breaks      : Number of increments with which to sample the break frequency range (type = int)
break_range   : Accepted range for the log(break frequency) (type = list)
n_injects     : Number of increments with which to sample the injection index range (type = int)
inject_range  : Accepted range for the energy injection index (type = list)
n_remnants    : Number of increments with which to sample the remnant ratio range (type = int)
remnant_range : Accepted range for the remnant ratio (type = list)
n_iterations  : Number of iterations (type = int)
options       : Options parsed through argparse (required only if synchrofit is executed from __main__)
**Returns**
params : A tuple containing (fit_type, break frequency, break frequency uncertainty, injection index, injection index uncertainty, quiescent fraction, quiescent fraction uncertainty, normalisation)
```
An optional feature of `synchrofit` is to evaluate the spectral age using the parameters estimated by `spectral_fitter`. This is perfomed by the `spectral_age` function, which is based upon Equation 4 of [Turner et al (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract). 
```
spectral_ages(params, B, z)
**Accepts**
params : A tuple containing (fit_type, break frequency, quiescent fraction)
B      : The magnetic field strength (nT)
z      : The cosmological redshift. (dimensionless)
**Returns**
tau   : Total spectral age (Myr)
t_on  : Duration of active phase (Myr)
t_off : Duration of remnant phase (Myr)
```
Note, only the CI-off model will return a non-zero value for `t_off`. 

## Usage
### How do I run synchrofit ?
To run `synchrofit` simply execute the following from terminal: <br />
`synchrofit --data ${data.dat} --fit_type ${fit_type}`. <br />
In this example, `${data.dat}` is a `.dat` file containing the input spectrum (see [test_data.dat](https://github.com/synchrofit/synchrofit/test) for an example of the format required), and `${fit_type}` describes the model to be fit (e.g. KP, JP, CI, TKP, TJP, TCI). 

Alternatively, one can manually supply a spectrum by executing the following <br />
`synchrofit --freq f1 f2 fn --flux s1 s2 sn --err_flux es1 es2 esn --fit_type ${fit_type}`. <br />

To integrate this code into your own workflow, simply `import synchrofit` as a package. <br />

<!-- 

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

**Case 4: What if my radio source is ultra-steepa at all frequencies ?**  -->
