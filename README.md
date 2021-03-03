# synchrofit
The ```synchrofit``` (**synchro**tron **fit**ter) package implements a reduced dimensionality parameterisation ([Turner et al. 2018b](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract), [Turner 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract)) of standard synchrotron spectrum models, and provides fitting routines applicable for active galactic nuclei and supernova remnants. The Python code includes the Jaffe-Parola model (JP; [Jaffe & Perola 1973](https://ui.adsabs.harvard.edu/abs/1973A%26A....26..423J/abstract)), Kardashev-Pacholczyk model (KP; [Kardashev 1962](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract), [Pacholczyk 1970](https://ui.adsabs.harvard.edu/abs/1970ranp.book.....P/abstract)), and continuous injection models (CI/KGJP; [Komissarov & Gubanov 1994](https://ui.adsabs.harvard.edu/abs/1994A%26A...285...27K/abstract)) for both constant or Maxwell-Boltzmann magnetic field distributions. An adaptive maximum likelihood algorithm is invoked to fit these models to multi-frequency radio observations; the adaptive mesh is customisable for either optimal precision or computational efficiency. Functions are additionally provided to plot the fitted spectral model with its confidence interval, and to derive the spectral age of the synchrotron emitting particles. 

<!-- Welcome to ```synchrofit``` (**synchro**tron **fit**ter) -- a user-friendly Python package designed to model a synchrotron spectrum. The goal for this package is to provide an accurate<sup>[**1**]</sup> parameterization of a radio spectrum, while requiring little prior knowledge of the source other than its observed spectrum. This code is based on the modified synchrotron models presented by [Turner et al (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract) and [Turner et al (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract).<br />

<sup>[**1**]</sup>Accounting for dynamical changes within the radio source, e.g. an evolving magnetic field, is beyond the scope of this code. -->

## Credits
Please credit Ross J. Turner and Benjamin Quici if you use this code, or incorporate it into your own workflow. Please acknowledge the use of this code by providing a link to this repository (a citation will be available shortly). 

## Installation
```synchrofit``` is built and tested on python 3.8.5.<br /> 

To obtain this code you can either download the repository, or, clone with git using<br /> 
`git clone https://github.com/synchrofit/synchrofit.git`<br /> 

To install `synchrofit` from the command line, `cd` into the root directory and use either:
`python3 setup.py install` <br /> 
or <br /> 
`pip3 install .`

## Help
Please read through the README.md for a description of the package as well as workflow and usage examples. If you have found a bug or inconsistency in the code please [submit a ticket](https://github.com/synchrofit/synchrofit/issues).  

## Contents
**Skip to:**<br />
- [Spectral models](#spectral-models)
    - [The KP and JP models](#the-kp-and-jp-models)
    - [CI models](#ci-models)
    - [The standard *(KP, JP, CI)* and Tribble *(TKP, TJP, TCI)* forms](#the-standard-kp-jp-ci-and-tribble-tkp-tjp-tci-forms)
    - [Free parameters](#free-parameters)
- [How does synchrofit work?](#how-does-synchrofit-work-)
    - [spectral_fitter](#spectral_fitter)
    - [spectral_models](#spectral_models)
    - [spectral_data](#spectral_data)
    - [spectral_ages](#spectral_ages)
- [Usage](#Usage)
    - [How do I run synchrofit ?](#how-do-I-run-synchrofit-)
    - [I have an integrated radio galaxy spectrum, what should I do ?](#i-have-an-integrated-radio-galaxy-spectrum-what-should-I-do-)
    - [I want to model the spectrum of a supernova remnant, what should I do ?](#i-want-to-model-the-spectrum-of-a-supernova-remnant-what-should-I-do-)
    - [I want to evaluate the spectral age from my radio spectrum, what should I do ?](#i-want-to-evaluate-the-spectral-age-from-my-radio-spectrum-what-should-I-do-)
- [Default and custom configurations](#default-and-custom-configurations)
    
## Spectral models
This code offers three models describing the synchrotron spectrum, each of which comes in a standard and Tribble form. A brief qualitative description of each model is provided below. <br /> 

### The KP and JP models
The Kardashev-Pacholczyk (KP; [Kardashev (1962)](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract)) and Jaffe-Perola (JP; [Jaffe & Perola (1973)](https://ui.adsabs.harvard.edu/abs/1973A%26A....26..423J/abstract)) model describe the synchrotron spectrum arising from an **impulsively injected** population of electrons -- that is, an entire electron population injected at *t=0* that undergoes radiative losses thereafter. The main constrast between these two models is the occurrence (JP model) or absence (KP model) of electron pitch angle scattering. 

### CI models
In contrast to the KP and JP models, the Continuous Injection models (CI-on; [Kardashev (1962)](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract)) and (CI-off; [Komissarov & Gubanov (1994)](https://ui.adsabs.harvard.edu/abs/1994A%26A...285...27K/abstract)) describe the synchrotron spectrum arising from a **continuously injected** electron population -- that is, a mixed-age population of electrons with ages ranging anywhere between *t=0* and *t=τ*. <br /> The CI-on model describes sources for which energy injection is currently taking place, whereas the CI-off model extends this by assuming the injection has switched off some time ago. 

### The standard *(KP, JP, CI)* and Tribble *(TKP, TJP, TCI)* forms
For each model described above, we offer a standard and Tribble form that describe the structure of the magnetic field strength across the source. The **standard** form assumes a **uniform magnetic field strength** across the source. By contrast, the **Tribble** form assumes an **inhomogeneous magnetic field strength**, e.g. a Maxwell-Boltzmann distribution as proposed by [Tribble (1991)](https://ui.adsabs.harvard.edu/abs/1991MNRAS.253..147T/abstract). 

The advantage to the synchrotron spectrum described by the standard form is its independence of the magnetic field strength; see Equation 9 of [Turner et al (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract) who demonstrate that the magnetic field strength can be taken out of the integration and simply scale the spectrum. The caveat here is that the assumption of a uniform magnetic field strength is violated within radio lobes. While the Tribble form thus provides a more accurate description of the magnetic field strength structure, the caveat here is that the magnetic field strength must be known in order to parameterize the spectral shape. It should be noted that while there is a noticeable difference in the spectrum expected by the standard versus Tribble forms of the JP and KP, the difference in spectral shape between the Tribble-CI and standard-CI spectrum is negligible (see Section 2.3 of [Turner et al (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract)) <br />

### Free parameters
Both the standard and Tribble forms of the KP and JP spectrum are parameterised in terms of:
- **the injection index, *s***. The injection index is defined through N(E)dE ∝ E<sup>-s</sup>dE and describes the slope of the electron energy distribution. The injection index is related to the spectral injection index, α<sub>inj</sub>, through α<sub>inj</sub> = (*s* − 1)/2).
- **the break frequency, *ν<sub>b</sub>***. The break frequency represents the frequency above which the spectrum steepens steepens as a result of the energy loss mechanisms. <br />
 
In addition to the injection index and break frequency, the parameterisation of the CI model requires:
- **the remnant fraction, *T***. The remnant fraction is defined through *T = t<sub>off</sub>/τ* and gives the fractional time spent in an inactive phase (t<sub>off</sub>) with respect to the total source age (τ). 

## How does synchrofit work? ##
In essence, `synchrofit` takes a spectral model and estimates its free parameters. This is carried out in a number of primary functions described below.<br />

### spectral_models
The functions `spectral_models` and `spectral_models_tribble` contain the standard and Tribble forms of the spectral models described above.
```
spectral_models(frequency, luminosity, fit_type, break_frequency, injection_index, remnant_ratio, normalisation, bessel_x, bessel_F)
**Accepts:**
frequency       : Input frequencies
                  (type = 1darray, unit = Hz)
luminosity      : Input flux densities
                  (type = 1darray, unit = Jy)
fit_type        : The type of model to fit
                  (type = str)
break_frequency : The break frequency 
                  (type = float, unit = Hz)
injection_index : The injection index
                  (type = float, unit = dimensionless)
remnant_ratio   : The remnant ratio
                  (type = float, unit = dimensionless)
normalisation   : The normalisation factor
                  (type = float)
bessel_x        : Values at which to evaluate the Bessel function
                  (type = 1darray)
bessel_F        : Evaluated Bessel function
                  (type = 1darray)
**Returns:**
luminosity_predict : The predicted spectrum
                     (type = 1darray, unit = Jy)
normalisation      : Normalisation factor to correctly scale the spectrum
                     (type = float, unit = dimensionless)
```
`spectral_models_tribble` is setup identical to this, with the one difference being an additional argument required for the magnetic field strength, e.g:
```
spectral_models_tribble(frequency, luminosity, fit_type, bfield, redshift, break_frequency, injection_index, remnant_ratio, normalisation, bessel_x, bessel_F)
bfield : The magnetic field strength.
```

### spectral_fitter
Model fitting is performed by the `spectral_fitter` function, which uses an adaptive grid model to estimate the peak probable values for each free parameter. The uncertainty on each free parameter is estimated by taking the standard deviation of its marginal distribution. `spectral_fitter` is setup as follows:
```
spectral_fitter(frequency, luminosity, dluminosity, fit_type, n_breaks=31, break_range=[8,11], 
                n_injects=31, inject_range=[2.01,2.99], n_remnants=31, remnant_range=[0,1], 
                n_iterations=3, options=None)
**Accepts**
frequency     : Input frequencies
                (type = 1darray, unit = Hz)
luminosity    : Input flux densities 
                (type = 1darray, unit = Jy)
dluminosity   : Input flux density uncertainties 
                (type = 1darray, unit = Jy)
fit_type      : The type of model to fit 
                (type = str)
n_breaks      : Number of increments with which to sample the break frequency range 
                (type = int)
break_range   : Accepted range for the log(break_frequency) 
                (type = list)
n_injects     : Number of increments with which to sample the injection index range 
                (type = int)
inject_range  : Accepted range for the energy injection index 
                (type = list)
n_remnants    : Number of increments with which to sample the remnant ratio range 
                (type = int)
remnant_range : Accepted range for the remnant ratio 
                (type = list)
n_iterations  : Number of iterations 
                (type = int)
options       : Options parsed through argparse 
                (required only if synchrofit is executed from __main__)
**Returns**
params : fit_type, log(break frequency), log(break frequency uncertainty), injection index, 
         injection index uncertainty, quiescent fraction, quiescent fraction uncertainty, normalisation
         (type = tuple)
```
Note, you do not need to interface with `spectral_fitter` as it is an internal function. <br />

### spectral_data
Once you have determined the optimal fit, you might want to construct a model spectrum e.g. to compare the observed and model data, or to simulate the model over a range of frequencies to visualize on a plot. This is performed using the `spectral_data` function which takes the parameters estimated by `spectral_fitter` and simulates the model spectrum. `spectral_data` uses the uncertainties in each free parameter and estimates the uncertainties in the model using a standard Monte-Carlo approach.
```
spectral_data(params, frequency_observed=None, n_model_freqs=100, mc_length=500, err_model_width=2,
              work_dir=None, write_model=None)
**Accepts**
params             : fit_type, log(break frequency), log(break frequency uncertainty), injection index, 
                     injection index uncertainty, quiescent fraction, quiescent fraction uncertainty, normalisation
                     (type = tuple)
frequency_observed : Oberserved frequencies. If not None, will evaluate the model at each observing frequency
                     and will use the observed frequencies to define the bounds of the simulated plotting frequencies
                     If None, will simply evaluate the model at the simulate plotting frequencies, which by default
                     range between 50 MHz and 50 GHz. 
                     (type = 1darray, unit = Hz)
n_model_freqs      : Number of plotting frequencies to simulate within the allowed range
                     (type = int)
mc_length          : Number of Monte-Carlo iterations
                     (type = int)
err_model_width    : Width of the model uncertainty envelope in multiples of sigma 
                     (type = int)
work_dir           : Directory to write outputs to
                     (type = str)
write_model        : If True, writes model spectrum to work_dir
                     (type = Bool)
**Returns**
spectral_model_plot_data  : Matrix containing the simulated plotting frequencies, 
                            model flux densities evaluated for the simulated plotting frequencies
                            uncertainties in the model flux densities
                            lower bound for the model flux densities
                            upper bound for the model flux densities   
                            (type = np.array)
luminosity_model_observed : The model evaluate at the observed frequencies. 
                            If frequency_observed == None, luminosity_model_observed = None
                            (type = np.1darray)
```

### spectral_ages
An optional feature of `synchrofit` is to evaluate the spectral age using the parameters estimated by `spectral_fitter`. This is perfomed by the `spectral_age` function, which is based upon Equation 4 of [Turner et al (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract). 
```
spectral_ages(params, B, z)
**Accepts**
params : fit_type, break frequency, quiescent fraction
         (type = tuple)
B      : The magnetic field strength
         (type = float, unit = nT)
z      : The cosmological redshift.
         (type = float, unit = dimensionless)
**Returns**
tau   : Total spectral age 
        (type = float, unit = Myr)
t_on  : Duration of active phase
        (type = float, unit = Myr)
t_off : Duration of remnant phase
        (type = float, unit = Myr)
```
Note, only the CI-off model will return a non-zero value for `t_off`. 

## Usage
### How do I run synchrofit ?
**Command-line execution**<br />
To run `synchrofit` simply execute the following from terminal: <br />
`synchrofit --data ${data.dat} --fit_type ${fit_type}`. <br />
In this example, `${data.dat}` contains the input spectrum (see [test_spectra.dat](https://github.com/synchrofit/synchrofit/tree/main/Example) for an example of the format required), and `${fit_type}` describes the model to be fit (e.g. KP, JP, CI, TKP, TJP, TCI). 

Alternatively, one can manually supply a spectrum by executing the following <br />
`synchrofit --freq f1 f2 fn --flux s1 s2 sn --err_flux es1 es2 esn --fit_type ${fit_type}`. <br />

**Integrate modules into workflow**<br />
To integrate this code into your own workflow, simply import synchrofit into your Python code:<br />
 `from SynchrofitTool import synchrofit`. <br />
 or:<br />
 `from SynchrofitTool.synchrofit import spectral_fitter, spectral_data, spectral_plotter, spectral_ages, spectral_units`<br />

### I have an integrated radio galaxy spectrum, what should I do ? ###
In this case fitting the standard forms of the Continuous Injection models is most applicable. By default, `--fit_type CI` will fit the spectrum using a CI-off model. If the radio galaxy is **known to be active** the spectrum needs to be modelled using the simpler **CI-on** model. This is done setting `--remnant_range 0`. This will look as follows:<br />
`synchrofit --data ${data_file.dat} --fit_type CI --remnant_range 0` <br />
or as follows if executing the function within your own workflow: <br />
`spectral_fitter($frequency, $luminosity, $dluminosity, CI, remnant_range=0)`

### I want to model the spectrum of a supernova remnant, what should I do ? ###
In this case fitting the standard forms of the JP or KP models is best, given the supernova remnant is just a shell of impulsively-injected electrons. This will look as follows:<br />
`synchrofit --data ${data_file.dat} --fit_type JP` <br />
or as follows if executing the function within your own workflow: <br />
`spectral_fitter($frequency, $luminosity, $dluminosity, JP)`

### I want to evaluate the spectral age from my radio spectrum, what should I do ? ###
Firstly, this requires that you provide a value for the magnetic field strength (nT) and a cosmological redshift. In the example below, we evaluate the spectral ages of an inactive (remnant) radio galaxy at redshift `z = 0.2` and with a lobe magnetic field strength of `B = 0.5 nT`:<br />
`synchrofit --data ${data_file.dat} --fit_type CI --age --b_field 0.5--z 0.2` <br />
or as follows if executing the function within your own workflow: <br />
```
params = spectral_fitter($frequency, $luminosity, $dluminosity, CI)
params = (params[0], 10**params[1], params[5])
spectral_age(params, 0.5, 0.2)
```

If you have a spatially-resolved radio spectrum, you may want to consider mapping the age across the lobes. In this case you will need to integrate `synchrofit` into your own workflow by fitting each resolved spectrum yourself (currently `synchrofit` will not do this automatically). Assuming you are able to do this, follow the example above using either the JP or KP model instead. We note however that if the plasma within the lobes is well-mixed, the resolved age map will not give the true spectral age of the source as even the oldest regions will contain relatively young electrons that will dominate the radio spectrum. 

## Default and custom configurations
Most parameters accepted by `spectral_fitter` already have default values. The current default values seem to provide a good balance between the coarseness of the adaptive grid and the processing speeds. Any of these values can, however, be adjusted by the user based on their requirements. A complete list of arguments accepted by ```synchrofit``` and their descriptions is listed below. 
- `--work_dir` the directory to which fitting outputs and plots are written to. If None, defaults to current working directory. Default = None.
- `--data` name of the data file containing the input spectrum (requires .dat format). Default = None.
- `--freq` list of input frequencies (if `--data` is not specified). Default = None. 
- `--flux` list of input flux densities (if `--data` is not specified). Default = None. 
- `--err_flux` list of input flux density errors (if `--data` is not specified). Default = None. 
- `--freq_unit` input frequency units. Default = Hz.
- `--flux_unit` input flux density units. Default = Jy.
- `--fit_type` Model to fit, e.g. KP, JP, CI TKP, TJP, TCI. No default. 
- `--n_breaks` Number of increments with which to sample the break frequency range. Default = 31.
- `--n_injects` Number of increments with which to sample the injection index range. Default = 31.
- `--n_remnants` Number of increments with which to sample the remnant ratio range. Default = 31.
- `--n_iterations` Number of iterations
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
