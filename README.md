# synchrofit
The ```synchrofit``` (**synchro**tron **fit**ter) package implements a reduced dimensionality parameterisation ([Turner et al. 2018b](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract), [Turner 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract)) of standard synchrotron spectrum models, and provides fitting routines applicable for active galactic nuclei and supernova remnants. The Python code includes the Jaffe-Parola model (JP; [Jaffe & Perola 1973](https://ui.adsabs.harvard.edu/abs/1973A%26A....26..423J/abstract)), Kardashev-Pacholczyk model (KP; [Kardashev 1962](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract), [Pacholczyk 1970](https://ui.adsabs.harvard.edu/abs/1970ranp.book.....P/abstract)), and continuous injection models (CI/KGJP; [Komissarov & Gubanov 1994](https://ui.adsabs.harvard.edu/abs/1994A%26A...285...27K/abstract)) for both constant or Maxwell-Boltzmann magnetic field distributions. An adaptive maximum likelihood algorithm is invoked to fit these models to multi-frequency radio observations; the adaptive mesh is customisable for either optimal precision or computational efficiency. Functions are additionally provided to plot the fitted spectral model with its confidence interval, and to derive the spectral age of the synchrotron emitting particles. 

<!-- Welcome to ```synchrofit``` (**synchro**tron **fit**ter) -- a user-friendly Python package designed to model a synchrotron spectrum. The goal for this package is to provide an accurate<sup>[**1**]</sup> parameterization of a radio spectrum, while requiring little prior knowledge of the source other than its observed spectrum. This code is based on the modified synchrotron models presented by [Turner et al (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract) and [Turner et al (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract).<br />

<!--<sup>[**1**]</sup>Accounting for dynamical changes within the radio source, e.g. an evolving magnetic field, is beyond the scope of this code. -->

## Credits
Please credit Ross J. Turner and Benjamin Quici if you use this code, or incorporate it into your own workflow. Please acknowledge the use of this code by providing a link to this repository and citing [Quici et al. 2022](https://academic.oup.com/mnras/article/514/3/3466/6588050). 

## Installation
```synchrofit``` is built and tested on python 3.8.5.<br /> 

To obtain this code you can either download the repository, or, clone with git using<br /> 
`git clone https://github.com/synchrofit/synchrofit.git`<br /> 

To install `synchrofit` from the command line, `cd` into the root directory and use either:<br /> 
`python setup.py install` <br /> 
or <br /> 
`pip install .` <br /> 

Note, `pip` and `pip3` can be used interchangeably.

## Help
Please read through the README.md for a description of the package as well as workflow and usage examples. If you have found a bug or inconsistency in the code please [submit a ticket](https://github.com/synchrofit/synchrofit/issues).  

## Contents
**Skip to:**<br />
[Theory](#theory)
- [The KP and JP models](#the-kp-and-jp-models)
- [CI models](#ci-models)
- [The standard and Tribble forms](#the-standard-and-tribble-forms)
- [Free parameters](#free-parameters) 

[Functions](#functions)
- [spectral_fitter](#spectral_fitter)
- [spectral_model](#spectral_model)
- [spectral_ages](#spectral_ages)
- [Private functions](#private_functions)

[Usage](#Usage)
- [How do I run synchrofit ?](#how-do-I-run-synchrofit-)
    - [Command-line execution](#command-line-execution)
    - [Integrate modules into workflow](#integrate-modules-into-workflow)
- [Example applications for synchrofit](#example-applications-for-synchrofit)
    - [I have an integrated radio galaxy spectrum, what should I do ?](#i-have-an-integrated-radio-galaxy-spectrum-what-should-I-do-)
    - [I want to model the spectrum of a supernova remnant, what should I do ?](#i-want-to-model-the-spectrum-of-a-supernova-remnant-what-should-I-do-)
    - [I want to evaluate the spectral age from my radio spectrum, what should I do ?](#i-want-to-evaluate-the-spectral-age-from-my-radio-spectrum-what-should-I-do-)
- [Default and custom configurations](#default-and-custom-configurations)


## Theory
The `synchrofit` package offers the standard and Tribble forms of **three** synchrotron spectrum models. A brief qualitative description of each model is provided below. <br /> 

### The KP and JP models
The Kardashev-Pacholczyk (KP; [Kardashev 1962](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract), [Pacholczyk 1970](https://ui.adsabs.harvard.edu/abs/1970ranp.book.....P/abstract)) and Jaffe-Perola (JP; [Jaffe & Perola 1973](https://ui.adsabs.harvard.edu/abs/1973A%26A....26..423J/abstract)) models describe the synchrotron spectrum arising from an **impulsively injected** population of electrons -- that is, the entire electron population is injected at *t=0* and thereafter undergoes radiative losses. The difference between these two models is the presence (JP model) or absence (KP model) of electron pitch angle scattering. 

### CI models
In contrast to the KP and JP models, the continuous injection models (CI-on; [Kardashev 1962](https://ui.adsabs.harvard.edu/abs/1962SvA.....6..317K/abstract), CI-off; [Komissarov & Gubanov 1994](https://ui.adsabs.harvard.edu/abs/1994A%26A...285...27K/abstract)) describe the synchrotron spectrum arising from a **continuously injected** electron population -- that is, a mixed-age population of electrons with ages ranging uniformly between *t=0* and the source age *t=τ*. <br /> The CI-on model describes sources for which energy injection is currently taking place, whereas the CI-off model extends this by assuming the injection has switched off for a period of time *t<sub>off</sub>*.

### The standard and Tribble forms
For each model described above, we offer a standard and Tribble form that describe the local structure of the magnetic field strength. These are referred to as TJP, TKP and TCI. The **standard** form assumes a **constant magnetic field strength** across the source. By contrast, the **Tribble** form assumes a locally **inhomogeneous magnetic field strength**, e.g. a Maxwell-Boltzmann distribution as proposed by [Tribble (1991)](https://ui.adsabs.harvard.edu/abs/1991MNRAS.253..147T/abstract).

The advantage to the synchrotron spectrum described by the standard form is its independence of the magnetic field strength; see Equation 9 of [Turner et al. (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract).<!--  who reduce to dimensionality of a triple integral by parameterising in terms of a varible combining the effect of the energy and magnetic field strength. -->
This form with greatly improved computational efficiency provides comparable spectra to the Tribble model for the CI, and to a lesser extent, the KP models; the JP model is quite different when using the standard and Tribble forms (see Section 2.3 of [Turner et al. 2018b](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract)). The Tribble model does provide a more accurate description of the magnetic field strength structure but the further caveat is that the magnetic field strength must be known in order to fit the spectral shape. 

<!-- It should be noted that while there is a noticeable difference in the spectrum expected by the standard versus Tribble forms of the JP and KP, the difference in spectral shape between the Tribble-CI and standard-CI spectrum is negligible (see Section 2.3 of [Turner et al (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3361T/abstract)) <br /> -->

### Free parameters
Both the standard and Tribble forms of the JP, KP and CI-on spectra are parameterised in terms of:
- **the injection index, *s***. The injection index is defined through *N(E)dE ∝ E<sup>-s</sup>dE* and describes the slope of the electron energy distribution at acceleration (or *injection* into an AGN lobe). The injection index is related to the injection spectral index, α<sub>inj</sub>, through α<sub>inj</sub> = (*s* − 1)/2.
- **the break frequency, *ν<sub>b</sub>***. The break frequency represents the frequency above which the spectrum steepens as a result of the energy loss mechanisms. <br />
 
In addition to the injection index and break frequency, the parameterisation of the CI-off model requires:
- **the remnant fraction, *T***. The remnant fraction is defined through *T = t<sub>off</sub>/τ* and gives the fractional time spent in an inactive phase (t<sub>off</sub>) with respect to the total source age (τ). 

## Functions ##
In short, `synchrofit` fits any of the models described in [Theory](#theory) to a multi-frequency radio spectrum. This is carried out in a number of functions, which are described below.<br />

### spectral_fitter
Model fitting is performed by the `spectral_fitter` function, which uses an adaptive maximum likelihood algorithm to fit the observed radio spectrum; the adaptive mesh is customisable for either optimal precision or computational efficiency. In this way, the spectral index, break frequency and remnant fraction are estimated. Uncertainties on each parameter are quantified by taking the standard deviation of its marginal distribution. `spectral_fitter` is setup as follows:
```
spectral_fitter(frequency : (list, np.ndarray), luminosity : (list, np.ndarray), dluminosity : (list, np.ndarray), \
    fit_type : str, n_breaks=31, break_range=[8,11], n_injects=31, inject_range=[2.01,2.99], n_remnants=31, \
    remnant_range=[0,1], n_iterations=3, b_field=None, redshift=None, write_model=False, work_dir=None, save_prefix=None):
    
    (usage) Finds the optimal fit for a radio spectrum modelled by either the JP, KP or CI model.
    
    Parameters
    ----------
    frequency       : an input list of frequencies (Hz).

    luminosity      : an input list of flux densities (Jy).

    dluminosity     : an input list of flux density uncertainties (Jy).

    fit_type        : the spectral model, must be one of [KP, TKP, JP, TJP, CI, TCI].

    n_breaks        : number of increments with which to sample the break frequency range.

    break_range     : accepted range for the log(break_frequency) (log Hz).

    n_injects       : number of increments with which to sample the injection index range.

    inject_range    : accepted range for the energy injection index.

    n_remnants      : number of increments with which to sample the remnant ratio range.

    remnant_range   : accepted range for the remnant ratio.

    n_iterations    : number of iterations.

    b_field         : the magnetic field strength (T).

    redshift        : the cosmological redshift (dimensionless).

    Returns
    -------
    params : (tuple) fit_type, break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation
             fit_type         : the chosen model for fitting.
             break_predict    : the log break frequency (Hz).
             dbreak_predict   : the uncertainty in the log break frequency (Hz).
             inject_predict   : the energy injection index (dimensionless).
             dinject_predict  : the uncertainty in the energy injection index (dimensionless).
             remnant_predict  : the remnant fraction, e.g. the fractiononal inactive time (dimensionless).
             dremnant_predict : the uncertainty in the remnant fraction (dimensionless).
             normalisation    : the normalisation factor for correct scaling (dimensionless).
```

### spectral_model
Once you have determined the optimal fit, you might want to construct a model spectrum e.g. to compare the observed and model data, or to simulate the model over a range of frequencies to visualize on a plot. This is performed using the `spectral_model` function which takes the parameters estimated by `spectral_fitter` and simulates the model spectrum. `spectral_model` uses the uncertainties in each free parameter and estimates the uncertainties in the model using a standard Monte-Carlo approach.
```
spectral_model(params : tuple, frequency : (list, np.ndarray), mc_length=500, err_width=2, \
    b_field=None, redshift=None, work_dir=None, write_model=False, save_prefix=None):

    (usage) Uses the optimized parameters to return a 1darray of model flux densities for a given frequency list. 
            Uncertainties on the model are calculated via Monte-Carlo simulation.
    
    Parameters
    ----------
    params        : (tuple) fit_type, break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation
                    fit_type         : the chosen model for fitting.
                    break_predict    : the log break frequency (Hz).
                    dbreak_predict   : the uncertainty in the log break frequency (Hz).
                    inject_predict   : the energy injection index (dimensionless).
                    dinject_predict  : the uncertainty in the energy injection index (dimensionless).
                    remnant_predict  : the remnant fraction, e.g. the fractiononal inactive time (dimensionless).
                    dremnant_predict : the uncertainty in the remnant fraction (dimensionless).
                    normalisation    : the normalisation factor for correct scaling (dimensionless).
    
    frequency     : a list of observed frequencies.
    
    mc_length     : number of MC iterations used for uncertainty estimation
    
    err_width     : The width of the uncertainty envelope in the model
    
    work_dir      : if not None, writes outputs to this directory
    
    write_model   : if True, writes outputs

        
    Returns
    -------
    model_data_tuple : a tuple containing (model_data, err_model_data, model_data_min, model_data_max)
                         model_data     : a list of model flux densities
                         err_model_data : a list of model flux density uncertainties
                         model_data_min : a list of the lower-bound model flux densities
                         model_data_max : a list of the upper-bound model flux densities
```

### spectral_ages
An optional feature of `synchrofit` is to evaluate the spectral age using the parameters estimated by `spectral_fitter`. This is perfomed by the `spectral_ages` function, which is based upon Equation 4 of [Turner et al (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2522T/abstract). 
```
spectral_ages(params : tuple, b_field : float, redshift : float):

    (usage) Derives the total, active and inactive spectral age using the break frequency, quiescent fraction, magnetic field strength and redshift.
    
    parameters
    ----------
    params   : a tuple containing (fit_type, break_frequency, remnant_fraction)
               fit_type         : the spectral model, must be one of [KP, TKP, JP, TJP, CI, TCI].
               break_frequency  : the break frequency in Hz.
               remnant_fraction : the fraction of time spent inactive (if the source is active, set this parameter to zero).
               
    b_field  :  magnetic field strength (T).
    
    redshift :  cosmological redshift (dimensionless).

    Returns
    -------
    spectral_ages : a tuple containing (tau, t_on, t_off)
                    tau   : total spectral age in Myr
                    t_on  : duration of the active phase in Myr
                    t_off : duration of the inactive phase in Myr (returns zero if the source is active)
```
Note, only the CI-off model will return a non-zero value for `t_off`. 

### Private functions
The functions `__spectral_models_standard` and `__spectral_models_tribble` are private functions that contain the standard and Tribble forms of the spectral models described in [Theory](#theory). These functions are not accessed directly in expected workflow for the package.
```
__spectral_models_standard(frequency : (list, np.ndarray), luminosity : (list, np.ndarray), fit_type : str, break_frequency : float,
     injection_index : float, remnant_ratio : float, normalisation : float, bessel_x, bessel_F):
     
    (usage) The standard forms of the JP, KP and CI models.  
    
    Parameters
    ----------
    frequency       : an input list of frequencies (Hz).

    luminosity      : an input list of flux densities (Jy).

    fit_type        : the spectral model, must be one of [KP, JP, CI].

    break_frequency : the break frequency (Hz).

    injection_index : the energy injection index, s. (dimensionless).

    remnant_ratio   : the remnant fraction, e.g. the fractional inactive time (dimensionless).

    normalisation   : the normalisation factor (dimensionless).
        
    Returns
    -------
    luminosity_predict : fitted flux density for given frequency list

    normalisation      : normalisation factor for correct scaling of 
```

`spectral_models_tribble` is setup identical to this, however with the inclusion of the magnetic field strength and redshift, e.g:
```
__spectral_models_tribble(frequency, luminosity, fit_type : str, b_field : float, redshift : float, \
    break_frequency : float, injection_index : float, remnant_ratio : float, normalisation : float, bessel_x, bessel_F)

    b_field         : the magnetic field strength (T)

    redshift        : the cosmological redshift (dimensionless)
```

## Usage
### How do I run synchrofit ?
#### Command-line execution<br />
To run `synchrofit` simply execute the following from terminal: <br />
`synchrofit --data ${data.dat} --fit_type ${fit_type}`. <br />
In this example, `${data.dat}` contains the input spectrum (see [test_spectra.dat](https://github.com/synchrofit/synchrofit/blob/main/example/test_spectra.dat) for an example of the format required), and `${fit_type}` describes the model to be fit (e.g. KP, JP, CI, TKP, TJP, TCI). 

Alternatively, one can manually supply a spectrum by executing the following <br />
`synchrofit --freq f1 f2 fn --flux s1 s2 sn --err_flux es1 es2 esn --fit_type ${fit_type}`. <br />

#### Integrate modules into workflow<br />
To integrate this code into your own workflow, simply import synchrofit into your Python code:<br />
 `from sf import synchrofit`. <br />
 or:<br />
 `from sf.synchrofit import spectral_fitter, spectral_model, spectral_plotter, spectral_ages, spectral_units`<br />

### Example applications for synchrofit
#### I have an integrated radio galaxy spectrum, what should I do ? ###
In this case fitting the standard forms of the Continuous Injection models is most applicable. By default, `--fit_type CI` will fit the spectrum using a CI-off model. If the radio galaxy is **known to be active** the spectrum needs to be modelled using the simpler **CI-on** model. This is done setting `--remnant_range 0`. This will look as follows:<br />
`synchrofit --data ${data_file.dat} --fit_type CI --remnant_range 0` <br />
or as follows if executing the function within your own workflow: <br />
`spectral_fitter($frequency, $luminosity, $dluminosity, CI, remnant_range=0)`

#### I want to model the spectrum of a supernova remnant, what should I do ? ###
In this case fitting the standard forms of the JP or KP models is best, given the supernova remnant is just a shell of impulsively-injected electrons. This will look as follows:<br />
`synchrofit --data ${data_file.dat} --fit_type JP` <br />
or as follows if executing the function within your own workflow: <br />
`spectral_fitter($frequency, $luminosity, $dluminosity, JP)`

#### I want to evaluate the spectral age from my radio spectrum, what should I do ? ###
Firstly, this requires that you provide a value for the magnetic field strength (nT) and a cosmological redshift. In the example below, we evaluate the spectral ages of an inactive (remnant) radio galaxy at redshift `z = 0.2` and with a lobe magnetic field strength of `B = 0.5 nT`:<br />
`synchrofit --data ${data_file.dat} --fit_type CI --age --b_field 0.5e-9 --redshift 0.2` <br />
or as follows if executing the function within your own workflow: <br />
```
params = spectral_fitter($frequency, $luminosity, $dluminosity, CI)
params = (params[0], 10**params[1], params[5])
spectral_age(params, 0.5, 0.2)
```

If you have a spatially-resolved radio spectrum, you may want to consider mapping the age across the lobes. In this case you will need to integrate `synchrofit` into your own workflow by fitting each resolved spectrum yourself (currently `synchrofit` will not do this automatically). Assuming you are able to do this, follow the example above using either the JP or KP model instead. We note however that if the plasma within the lobes is well-mixed, the resolved age map will not give the true spectral age of the source as even the oldest regions will contain relatively young electrons that will dominate the radio spectrum. 

### Default and custom configurations
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
- `--age` If True, determine the spectral age of the source (requires `--b_field` and `--redshift`). Default = False.
- `--b_field` Magnetic field strength in units of T. No default. 
- `--redshift` Cosmological redshift of the source. No default. 

An example of a custom configuration might look as follows: <br />
`synchrofit --data ${data.dat} --fit_type CI --n_breaks 21 --n_injects 21 --n_remnants 21`<br />` --n_iterations 5 --break_range 8 10 --inject_range 2.0 3.0 --mc_length 1000`

Consider loading any custom configuration into `run_synchrofit.sh`, which will allow you to save and store these presets for later re-use. To execute this custom setup, simply run `./run_synchrofit.sh` from the terminal.
