#! /usr/bin/python3

import argparse
import math
import logging
import numpy as np
import pandas as pd
from numba import jit
from scipy.stats import chi2
from matplotlib import pyplot as plt

logging.basicConfig(format="%(levelname)s (%(funcName)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Colors:
    DogderBlue = (30, 144, 255)
    Green = (0,200,0)

def _join(*values):
    return ";".join(str(v) for v in values)

def color_text(s, c, base=30):
    template = "\x1b[{0}m{1}\x1b[0m"
    t = _join(base+8, 2, _join(*c))
    return template.format(t, s)

def spectral_fitter(frequency, luminosity, dluminosity, fit_type, nbreaks, break_range, ninjects, inject_range, nremnants, remnant_range, niterations, workdir, write_model):
    """
    (usage) Finds the optimal fit for a radio spectrum modelled by either the JP, KP or CI model.
    
    parameters
    ----------
    frequency : 1darray
        The input frequency list
    luminosity : 1darray
        The input flux density list
    dluminosity : 1darray
        The uncertainty on the input flux density list
    fit_type : str
        The type of model to fit (JP, KP, KGJP, CI)
    nbreaks : int
        Number of break frequencies used in adaptive grid
    break_range : list
        bounds for the log(break frequency) range
    ninjects : int
        Number of injection indices used in adaptive grid
    inject_range : list
        bounds for the injection index range
    nremnants : int
        Number of  used in adaptive grid
    remnant_range : list
        bounds for the remnant ratio range
    niterations : int
        
    returns
    -------
    luminosity_predict : 1darray
        fitted flux density for given frequency list
    normalisation : float
        normalisation factor for correct scaling
    """
    # check inputs are of correct data types
    if not isinstance(frequency, (list, np.ndarray)) or not isinstance(luminosity, (list, np.ndarray)) or not isinstance(dluminosity, (list, np.ndarray)) or not len(luminosity) == len(frequency) or not len(dluminosity) == len(frequency):
        raise Exception('Frequency, luminosity and uncertainty arrays must be lists or numpy arrays of the same length.')
    if not isinstance(fit_type, str) or not (fit_type == 'CI' or fit_type == 'JP' or fit_type == 'KP'):
        raise Exception('Spectral fit must be either \'CI\', \'JP\' or \'KP\' model.')
    if not isinstance(nbreaks, (int, float)) or not isinstance(ninjects, (int, float)) or not isinstance(nremnants, (int, float)):
        raise Exception('Number of break frequencies, injection indices and remnant ratios must be integers.')
    if isinstance(nbreaks, float):
        nbreaks = int(nbreaks)
    if isinstance(ninjects, float):
        ninjects = int(ninjects)
    if isinstance(nremnants, float):
        nremnants = int(nremnants)
    if not isinstance(break_range, (list, np.ndarray)) or not len(break_range) == 2 or not isinstance(inject_range, (list, np.ndarray)) or not len(inject_range) == 2 or not isinstance(remnant_range, (list, np.ndarray)) or not len(remnant_range) == 2:
        raise Exception('Break frequency, injection indea and remnant ratio arrays must be two element lists or numpy arrays.')

    # set nremnants=1 if JP or KP
    if fit_type == 'JP' or fit_type == 'KP':
        nremnants = 1
        colorstring = color_text("Overriding nremnants=1 for {} model".format(fit_type), Colors.DogderBlue)
        logger.info(colorstring)
    
    #print accepted parameters to terminal
    espace='                      '
    colorstring = color_text("Modelling parameters accepted:", Colors.DogderBlue)
    logger.info(colorstring)
    colorstring=color_text(" {} inject_range = {} \n {} ninjects = {} \n {} nbreaks = {} \n {} break_range = {} \n {} nremnants = {} \n {} remnant_range = {}".format(espace,inject_range, espace, ninjects, espace, nbreaks, espace, break_range, espace, nremnants, espace, remnant_range), Colors.Green)
    print(colorstring)
    
    # convert luminosities and uncertainties to log-scale
    log_luminosity = np.log10(luminosity + 1e-307)
    dlog_luminosity = np.zeros_like(log_luminosity)
    for freqPointer in range(0, len(frequency)):
        if (luminosity[freqPointer] - dluminosity[freqPointer] > 0):
            dlog_luminosity[freqPointer] = (np.log10(luminosity[freqPointer] + dluminosity[freqPointer] + 1e-307) - np.log10(luminosity[freqPointer] - dluminosity[freqPointer] + 1e-307))/2
        else:
            dlog_luminosity[freqPointer] = 1e-307
    
    # read-in integral of BesselK_5/3 datafile
    df_bessel = pd.read_csv('{}/besselK53.txt'.format(workdir), header=None)
    bessel_x, bessel_F = df_bessel[0].values, df_bessel[1].values
    
    # calculate dof for chi-squared functions
    dof = -3
    if (nbreaks <= 2):
        dof = dof + 1 # number of parameters being fitted
    if (ninjects <= 2):
        dof = dof + 1 # number of parameters being fitted
    if (nremnants <= 2):
        dof = dof + 1 # number of parameters being fitted
    for freqPointer in range(0, len(frequency)):
        if (dlog_luminosity[freqPointer] > 0):
            dof = dof + 1
            
    # instantiate variables for adaptive regions
    break_frequency = np.zeros(max(1, nbreaks))
    injection_index = np.zeros(max(1, ninjects))
    remnant_ratio = np.zeros(max(1, nremnants))
    
    # set adaptive regions
    if (nbreaks > 2):
        for breakPointer in range(0, nbreaks):
            break_frequency[breakPointer] = 10**(break_range[0] + breakPointer*(break_range[1] - break_range[0])/(nbreaks - 1))
    else:
        break_frequency[0] = 10**((break_range[0] + break_range[1])/2.)
    if (ninjects > 2):
        for injectPointer in range(0, ninjects):
            injection_index[injectPointer] = inject_range[0] + injectPointer*(inject_range[1] - inject_range[0])/(ninjects - 1)
    else:
        injection_index[0] = (inject_range[0] + inject_range[1])/2
    if (nremnants > 2):
        for remnantPointer in range(0, nremnants):
            remnant_ratio[remnantPointer] = remnant_range[0] + remnantPointer*(remnant_range[1] - remnant_range[0])/(nremnants - 1)
    else:
        remnant_ratio[0] = 0.
        
    # instantiate array to store probabilities from chi-squared statistics
    probability = np.zeros((max(1, nbreaks), max(1, ninjects), max(1, nremnants)))
    normalisation_array = np.zeros((max(1, nbreaks), max(1, ninjects), max(1, nremnants)))
    # find chi-squared statistic for each set of parameters, and iterate through with adaptive mesh
    for iterationPointer in range(0, max(1, niterations)):
        for breakPointer in range(0, max(1, nbreaks)):
            for injectPointer in range(0, max(1, ninjects)):
                for remnantPointer in range(0, max(1, nremnants)):
                    
                    # find spectral fit for current set of parameters
                    normalisation = 0 # specify that fit needs to be scaled
                    luminosity_predict, normalisation = spectral_models(frequency, luminosity, fit_type, break_frequency[breakPointer], injection_index[injectPointer], remnant_ratio[remnantPointer], normalisation, bessel_x, bessel_F)
                    normalisation_array[breakPointer,injectPointer,remnantPointer] = normalisation

                    # calculate chi-squared statistic and probability */                    
                    probability[breakPointer,injectPointer,remnantPointer] = np.nansum(((log_luminosity - np.log10(luminosity_predict + 1e-307))/dlog_luminosity)**2/2.)
                    probability[breakPointer,injectPointer,remnantPointer] = 1 - chi2.cdf(probability[breakPointer,injectPointer,remnantPointer], dof)
       
        # find peak of joint probability distribution
        max_probability = 0.
        max_break, max_inject, max_remnant = 0, 0, 0
        for breakPointer in range(0, max(1, nbreaks)):
            for injectPointer in range(0, max(1, ninjects)):
                for remnantPointer in range(0, max(1, nremnants)):
                    if (probability[breakPointer,injectPointer,remnantPointer] > max_probability):
                        max_probability = probability[breakPointer,injectPointer,remnantPointer]
                        normalisation = normalisation_array[breakPointer,injectPointer,remnantPointer]
                        max_break = breakPointer
                        max_inject = injectPointer
                        max_remnant = remnantPointer
        
        break_predict = break_frequency[max_break]
        inject_predict = injection_index[max_inject]
        remnant_predict = remnant_ratio[max_remnant]
        
        # calculate standard deviation from marginal distributions
        sum_probability = 0
        dbreak_predict = 0
        if (nbreaks > 2):
            for breakPointer in range(0, max(1, nbreaks)):
                sum_probability = sum_probability + probability[breakPointer,max_inject,max_remnant]
                dbreak_predict = dbreak_predict + probability[breakPointer,max_inject,max_remnant]*(np.log10(break_frequency[breakPointer]) - np.log10(break_predict))**2
            dbreak_predict = np.sqrt(dbreak_predict/sum_probability)

        sum_probability = 0
        dinject_predict = 0
        if (ninjects > 2):
            for injectPointer in range(0, max(1, ninjects)):
                sum_probability = sum_probability + probability[max_break,injectPointer,max_remnant]
                dinject_predict = dinject_predict + probability[max_break,injectPointer,max_remnant]*(injection_index[injectPointer] - inject_predict)**2
            dinject_predict = np.sqrt(dinject_predict/sum_probability)

        sum_probability = 0
        dremnant_predict = 0
        if (nremnants > 2):
            for remnantPointer in range(0, max(1, nremnants)):
                sum_probability = sum_probability + probability[max_break,max_inject,remnantPointer]
                dremnant_predict = dremnant_predict + probability[max_break,max_inject,remnantPointer]*(remnant_ratio[remnantPointer] - remnant_predict)**2
            dremnant_predict = np.sqrt(dremnant_predict/sum_probability)
        
        # update adaptive regions
        if (iterationPointer < niterations - 1):
            if (nbreaks > 2):
                for breakPointer in range(0, max(1, nbreaks)):
                    break_frequency[breakPointer] = 10**(np.log10(break_predict) + (breakPointer - (nbreaks - 1)/2)*(break_range[1] - break_range[0])/(nbreaks - 1)/((nbreaks - 1)/2)**(0.5*(iterationPointer + 1)))
            if (ninjects > 2):
                for injectPointer in range(0, max(1, ninjects)):
                    injection_index[injectPointer] = inject_predict + (injectPointer - (ninjects - 1)/2)*(inject_range[1] - inject_range[0])/(ninjects - 1)/((ninjects - 1)/2)**(0.5*(iterationPointer + 1))
            if (nremnants > 2):
                for remnantPointer in range(0, max(1, nremnants)):
                    remnant_ratio[remnantPointer] = remnant_predict + (remnantPointer - (nremnants - 1)/2)*(remnant_range[1] - remnant_range[0])/(nremnants - 1)/((nremnants - 1)/2)**(0.5*(remnantPointer + 1))

    # return set of best-fit parameters with uncertainties
    colorstring = color_text("Optimal parameters estimated for {} model:".format(fit_type), Colors.DogderBlue)
    logger.info(colorstring)
    colorstring = color_text(" {} s = {} +\- {} \n {} break_freq = {} +\- {} \n {} remnant_predict = {} +\- {} ".format(espace, inject_predict, dinject_predict, espace, break_predict, 10**dbreak_predict, espace, remnant_predict, dremnant_predict), Colors.Green)
    print(colorstring)

    # write fitting outputs to file
    if write_model:
        filename = "{}/estimated_params_{}.dat".format(workdir, fit_type)
        colorstring = color_text("Writing fitting outputs to: {}".format(filename), Colors.DogderBlue)
        logger.info(colorstring)
    
        file = open(filename, "w")
        file.write("break_freq, unc_break_freq, inj_index, unc_inj_index, remnant_predict, unc_remnant_predict, normalisation \n")
        file.write("{}, {}, {}, {}, {}, {}, {} \n".format(break_predict, 10**dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation))
        file.close()

    return(break_predict, 10**dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def spectral_models(frequency, luminosity, fit_type, break_frequency, injection_index, remnant_ratio, normalisation, bessel_x, bessel_F):
    """
    (usage) Numerical forms for the JP, KP and CI models.  
    
    parameters
    ----------
    frequency : 1darray
        The input frequency list
    luminosity : 1darray
        The input flux density list
    dluminosity : 1darray
        The uncertainty on the input flux density list
    fit_type : str
        The type of model to fit (JP, KP, KGJP, CI)
    sigma_level : int
        range of the model uncertainty envelope in sigma
    nremnants : int
        The remnant ratio range
    nfreqplots : int
        The number of plotting frequencies
    mcLength : int
        Number of MC iterations
        
    returns
    -------
    luminosity_predict : 1darray
        fitted flux density for given frequency list
    normalisation : float
        normalisation factor for correct scaling
    """ 
    if fit_type == 'JP' or fit_type == 'KP':
        remnant_ratio = 0
    nalpha, nenergiesJP, nenergiesCI = 32, 32, 32 # can be increased for extra precision
    nenergies = nenergiesJP + nenergiesCI
    
    # calculate the best fit to frequency-luminosity data
    luminosity_sum, predict_sum, nfreqs = 0., 0., 0
    luminosity_predict = np.zeros(len(frequency))
    
    for freqPointer in range(0, len(frequency)):
        if (normalisation > 0 or luminosity[freqPointer] > 0): # either valid luminosity data or not fitting
            
            luminosity_predict[freqPointer] = 0. # should already be zero
            
            # integral over pitch angle 
            alpha_min = 0
            alpha_max = np.pi/2 # should be pi, but sin(alpha) is symmetric about pi/2
            for j in range(0, nalpha):
                
                # set up numerical integration
                alpha = ((j + 0.5)/nalpha)*(alpha_max - alpha_min) + alpha_min
                dalpha = (alpha_max - alpha_min)/nalpha
                
                # integrate over energy (as x)
                if (fit_type == 'CI' or fit_type == 'JP'):
                    x_crit = np.log10(frequency[freqPointer]/(break_frequency*np.sin(alpha)))
                elif (fit_type == 'KP'):
                    x_crit = np.log10(frequency[freqPointer]*np.sin(alpha)**3/(break_frequency))
                else:
                    raise Exception('Spectral fit must be either \'CI\', \'JP\' or \'KP\' model.')
                if remnant_ratio > 0:
                    x_crit_star = np.log10(frequency[freqPointer]/(break_frequency*np.sin(alpha))*remnant_ratio**2)
                else:
                    x_crit_star = -307. # close to zero in log space
                x_min = x_crit - 8 # can be increased for extra precision away from break
                x_max = x_crit + 8
                for k in range(0, nenergies):
                    
                    # set up numerical integration, allowing for different spacing above and below x_crit
                    if (k < nenergiesCI):
                        x = (10**((k + 1)*(x_crit - x_min)/nenergiesCI + x_min) + 10**(k*(x_crit - x_min)/nenergiesCI + x_min))/2 # trapezoidal rule
                        dx = 10**((k + 1)*(x_crit - x_min)/nenergiesCI + x_min) - 10**(k*(x_crit - x_min)/nenergiesCI + x_min)
                    else:
                        x = (10**((k + 1 - nenergiesCI)*(x_max - x_crit)/nenergiesJP + x_crit) + 10**((k - nenergiesCI)*(x_max - x_crit)/nenergiesJP + x_crit))/2
                        dx = 10**((k + 1 - nenergiesCI)*(x_max - x_crit)/nenergiesJP + x_crit) - 10**((k - nenergiesCI)*(x_max - x_crit)/nenergiesJP + x_crit)
                    
                    # calculate the spectrum for JP, KP or CI/off models
                    if (x > 10**x_crit):
                        if (fit_type == 'CI'):
                            if remnant_ratio > 0:
                                N_x = x**(-1./2)*((np.sqrt(x) - 10**(x_crit_star/2))**(injection_index - 1) - (np.sqrt(x) - 10**(x_crit/2))**(injection_index - 1))
                            else:
                                N_x = x**((injection_index - 2)/2.)*(1 - x**((1 - injection_index)/2.)*(np.sqrt(x) - 10**(x_crit/2))**(injection_index - 1))
                        elif (fit_type == 'JP' or fit_type == 'KP'):
                            N_x = x**(-1./2)*(np.sqrt(x) - 10**(x_crit/2))**(injection_index - 2)
                    elif (x > 10**x_crit_star): # only CI-off model should meet this condition
                        if (fit_type == 'CI'):
                            if remnant_ratio > 0:
                                N_x = x**(-1./2)*(np.sqrt(x) - 10**(x_crit_star/2))**(injection_index - 1)
                            else:
                                N_x = x**((injection_index - 2)/2.)
                        else:
                            N_x = 0
                    else:
                        N_x = 0
                    
                    # evalute x \int_x^inf BesselK_5/3 using lookup table
                    if (x < bessel_x[0]):
                        F_x = 2.15*x**(1./3)
                    else:
                        if (x > bessel_x[-1]):
                            F_x = 0
                        else:
                            # use bisection method to improve computational efficiency
                            bessla = 0
                            besslb = len(bessel_x) - 1
                            besslc = (bessla + besslb)//2
                            while (bessla < besslb):
                                if (bessel_x[besslc] < x):
                                    bessla = besslc + 1
                                else:
                                    besslb = besslc - 1
                                besslc = (bessla + besslb)//2
                            F_x = bessel_F[besslc]

                    # add contribution to the model spectrum flux
                    if (fit_type == 'CI'):
                        luminosity_predict[freqPointer] = luminosity_predict[freqPointer] + frequency[freqPointer]**(-injection_index/2.)*np.sin(alpha)**((injection_index + 4)/2.)*F_x*N_x*dx*dalpha
                    elif (fit_type == 'JP'):
                        luminosity_predict[freqPointer] = luminosity_predict[freqPointer] + frequency[freqPointer]**((1 - injection_index)/2.)*np.sin(alpha)**((injection_index + 3)/2.)*F_x*N_x*dx*dalpha
                    elif (fit_type == 'KP'):    
                        luminosity_predict[freqPointer] = luminosity_predict[freqPointer] + frequency[freqPointer]**((1 - injection_index)/2.)*np.sin(alpha)**((3*injection_index + 1)/2.)*F_x*N_x*dx*dalpha

            if (normalisation <= 0):
                luminosity_sum = luminosity_sum + np.log10(luminosity[freqPointer] + 1e-307)
                predict_sum = predict_sum + np.log10(luminosity_predict[freqPointer] + 1e-307)
                nfreqs = nfreqs + 1
    
    # calculate constant of proportionality
    if (normalisation <= 0):
        normalisation = 10**((luminosity_sum - predict_sum)/nfreqs)
    
    # record normalised, predicted values for each frequency
    luminosity_predict = normalisation*luminosity_predict
    
    # return outputs
    return luminosity_predict, normalisation

def evaluate_model(frequency, luminosity, dluminosity, fit_type, nbreaks, break_range, ninjects, inject_range, nremnants, remnant_range, nfreqplots, mcLength, sigma_level, niterations, workdir, write_model):
    """
    (usage) Uses the optimized parameters to return a 1darray of model flux densities for a given frequency list. An uncertainty envelope on the model is calculated following an MC approach. 
    
    parameters
    ----------
    frequency : 1darray
        The input frequency list
    luminosity : 1darray
        The input flux density list
    dluminosity : 1darray
        The uncertainty on the input flux density list
    fit_type : str
        The type of model to fit (JP, KP, KGJP, CI)
    sigma_level : int
        range of the model uncertainty envelope in sigma
    nremnants : int
        The remnant ratio range
    nfreqplots : int
        The number of plotting frequencies
    mcLength : int
        Number of MC iterations
        
    returns
    -------
    plotting_frequency : 1darray
        list of frequencies at which to evaluate the model
    Luminosityfit : 1darray
        Peak probable fit of the model
    dLuminosityfit : 1darray
        1-sigma uncertainty on Luminosityfit
    Luminosityfit_lower : 1darray
        Luminosityfit - dLuminosityfit
    Luminosityfit_upper : 1darray
        Luminosityfit + dLuminosityfit
    
    """
    df = pd.read_csv('{}/besselK53.txt'.format(options.workdir), header=None)
    # fit the spectrum for the optimal estimates of the injection index and break frequency
    break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation = spectral_fitter(frequency, luminosity, dluminosity, fit_type, nbreaks, break_range, ninjects, inject_range, nremnants, remnant_range, niterations, workdir, write_model)
    # determine the model for a list of plotting frequencies
    plotting_frequency=np.geomspace(10**(math.floor(np.min(np.log10(frequency)))),10**(math.ceil(np.max(np.log10(frequency)))), num=nfreqplots)
    Luminosityfit, norm = spectral_models(plotting_frequency, np.zeros(len(plotting_frequency)), fit_type, break_predict, inject_predict, remnant_predict, normalisation, df[0].values, df[1].values)
    # simulate a list of injection indices and break frequencies using their gaussian errors 
    break_predict_vec = np.random.normal(break_predict, dbreak_predict, mcLength)
    inject_predict_vec = np.random.normal(inject_predict, dinject_predict, mcLength)
    remnant_predict_vec = np.random.normal(remnant_predict, dremnant_predict, mcLength)
    
    # MC simulate an array of model luminosities
    luminosityArray = np.zeros([mcLength,len(plotting_frequency)])
    for mcPointer in range(0,mcLength):
        fitmc, normmc = spectral_models(plotting_frequency, np.zeros(len(plotting_frequency)), fit_type, break_predict_vec[mcPointer], inject_predict_vec[mcPointer], remnant_predict_vec[mcPointer], normalisation, df[0].values, df[1].values)
        luminosityArray[mcPointer] = (np.asarray(fitmc))
    
    dLuminosityfit=np.zeros([len(plotting_frequency)])
    Luminosityfit_lower=np.zeros([len(plotting_frequency)])
    Luminosityfit_upper=np.zeros([len(plotting_frequency)])
    # at each plotting frequency use the std dev to determine the uncertainty on the model luminosity
    for plotfreqPointer in range(0,len(plotting_frequency)):
        dLuminosityfit[plotfreqPointer] = np.std(luminosityArray.T[plotfreqPointer])
        Luminosityfit_lower[plotfreqPointer] = Luminosityfit[plotfreqPointer] - sigma_level*np.std(luminosityArray.T[plotfreqPointer])
        Luminosityfit_upper[plotfreqPointer] = Luminosityfit[plotfreqPointer] + sigma_level*np.std(luminosityArray.T[plotfreqPointer])
    
    # write fitting outputs to file
    if write_model:
        filename = "{}/modelspectrum_{}.dat".format(workdir, fit_type)
        colorstring = color_text("Writing fitting outputs to: {}".format(filename), Colors.DogderBlue)
        logger.info(colorstring)
    
        file = open(filename, "w")
        file.write("Frequency, {0} Model, unc {0} Model, {0} Model +- {1} sigma, {0} Model +- {1} sigma \n".format(fit_type, sigma_level))
        for i in range(0,len(plotting_frequency)):
            file.write("{}, {}, {}, {}, {} \n".format(plotting_frequency[i], Luminosityfit[i], dLuminosityfit[i], Luminosityfit_lower[i], Luminosityfit_upper[i]))
        file.close()

    return(break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, plotting_frequency, Luminosityfit, dLuminosityfit, Luminosityfit_lower, Luminosityfit_upper)

def make_plot(frequency, luminosity, dluminosity, plotting_frequency, Luminosityfit, dLuminosityfit, Luminosityfit_lower, Luminosityfit_upper, workdir, fit_type, sigma_level=None):
    """
    (usage) Plots the data and optimised model fit and writes to file. 
    
    parameters
    ----------
    frequency : 1darray
        The input frequency list (observed data)
    luminosity : 1darray
        The input flux density list (observed data)
    dluminosity : 1darray
        The uncertainty on the input flux density list (observed data)
    plotting_frequency : 1darray
        List of frequencies at which the model is evaluated
    Luminosityfit : 1darray
        Best-fit model evaluated for plotting_frequency
    dLuminosityfit : 1darray
        Uncertainty on Luminosityfit
    Luminosityfit_lower : 1darray
        Lower bound on Luminosityfit
    Luminosityfit_upper : 1darray
        Upper bound on Luminosityfit
    workdir : str
        Directory to write plot to
    fit_type : str
        Print model type on figure
    sigma_level : int
        Print width of uncertainty envelope on figure
    """
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_axes([0.1,0.1,0.86,0.86])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency / Hz', fontsize=20)
    ax.set_ylabel('Integrated flux density / Jy', fontsize=20)
    ax.set_xlim([10**(math.floor(np.min(np.log10(frequency)))),10**(math.ceil(np.max(np.log10(frequency))))])
    ax.set_ylim([0.2*np.min(luminosity),5*np.max(luminosity)])
    ax.tick_params(axis='both', labelsize=20, which='both', direction='in', length=5, width=2)

    # plot data + model
    ax.plot(plotting_frequency, Luminosityfit, c='C0', label='{} Model'.format(fit_type), zorder=2)
    ax.fill_between(plotting_frequency, Luminosityfit_lower.T, Luminosityfit_upper.T, color='purple', alpha=0.2, label='{} Model $\\pm{}\\sigma$'.format(fit_type, sigma_level), zorder=3)
    ax.scatter(frequency, luminosity, marker='.', c='black', label='Data', zorder=1)
    ax.errorbar(frequency,luminosity,xerr=0,yerr=dluminosity,color='black',capsize=1,linestyle='None',hold=True,fmt='none',alpha=0.9)

    plt.legend(loc='upper right', fontsize=20)
    plt.savefig('{}/{}_fit.png'.format(workdir, fit_type),dpi=400)

def derive_spectra_age(fit_type, vb, T, B, z, dyn_age=None):
    """
    (usage) Derives the total, active and inactive spectral age using the break frequency, quiescent fraction, magnetic field strength and redshift.
    
    parameters
    ----------
    fit_type : str
        The fitted model (JP, KP or CI)
    vb : float
        The break frequency (Hz)
    T : float
        The quiescent fraction, this is remnant_predict from spectral_fitter (dimensionless)
    B : float
        The magnetic field strength (nT)
    z : float
        The cosmological redshift. (dimensionless)

    returns
    -------
    tau : float
        The spectral age of the radio emision
    t_on : float
        Duration of active phase
    t_off : float
        Duration of remnant phase
    """
    # define constants (SI units)
    c = 299792458       # light speed
    me = 9.10938356e-31 # electron mass
    mu0 = 4*np.pi*1e-7  # magnetic permeability of free space
    e = 1.60217662e-19  # charge on electron
    
    if fit_type in ['CI', 'JP']:
        v = ((243*np.pi*(me**5)*(c**2))/(4*(mu0**2)*(e**7)))**(0.5)
    elif fit_type == 'KP':
        v = (1/2.25)*(((243*np.pi*(me**5)*(c**2))/(4*(mu0**2)*(e**7)))**(0.5))
    
    Bic = 0.318*((1+z)**2)*1e-9
    B = B*1e-9
    tau = ((v*(B**(0.5)))/((B**2)+(Bic**2)))*((vb*(1+z))**(-0.5)) # seconds
    tau = tau/(3.154e+13) # Myr
    
    # override spectral age with the dynamical age
    if dyn_age is not None:
        tau = dyn_age
    
    t_on = tau*(1-T)
    t_off = tau - t_on

    espace='                         '
    colorstring = color_text("Spectral ages estimated for {} model:".format(fit_type), Colors.DogderBlue)
    logger.info(colorstring)
    colorstring = color_text(" {} tau = {} Myr \n {} t_on = {} Myr \n {} t_off = {} Myr".format(espace, tau, espace, t_on, espace, t_off), Colors.Green)
    print(colorstring)

def convert_to_native(data, unit):
    """
    (usage) Converts input frequency and flux density into Hz and Jy, respectively. 
    
    parameters
    ----------
    data : 1darray
        Input frequency or flux density data
    unit : str
        The units corresponding to the data

    returns
    -------
    data : 1darray
        Input data in native units
    """
    if unit == 'Hz':
        return(data)
    if unit == 'MHz':
        return(1e+6*data)
    if unit == 'GHz':
        return(1e+9*data)
    if unit == 'Jy':
        return(data)
    if unit == 'mJy':
        return(1e-3*data)
    if unit == 'uJy':
        return(1e-6*data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prefix_chars='-')
    group1 = parser.add_argument_group('Configuration Options')
    group1.add_argument('--workdir', dest='workdir', type=str, help='Path to working directory containing data files. ')
    group1.add_argument("--data", dest='data', type=str, help='Name of .dat file containing measured spectra.')
    group1.add_argument("--freq", dest='freq', type=float, nargs='+', default=None, help='Measured frequencies.')
    group1.add_argument("--flux", dest='flux', type=float, nargs='+', default=None, help='Measured flux densities')
    group1.add_argument("--err_flux", dest='err_flux', type=float, nargs='+', default=None, help='Measured flux density uncertainties')
    group1.add_argument("--freq_unit", dest='freq_unit', type=str, default='Hz', help='Frequency units (default = Hz)')
    group1.add_argument("--flux_unit", dest='flux_unit', type=str, default='Jy', help='Flux density units (default = Jy)')
    
    group2 = parser.add_argument_group('Fitting Options')
    group2.add_argument("--fit_type", dest='fit_type', type=str, default=None, help='Model to fit: JP, KP, CI')
    group2.add_argument("--nbreaks", dest='nbreaks', type=int, default=31, help='Number of break frequencies for adaptive grid')
    group2.add_argument("--ninjects", dest='ninjects', type=int, default=21, help='Number of injection indices for adaptive grid')
    group2.add_argument("--nremnants", dest='nremnants', type=int, default=21, help='')
    group2.add_argument("--niterations", dest='niterations', type=int, default=3, help='')
    group2.add_argument("--break_range", dest='break_range', type=float, nargs='+', default=[8, 11], help='Allowed range for log10(break frequency) in Hz')
    group2.add_argument("--inject_range", dest='inject_range', type=float, nargs='+', default=[2.01, 2.99], help='Allowed range for energy injection index "s"')
    group2.add_argument("--remnant_range", dest='remnant_range', type=float, nargs='+', default=[0, 1], help='Allowed range for remnant fraction')

    group3 = parser.add_argument_group('Output Model Options')
    group3.add_argument("--nfreqplots", dest='nfreqplots', type=int, default=100, help='Number of plotting frequencies.')
    group3.add_argument("--mcLength", dest='mcLength', type=int, default=1000, help='Number of MC iterations.')
    group3.add_argument("--sigma_level", dest='sigma_level', type=int, default=2, help='Width of uncertainty envelope in sigma')

    group4 = parser.add_argument_group('Extra fitting options')
    group4.add_argument('--plot', dest='plot', action='store_true',default=False, help='Plot data and optimized model fit.')
    group4.add_argument('--write_model', dest='write_model', action='store_true', default=False, help='Write model and fitting outputs to file. Requires --workdir to be specfied.')

    group5 = parser.add_argument_group('Spectral age options')
    group5.add_argument('--age', dest='age', action='store_true', default=False, help='Determine spectral age from fit and B-field assumption. (Default = False). Requires --bfield')
    group5.add_argument('--bfield', dest='bfield', type=float, help='Magnetic field strength.')
    group5.add_argument('--z', dest='z', type=float, help='Cosmological redshift of source. ')
    options = parser.parse_args()

    # Read in input data and convert to native units. 
    if options.data:
        df_data = pd.read_csv('{}/{}'.format(options.workdir, options.data))
        frequency = convert_to_native(df_data.iloc[:,1].values, options.freq_unit)
        luminosity = convert_to_native(df_data.iloc[:,2].values, options.flux_unit)
        dluminosity = convert_to_native(df_data.iloc[:,3].values, options.flux_unit)
    elif options.freq:
        frequency = convert_to_native(np.asarray(options.freq), options.freq_unit)
        luminosity = convert_to_native(np.asarray(options.flux), options.flux_unit)
        dluminosity = convert_to_native(np.asarray(options.err_flux), options.flux_unit)
    colorstring = color_text("Succesfully read in input data", Colors.DogderBlue)
    print("INFO (main): {}".format(colorstring))
    espace='            '
    colorstring = color_text('{} Frequency (Hz) = {} \n {} flux_density (Jy) = {} \n {} err_flux_density (Jy) = {}'.format(espace, frequency, espace, luminosity, espace, dluminosity), Colors.Green)
    print(colorstring)

    break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, plotting_frequency, Luminosityfit, dLuminosityfit, Luminosityfit_lower, Luminosityfit_upper = evaluate_model(frequency, luminosity, dluminosity, options.fit_type, options.nbreaks, options.break_range, options.ninjects, options.inject_range, options.nremnants, options.remnant_range, options.nfreqplots, options.mcLength, options.sigma_level, options.niterations, options.workdir, options.write_model)

    if options.plot:
        make_plot(frequency, luminosity, dluminosity, plotting_frequency, Luminosityfit, dLuminosityfit, Luminosityfit_lower, Luminosityfit_upper, options.workdir, options.fit_type, options.sigma_level)
    
    if options.age:
        derive_spectra_age(options.fit_type, break_predict, remnant_predict, options.bfield, options.z)