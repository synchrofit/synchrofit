#! /usr/bin/python3

from numba import jit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import chi2
import logging
import argparse

logging.basicConfig(format="%(levelname)s (%(funcName)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Colors:
    Yellow = (255, 255, 0)
    MediumPurple = (147, 112, 219)
    DogderBlue = (30, 144, 255)
    MediumSpringGreen = (0, 250, 154)
    Orange = (255, 165, 0)

def _join(*values):
    return ";".join(str(v) for v in values)

def color_text(s, c, base=30):
    template = "\x1b[{0}m{1}\x1b[0m"
    t = _join(base+8, 2, _join(*c))
    return template.format(t, s)

def spectral_fitter(frequency, luminosity, dluminosity, fit_type, nbreaks=31, break_range=[8,11], ninjects=21, inject_range=[2.01,2.99], nremnants=21, remnant_range=[0,1], niterations=3):
    """
    (usage) Fits a radio spectrum.
    
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
    break_range : list
        bounds for the log(break frequency) range
    ninjects : int
    inject_range : list
        bounds for the injection index range
    nremnants : int
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

    if fit_type == 'JP' or fit_type == 'KP':
        nremnants = 1
    
    # convert luminosities and uncertainties to log-scale
    log_luminosity = np.log10(luminosity + 1e-307)
    dlog_luminosity = np.zeros_like(log_luminosity)
    for freqPointer in range(0, len(frequency)):
        if (luminosity[freqPointer] - dluminosity[freqPointer] > 0):
            dlog_luminosity[freqPointer] = (np.log10(luminosity[freqPointer] + dluminosity[freqPointer] + 1e-307) - np.log10(luminosity[freqPointer] - dluminosity[freqPointer] + 1e-307))/2
        else:
            dlog_luminosity[freqPointer] = 1e-307
    
    # read-in integral of BesselK_5/3 datafile
    df_bessel = pd.read_csv('{}/besselK53.txt'.format(options.path), header=None)
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
                dremnant_predict = dinject_predict + probability[max_break,max_inject,remnantPointer]*(remnant_ratio[remnantPointer] - remnant_predict)**2
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
    colorstring = color_text("Optimal parameters estimated for {} model. s = {} +\- {}, break_freq = {} +\- {}".format(fit_type, inject_predict, dinject_predict, break_predict, dbreak_predict), Colors.DogderBlue)
    logger.info(colorstring)
    return break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def spectral_models(frequency, luminosity, fit_type, break_frequency, injection_index, remnant_ratio, normalisation, bessel_x, bessel_F):
    """
    (usage) Fits a radio spectrum and uses the modelled injection index and break frequency to return a list of modelled flux densities. An uncertainty envelope on the model is calculated following an MC approach. 
    
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

def evaluate_model(frequency, luminosity, dluminosity, fit_type, nremnants, nfreqplots, mcLength, sigma_level):
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
    
    # fit the spectrum for the optimal estimates of the injection index and break frequency
    break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation = spectral_fitter(frequency, luminosity, dluminosity, fit_type, nremnants=nremnants)
    # determine the model for a list of plotting frequencies
    plotting_frequency=np.geomspace(5e+7,5e+10,num=nfreqplots)
    Luminosityfit, norm = spectral_models(plotting_frequency, np.zeros(len(plotting_frequency)), fit_type, break_predict, inject_predict, remnant_predict, normalisation, df[0].values, df[1].values)
    colorstring = color_text("freq_model = {}".format(plotting_frequency), Colors.DogderBlue)
    logger.info(colorstring)
    colorstring = color_text("flux_model = {}".format(Luminosityfit), Colors.DogderBlue)
    logger.info(colorstring)
    # simulate a list of injection indices and break frequencies using their gaussian errors 
    break_predict_vec = np.random.normal(break_predict, dbreak_predict, mcLength)
    inject_predict_vec = np.random.normal(inject_predict, dinject_predict, mcLength)
    
    # MC simulate an array of model luminosities
    luminosityArray = np.zeros([mcLength,len(plotting_frequency)])
    for mcPointer in range(0,mcLength):
        fitmc, normmc = spectral_models(plotting_frequency, np.zeros(len(plotting_frequency)), fit_type, break_predict_vec[mcPointer], inject_predict_vec[mcPointer], remnant_predict, normalisation, df[0].values, df[1].values)
        luminosityArray[mcPointer] = (np.asarray(fitmc))
    
    dLuminosityfit=np.zeros([len(plotting_frequency)])
    Luminosityfit_lower=np.zeros([len(plotting_frequency)])
    Luminosityfit_upper=np.zeros([len(plotting_frequency)])
    # at each plotting frequency use the std dev to determine the uncertainty on the model luminosity
    for plotfreqPointer in range(0,len(plotting_frequency)):
        dLuminosityfit[plotfreqPointer] = np.std(luminosityArray.T[plotfreqPointer])
        Luminosityfit_lower[plotfreqPointer] = Luminosityfit[plotfreqPointer] - sigma_level*np.std(luminosityArray.T[plotfreqPointer])
        Luminosityfit_upper[plotfreqPointer] = Luminosityfit[plotfreqPointer] + sigma_level*np.std(luminosityArray.T[plotfreqPointer])
    colorstring = color_text("err_flux_model = {}".format(dLuminosityfit), Colors.DogderBlue)
    logger.info(colorstring)
    
    return(plotting_frequency, Luminosityfit, dLuminosityfit, Luminosityfit_lower, Luminosityfit_upper)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prefix_chars='-')
    group1 = parser.add_argument_group('Configuration Options')
    group1.add_argument('--path', dest='path', type=str, help='Path to folder containing data files. ')
    group1.add_argument("--data", dest='data', help='Name of .dat file containing measured spectra.')
    group1.add_argument("--freq", dest='freq', help='Measured frequencies.')
    group1.add_argument("--flux", dest='flux', help='Measured flux densities')
    group1.add_argument("--err_flux", dest='err_flux', help='Measured flux density uncertainties')
    group2 = parser.add_argument_group('Fitting Options')
    group2.add_argument("--fit_type", dest='fit_type', help='Model to fit: JP, KP, CI')
    group2.add_argument("--nfreqplots", dest='nfreqplots', type=int, default=100, help='Number of plotting frequencies.')
    group2.add_argument("--mcLength", dest='mcLength', type=int, default=1000, help='Number of MC iterations.')
    group2.add_argument("--sigma_level", dest='sigma_level', type=int, default=2, help='Width of uncertainty envelope in sigma')
    options = parser.parse_args()

    df = pd.read_csv('{}/besselK53.txt'.format(options.path), header=None)
    df_data = pd.read_csv('{}/{}'.format(options.path, options.data))

    frequency = df_data.iloc[:,1].values
    luminosity = df_data.iloc[:,2].values
    dluminosity = df_data.iloc[:,3].values
    
    nremnants = 1
    plotting_frequency, Luminosityfit, dLuminosityfit, Luminosityfit_lower, Luminosityfit_upper = evaluate_model(frequency, luminosity, dluminosity, options.fit_type, nremnants, options.nfreqplots, options.mcLength, options.sigma_level)

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10, 10))
    fig.subplots_adjust(hspace=0)

    axs.scatter(frequency, luminosity, marker='.', c='black', label='data')
    axs.errorbar(frequency,luminosity,xerr=0,yerr=dluminosity,color='black',capsize=1,linestyle='None',hold=True,fmt='none',alpha=0.9)

    axs.plot(plotting_frequency, Luminosityfit, c='C0', label='{} model'.format(options.fit_type))
    axs.fill_between(plotting_frequency, Luminosityfit_lower.T, Luminosityfit_upper.T, color='purple', alpha=0.2, label='model $\\pm2\\sigma$')

    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlim([1e8,1e10])
    axs.set_ylim([0.2*np.min(luminosity),5*np.max(luminosity)])
    axs.set_xlabel('Frequency / GHz')
    axs.set_ylabel('Integrated flux density / Jy')
    plt.legend(loc='upper right')
    plt.savefig('/home/sputnik/Downloads/modelerr.png',dpi=400)