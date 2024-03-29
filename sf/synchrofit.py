#!/usr/bin/env python

#TODO inspect bug where n_remnants 1 sets the remnant ratio to 0.5: #DONE
#TODO look into quantiles rather than gaussian errors
#TODO output histograms of the parameters fit during the grid

__author__ = "Benjamin Quici, Ross J. Turner"
__date__ = "25/02/2021"

import numpy as np
import math
import pylab

from matplotlib import pyplot as plt
from scipy.stats import chi2

import multiprocessing
import time

from matplotlib.backends.backend_pdf import PdfPages

# synchrofit imports
from sf.spectral_models import __spectral_models_standard, __spectral_models_tribble
from sf.helpers import Colors, color_text, logger, Const, CheckFunctionInputs

from scipy import optimize

def gaussian(x, mean, stddev):
    return 1 * np.exp(-((x - mean) / np.sqrt(2) / stddev)**2)

def ModelUncertainties(params, prob, maxval, plotmin, plotmax, nplots):
    popt, _ = optimize.curve_fit(f=gaussian, xdata=params, ydata=prob, p0=[maxval, 0.1])
    mean, stddev = popt
    x = np.linspace(plotmin, plotmax, nplots)
    f = gaussian(x, *popt)
    return x, f, mean, stddev

class Params():
    
    """
    Template to store outputs from spectral_fitter
    """
    
    def __init__(self, fit_type, break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation):
        """
        """
        self.fit_type = fit_type
        self.break_predict = break_predict
        self.dbreak_predict = dbreak_predict
        self.inject_predict = inject_predict
        self.dinject_predict = dinject_predict
        self.remnant_predict = remnant_predict
        self.dremnant_predict = dremnant_predict
        self.normalisation = normalisation


def __simulate_sf(fit_type, 
                frequency,
                break_frequency,
                injection_index,
                remnant_ratio,
                normalisation,
                b_field=None,
                redshift=None):
    """
    """

    # find spectral fit for current set of parameters
    
    if fit_type in ['JP', 'KP', 'CI']:
        luminosity_predict = __spectral_models_standard(frequency, 
                                                        np.zeros(len(frequency)), 
                                                        fit_type, 
                                                        break_frequency, 
                                                        injection_index, 
                                                        remnant_ratio, 
                                                        normalisation)[0]
    
    elif fit_type in ['TJP', 'TKP', 'TCI']:
        luminosity_predict = __spectral_models_tribble(frequency, 
                                                        np.zeros(len(frequency)), 
                                                        fit_type, 
                                                        b_field, 
                                                        redshift, 
                                                        break_frequency, 
                                                        injection_index, 
                                                        remnant_ratio, 
                                                        normalisation)[0]

    return luminosity_predict


def __compute_sf(fit_type, 
                frequency,
                log_luminosity,
                dlog_luminosity,
                dof,
                break_frequency,
                injection_index,
                remnant_ratio,
                b_field=None,
                redshift=None):
    """
    """

    # find spectral fit for current set of parameters
    
    normalisation = 0 # specify that fit needs to be scaled

    if fit_type in ['JP', 'KP', 'TJP', 'TKP']:
        k = 3
    elif fit_type in ['CI', 'TCI']:
        k = 4
    
    if fit_type in ['JP', 'KP', 'CI']:
        luminosity_predict, normalisation = __spectral_models_standard(frequency, 
                                                                        10**log_luminosity, 
                                                                        fit_type, 
                                                                        break_frequency, 
                                                                        injection_index, 
                                                                        remnant_ratio, 
                                                                        normalisation)
    
    elif fit_type in ['TJP', 'TKP', 'TCI']:
        luminosity_predict, normalisation = __spectral_models_tribble(frequency, 
                                                                        10**log_luminosity, 
                                                                        fit_type, 
                                                                        b_field, 
                                                                        redshift, 
                                                                        break_frequency, 
                                                                        injection_index, 
                                                                        remnant_ratio, 
                                                                        normalisation)

    """
    # calculate chisquare on spectral fit
    chi = np.nansum(((log_luminosity - np.log10(luminosity_predict + 1e-307))/dlog_luminosity)**2)
    # convert chisquare statistic into probability
    prob = 1 - chi2.cdf(chi, dof)
    return np.array([normalisation, chi, prob])
    """

    # calc prod_double
    prob_double = 1./np.sqrt(2*np.pi*dlog_luminosity**2)*np.exp(-(np.log10(luminosity_predict + 1e-307) - log_luminosity)**2/(2*dlog_luminosity**2))
    L_double = np.prod(prob_double)
    AIC_double = 2*(k) - 2*np.log(L_double)

    return np.array([normalisation, AIC_double])


def spectral_units(data, unit):
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
        colorstring = color_text("Input frequency registered as {}. No scaling required.".format(unit), Colors.DodgerBlue)
        logger.info(colorstring)
        return(data)
    if unit == 'MHz':
        colorstring = color_text("Input frequency registered as {}. Scaling by 10^6.".format(unit), Colors.DodgerBlue)
        logger.info(colorstring)
        return(1e+6*data)
    if unit == 'GHz':
        colorstring = color_text("Input frequency registered as {}. Scaling by 10^9.".format(unit), Colors.DodgerBlue)
        logger.info(colorstring)
        return(1e+9*data)
    if unit == 'Jy':
        colorstring = color_text("Input flux density registered as {}. No scaling required.".format(unit), Colors.DodgerBlue)
        logger.info(colorstring)
        return(data)
    if unit == 'mJy':
        colorstring = color_text("Input flux density registered as {}. Scaling by 10^(-3).".format(unit), Colors.DodgerBlue)
        logger.info(colorstring)
        return(1e-3*data)
    if unit == 'uJy':
        colorstring = color_text("Input flux density registered as {}. Scaling by 10^(-6).".format(unit), Colors.DodgerBlue)
        logger.info(colorstring)
        return(1e-6*data)


def spectral_fitter(frequency, 
                    luminosity, 
                    dluminosity,
                    fit_type, 
                    n_breaks=31, 
                    break_range=[8,11], 
                    n_injects=31, 
                    inject_range=[2.01,2.99], 
                    n_remnants=31, 
                    remnant_range=[0,1], 
                    n_iterations=3, 
                    search_space='prob',
                    b_field=None, 
                    redshift=None, 
                    write_model=False, 
                    work_dir=None, 
                    save_prefix=None):
    """
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
    """
    
    # ensure inputs have the correct type and sensible values
    frequency, luminosity, dluminosity, inject_range, break_range, remnant_range, n_injects, n_breaks, n_remnants = \
        CheckFunctionInputs.spectral_fitter(frequency,
        luminosity,
        dluminosity,
        fit_type,
        n_breaks,
        break_range,
        n_injects,
        inject_range,
        n_remnants,
        remnant_range,
        n_iterations,
        b_field,
        redshift)
    

    """
    Convert luminosities and uncertainties to log-scale
    """
    log_luminosity = np.log10(luminosity + 1e-307)
    dlog_luminosity = np.zeros_like(log_luminosity)
    for freqPointer in range(0, len(frequency)):
        if (luminosity[freqPointer] - dluminosity[freqPointer] > 0):
            dlog_luminosity[freqPointer] = (np.log10(luminosity[freqPointer] + dluminosity[freqPointer] + 1e-307) - np.log10(luminosity[freqPointer] - dluminosity[freqPointer] + 1e-307))/2
        else:
            dlog_luminosity[freqPointer] = 1e-307
    

    """
    Calculate dof for chi-squared functions
    """
    dof = -3
    if (n_breaks <= 2):
        dof = dof + 1 # number of parameters being fitted
    if (n_injects <= 2):
        dof = dof + 1 # number of parameters being fitted
    if (n_remnants <= 2):
        dof = dof + 1 # number of parameters being fitted
    for freqPointer in range(0, len(frequency)):
        if (dlog_luminosity[freqPointer] > 0):
            dof = dof + 1


    """
    Set adaptive regions
    """
    if (n_breaks >= 2):
        break_frequency = np.linspace(break_range[0], break_range[1], n_breaks, endpoint=True)
    else:
        break_frequency = np.zeros(1)
        break_frequency[0] = (break_range[0] + break_range[1])/2.
    if (n_injects >= 2):
        injection_index = np.linspace(inject_range[0], inject_range[1], n_injects, endpoint=True)
    else:
        injection_index = np.zeros(1)
        injection_index[0] = (inject_range[0] + inject_range[1])/2
    if (n_remnants >= 2):
        remnant_ratio = np.linspace(remnant_range[0], remnant_range[1], n_remnants, endpoint=True)
    else:
        remnant_ratio = np.zeros(1)
        remnant_ratio[0] = (remnant_range[0] + remnant_range[1])/2


    """
    Instantiate arrays to store fitting outputs
    """
    # length of meshgrid
    meshlength = n_breaks*n_injects*n_remnants
    # store chi-squared statistic on each fit for each adaptive mesh.
    adaptive_chisquare = np.zeros((max(1, n_iterations), meshlength))
    # store probabilities from chi-squared statistics for each adaptive mesh.
    adaptive_probability = np.zeros((max(1, n_iterations), meshlength))
    # store normalisation on each spectral fit
    adaptive_normalisation = np.zeros((max(1, n_iterations), meshlength))

    # store break frequency marginal dist.
    adaptive_break_marginal = np.zeros((max(1, n_iterations), n_breaks))
    # store injection index marginal dist.
    adaptive_inject_marginal = np.zeros((max(1, n_iterations), n_injects))
    # store remnant ratio marginal dist.
    adaptive_remnant_marginal = np.zeros((max(1, n_iterations), n_remnants))
    

    # store discretized list of break frequencies
    adaptive_break_frequency = np.zeros((max(1, n_iterations), n_breaks))
    # store discretized list of injection indices
    adaptive_injection_index = np.zeros((max(1, n_iterations), n_injects))
    # store discretized list of remnant ratios
    adaptive_remnant_ratio = np.zeros((max(1, n_iterations), n_remnants))

    # store most probable (peak) break frequency
    adaptive_break_predict = np.zeros(max(1, n_iterations))
    # store most probable (peak) injection index
    adaptive_inject_predict = np.zeros(max(1, n_iterations))
    # store most probable (peak) remnant ratio
    adaptive_remnant_predict = np.zeros(max(1, n_iterations))

    # store indices corresponding to peak break frequency
    adaptive_max_break = np.zeros(max(1, n_iterations))
    # store indices corresponding to peak injection index
    adaptive_max_inject = np.zeros(max(1, n_iterations))
    # store indices corresponding to peak remnant ratio
    adaptive_max_remnant = np.zeros(max(1, n_iterations))

    pdfs = PdfPages('Marginal_distributions.pdf')
    figv = plt.figure(figsize=(10,10))
    axv = figv.add_axes([0.05,0.05,0.9,0.9])
    figs = plt.figure(figsize=(12,8))
    axs = figs.add_axes([0.05,0.05,0.9,0.9])
    figT = plt.figure(figsize=(12,8))
    axT = figT.add_axes([0.05,0.05,0.9,0.9])
    """
    Iterate through adaptive mesh
    """
    for iterationPointer in range(0, max(1, n_iterations)):
        """
        Setup parallelisation
        """

        # store unique discretized list in each parameter
        unique_parameters = [break_frequency, injection_index, remnant_ratio]

        # instantiate list of pointers for each free parameter
        breakPointers = np.arange(max(1, n_breaks))
        injectPointers = np.arange(max(1, n_injects))
        remnantPointers = np.arange(max(1, n_remnants))

        # create meshgrid from each pointer list
        meshPointers = np.meshgrid(breakPointers, injectPointers, remnantPointers)

        # instantiate matrix to store unique value of each parameter
        mesh_parameters = np.zeros([len(meshPointers[0].flatten()), len(meshPointers)])

        # store mesh parameters in array
        for meshPointer in range(len(meshPointers)):
            uniqueparam = unique_parameters[meshPointer]
            indices = meshPointers[meshPointer].flatten()
            mesh_parameters[:,meshPointer] = uniqueparam[indices]

        # get the mesh list in each parameter 
        meshbreak = mesh_parameters[:,0]
        meshinject = mesh_parameters[:,1]
        meshremnant = mesh_parameters[:,2]

        # append list of arguments to parse to multiprocessed function
        args = []
        for ii in range(len(meshPointers[0].flatten())):
            break_ = meshbreak[ii]
            inject_ = meshinject[ii]
            remnant_ = meshremnant[ii]
            args.append([fit_type, frequency, log_luminosity, dlog_luminosity, dof, 10**break_, inject_, remnant_, b_field, redshift])


        """
        Pool computation of spectral fits over multiple cores
        """
        ncpus = 8
        with multiprocessing.Pool(processes=ncpus, maxtasksperchild=None) as pool:
            sf_stats_matrix = pool.starmap(__compute_sf, args)
            sf_stats_matrix = np.asarray(sf_stats_matrix)


        """
        Find optimal spectral fit
        """
        # # unpack normalisation, chisquare and probability vectors
        # normalisation_vector = sf_stats_matrix[:,0]
        # chisquare_vector = sf_stats_matrix[:,1]
        # probability_vector = sf_stats_matrix[:,2]

        # # set initial max prob and min chi
        # max_probability = 1e-99
        # min_chisquare = 1e+99

        # # iterate through to find optimal spectral fit
        # for Pointer in range(len(probability_vector)):
        #     # find minima of joint chisquare distribution
        #     if search_space == 'chi':
        #         if chisquare_vector[Pointer] < min_chisquare:
        #             min_chisquare = chisquare_vector[Pointer]
        #             normalisation = normalisation_vector[Pointer]
        #             bestPointer = Pointer
        #     elif search_space == 'prob':
        #         # find peak of joint probability distribution
        #         if probability_vector[Pointer] > max_probability: 
        #             max_probability = probability_vector[Pointer]
        #             normalisation = normalisation_vector[Pointer]
        #             bestPointer = Pointer

        # break_predict = meshbreak[bestPointer]
        # inject_predict = meshinject[bestPointer]
        # remnant_predict = meshremnant[bestPointer]

        """
        Find optimal spectral fit
        """
        # unpack normalisation, chisquare and probability vectors
        normalisation_vector = sf_stats_matrix[:,0]
        aic_vector = sf_stats_matrix[:,1]
        # lum_pred_vector = sf_stats_matrix[:,2]

        normalisation_vector = sf_stats_matrix[:,0]
        aic_vector = sf_stats_matrix[:,1]

        aic_best = np.nanmin(aic_vector)

        break_predict = meshbreak[np.where(aic_vector==aic_best)][0]
        inject_predict = meshinject[np.where(aic_vector==aic_best)][0]
        remnant_predict = meshremnant[np.where(aic_vector==aic_best)][0]
        normalisation = normalisation_vector[np.where(aic_vector==aic_best)][0]
        
        probability_vector = np.exp((aic_best - aic_vector)/2.)


        """
        Find standard deviation in each marginal distribution
        """
        if search_space == 'prob':
            weights = probability_vector
        elif search_space == 'chi':
            weights = chisquare_vector
        
        # calc std dev in break frequency
        dbreak_predict = 0
        weights_ = weights[np.where((meshinject==inject_predict) & (meshremnant==remnant_predict))]
        if len(weights_) > 2: 
            for ii in range(len(weights_)):
                dbreak_predict = dbreak_predict + weights_[ii]*((break_frequency[ii] - break_predict)**2)
            dbreak_predict = np.sqrt(dbreak_predict/np.nansum(weights_))

        # calc std dev in injection index
        dinject_predict = 0
        weights_ = weights[np.where((meshbreak==break_predict) & (meshremnant==remnant_predict))]
        if len(weights_) > 2: 
            for ii in range(len(weights_)):
                dinject_predict = dinject_predict + weights_[ii]*((injection_index[ii] - inject_predict)**2)
            dinject_predict = np.sqrt(dinject_predict/np.nansum(weights_))

        # calc std dev in remnant ratio
        figtest = plt.figure()
        axtest = figtest.add_axes([0.05,0.05,0.9,0.9])
        dremnant_predict = 0
        weights_ = weights[np.where((meshinject==inject_predict) & (meshbreak==break_predict))]
        if len(weights_) > 2: 
            for ii in range(len(weights_)):
                dremnant_predict = dremnant_predict + weights_[ii]*((remnant_ratio[ii] - remnant_predict)**2)
            dremnant_predict = np.sqrt(dremnant_predict/np.nansum(weights_))
        axtest.plot(remnant_ratio, weights_)
        axtest.plot([remnant_predict,remnant_predict],[np.nanmin(weights_), np.nanmax(weights_)], color='black')


        """
        Create marginal distribution in each parameter
        """
        create_marginal = False
        if create_marginal is True:
            marginal_break_prob = np.empty(n_breaks)
            if n_breaks > 2:
                for breakPointer in breakPointers:
                    break_ = break_frequency[breakPointer]
                    probability_ = probability_vector[np.where(meshbreak==break_)]
                    marginal_break_prob[breakPointer] = np.nanmax(probability_)
                axv.plot(break_frequency, marginal_break_prob, color='C{}'.format(iterationPointer))
                x, f, mean, dbreak_predict = ModelUncertainties(break_frequency, marginal_break_prob, break_predict, break_frequency[0], break_frequency[-1], 100)
                axv.plot(x, f, color='C{}'.format(iterationPointer), ls=':', lw=2)
            adaptive_break_marginal[iterationPointer,:] = marginal_break_prob
            adaptive_break_frequency[iterationPointer,:] = break_frequency

            marginal_inject_prob = np.empty(n_injects)
            if n_injects > 2:
                for injectPointer in injectPointers:
                    inject_ = injection_index[injectPointer]
                    probability_ = probability_vector[np.where(meshinject==inject_)]
                    marginal_inject_prob[injectPointer] = np.nanmax(probability_)
                axs.plot(injection_index, marginal_inject_prob, color='C{}'.format(iterationPointer))
            adaptive_inject_marginal[iterationPointer,:] = marginal_inject_prob
            adaptive_injection_index[iterationPointer,:] = injection_index

            marginal_remnant_prob = np.empty(n_remnants)
            if n_remnants > 2:
                for remnantPointer in remnantPointers:
                    remnant_ = remnant_ratio[remnantPointer]
                    probability_ = probability_vector[np.where(meshremnant==remnant_)]
                    marginal_remnant_prob[remnantPointer] = np.nanmax(probability_)
                axT.plot(remnant_ratio, marginal_remnant_prob, color='C{}'.format(iterationPointer))
                x, f, mean, dremnant_predict = ModelUncertainties(remnant_ratio, marginal_remnant_prob, remnant_predict, remnant_ratio[0], remnant_ratio[-1], 100)
                axT.plot(x, f, color='C{}'.format(iterationPointer), ls=':', lw=2)
            adaptive_remnant_marginal[iterationPointer,:] = marginal_remnant_prob
            adaptive_remnant_ratio[iterationPointer,:] = remnant_ratio


        """
        Update adaptive regions
        """
        if (iterationPointer < n_iterations - 1):
            if (n_breaks > 2):
                dbreak_frequency = (break_frequency[-1] - break_frequency[0])/np.sqrt(n_breaks)
                break_frequency = np.linspace(max(break_range[0], break_predict - dbreak_frequency), min(break_range[1], break_predict + dbreak_frequency), n_breaks, endpoint=True)
                logger.info(color_text('Current best break-frequency: {}'.format(break_predict), Colors.DodgerBlue))
                logger.info(color_text('Updated break_frequency range to: [{}, {}]'.format(break_frequency[0], break_frequency[-1]), Colors.DodgerBlue))

            if (n_injects > 2):
                dinjection_index = (injection_index[-1] - injection_index[0])/np.sqrt(n_injects)
                injection_index = np.linspace(max(inject_range[0], inject_predict - dinjection_index), min(inject_range[1], inject_predict + dinjection_index), n_injects, endpoint=True)
                logger.info(color_text('Current best injection-index: {}'.format(inject_predict), Colors.DodgerBlue))
                logger.info(color_text('Updated injection_index range to: [{}, {}]'.format(injection_index[0], injection_index[-1]), Colors.DodgerBlue))

            if (n_remnants > 2):
                dremnant_ratio = (remnant_ratio[-1] - remnant_ratio[0])/np.sqrt(n_remnants)
                remnant_ratio = np.linspace(max(remnant_range[0], remnant_predict - dremnant_ratio), min(remnant_range[1], remnant_predict + dremnant_ratio), n_remnants, endpoint=True)
                logger.info(color_text('Current best remnant-ratio: {}'.format(remnant_predict), Colors.DodgerBlue))
                logger.info(color_text('Updated remnant_ratio range to: [{}, {}]'.format(remnant_ratio[0], remnant_ratio[-1]), Colors.DodgerBlue))


    # return set of best-fit parameters with uncertainties
    espace='                      '
    colorstring = color_text("Free parameters optimised for {} model:".format(fit_type), Colors.DodgerBlue)
    logger.info(colorstring)
    colorstring = color_text(" {} s = {} +\- {} \n {} log_break_freq = {} +\- {} \n {} remnant_fraction = {} +\- {} ".format(espace, inject_predict, dinject_predict, espace, break_predict, dbreak_predict, espace, remnant_predict, dremnant_predict), Colors.Green)
    print(colorstring)


    # write fitting outputs to file
    if write_model is True:
        if fit_type is not None:
            filename = "{}_model_params.dat".format(fit_type)
        else:
            filename = "model_params.dat"
        if save_prefix is not None:
            filename = '{}_{}'.format(save_prefix, filename)
        else: 
            filename = filename
        if work_dir is not None:
            savename = "{}/{}".format(work_dir, filename)
        else:
            savename = filename
        colorstring = color_text("Writing fitting outputs to: {}".format(savename), Colors.DodgerBlue)
        logger.info(colorstring)
    
        file = open(savename, "w")
        file.write("log_break_freq, unc_log_break_freq, inj_index, unc_inj_index, remnant_fraction, unc_remnant_fraction, normalisation \n")
        file.write("{}, {}, {}, {}, {}, {}, {} \n".format(break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation))
        file.close()
    pdfs.savefig(figv)
    pdfs.savefig(figs)
    pdfs.savefig(figT)
    pdfs.close()

    params = fit_type, break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation
    marginal_distributions = [adaptive_break_marginal, adaptive_inject_marginal, adaptive_remnant_marginal]
    discretized_parameters = [adaptive_break_frequency, adaptive_injection_index, adaptive_remnant_ratio]
    return params, discretized_parameters, marginal_distributions, probability_vector, mesh_parameters, normalisation_vector


def spectral_fitter_(frequency : (list, np.ndarray), luminosity : (list, np.ndarray), dluminosity : (list, np.ndarray), \
    fit_type : str, n_breaks=31, break_range=[8,11], n_injects=31, inject_range=[2.01,2.99], n_remnants=31, \
    remnant_range=[0,1], n_iterations=3, b_field=None, redshift=None, write_model=False, work_dir=None, save_prefix=None):
    """
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
    """
    # logger.info(color_text('Entered function.',Colors.MediumSpringGreen))
    
    # ensure inputs have the correct type and sensible values
    frequency, luminosity, dluminosity, inject_range, break_range, remnant_range, n_injects, n_breaks, n_remnants = \
        CheckFunctionInputs.spectral_fitter(frequency,
        luminosity,
        dluminosity,
        fit_type,
        n_breaks,
        break_range,
        n_injects,
        inject_range,
        n_remnants,
        remnant_range,
        n_iterations,
        b_field,
        redshift)
    
    # convert luminosities and uncertainties to log-scale
    log_luminosity = np.log10(luminosity + 1e-307)
    dlog_luminosity = np.zeros_like(log_luminosity)
    for freqPointer in range(0, len(frequency)):
        if (luminosity[freqPointer] - dluminosity[freqPointer] > 0):
            dlog_luminosity[freqPointer] = (np.log10(luminosity[freqPointer] + dluminosity[freqPointer] + 1e-307) - np.log10(luminosity[freqPointer] - dluminosity[freqPointer] + 1e-307))/2
        else:
            dlog_luminosity[freqPointer] = 1e-307
    
    # calculate dof for chi-squared functions
    dof = -3
    if (n_breaks <= 2):
        dof = dof + 1 # number of parameters being fitted
    if (n_injects <= 2):
        dof = dof + 1 # number of parameters being fitted
    if (n_remnants <= 2):
        dof = dof + 1 # number of parameters being fitted
    for freqPointer in range(0, len(frequency)):
        if (dlog_luminosity[freqPointer] > 0):
            dof = dof + 1
    
    # set adaptive regions
    if (n_breaks >= 2):
        break_frequency = np.linspace(break_range[0], break_range[1], n_breaks, endpoint=True)
    else:
        break_frequency = np.zeros(1)
        break_frequency[0] = (break_range[0] + break_range[1])/2.
    if (n_injects >= 2):
        injection_index = np.linspace(inject_range[0], inject_range[1], n_injects, endpoint=True)
    else:
        injection_index = np.zeros(1)
        injection_index[0] = (inject_range[0] + inject_range[1])/2
    if (n_remnants >= 2):
        remnant_ratio = np.linspace(remnant_range[0], remnant_range[1], n_remnants, endpoint=True)
    else:
        remnant_ratio = np.zeros(1)
        remnant_ratio[0] = (remnant_range[0] + remnant_range[1])/2


    # instantiate array to store probabilities from chi-squared statistics for each adaptive mesh iteration.
    adaptive_probability = np.zeros((max(1, n_iterations), max(1, n_breaks), max(1, n_injects), max(1, n_remnants)))
    # instantiate array to store probabilities from chi-squared statistics
    probability = np.zeros((max(1, n_breaks), max(1, n_injects), max(1, n_remnants)))
    normalisation_array = np.zeros((max(1, n_breaks), max(1, n_injects), max(1, n_remnants)))
    # find chi-squared statistic for each set of parameters, and iterate through with adaptive mesh


    # instantiate arrays to store parameter-space grids for each adaptive loop
    adaptive_break_frequency = np.zeros((max(1, n_iterations), n_breaks))
    adaptive_injection_index = np.zeros((max(1, n_iterations), n_injects))
    adaptive_remnant_ratio = np.zeros((max(1, n_iterations), n_remnants))
    
    # instantiate arrays to store Pointers corresponding to peak probability
    adaptive_max_break = np.zeros(max(1, n_iterations))
    adaptive_max_inject = np.zeros(max(1, n_iterations))
    adaptive_max_remnant = np.zeros(max(1, n_iterations))

    # instantiate arrays to store free parameter estimates after each adaptive loop
    adaptive_break_peak = np.zeros(max(1, n_iterations))
    adaptive_inject_peak = np.zeros(max(1, n_iterations))
    adaptive_remnant_peak = np.zeros(max(1, n_iterations))


    # initiate adaptive mesh loop
    for iterationPointer in range(0, max(1, n_iterations)):

        # store current list of break frequencies
        adaptive_break_frequency[iterationPointer] = break_frequency
            
        # store current list of injection indices
        adaptive_injection_index[iterationPointer] = injection_index

        # store current list of remnant ratios
        adaptive_remnant_ratio[iterationPointer] = remnant_ratio

        # initiate the three-dimensional parameter space
        for breakPointer in range(0, max(1, n_breaks)):
            for injectPointer in range(0, max(1, n_injects)):
                for remnantPointer in range(0, max(1, n_remnants)):
                    
                    # find spectral fit for current set of parameters
                    normalisation = 0 # specify that fit needs to be scaled
                    if fit_type in ['JP', 'KP', 'CI']:
                        luminosity_predict, normalisation = __spectral_models_standard(frequency, luminosity, fit_type, \
                            10**break_frequency[breakPointer], injection_index[injectPointer], \
                            remnant_ratio[remnantPointer], normalisation)
                    elif fit_type in ['TJP', 'TKP', 'TCI']:
                        luminosity_predict, normalisation = __spectral_models_tribble(frequency, luminosity, fit_type, \
                            b_field, redshift, 10**break_frequency[breakPointer], injection_index[injectPointer], \
                            remnant_ratio[remnantPointer], normalisation)
                    
                    # store the normalisation for each spectral fit
                    normalisation_array[breakPointer,injectPointer,remnantPointer] = normalisation

                    # calculate chi-squared statistic and probability */                    
                    probability[breakPointer,injectPointer,remnantPointer] = np.nansum(((log_luminosity - np.log10(luminosity_predict + 1e-307))/dlog_luminosity)**2/2.)
                    # probability[breakPointer,injectPointer,remnantPointer] = 1 - chi2.cdf(probability[breakPointer,injectPointer,remnantPointer], dof)
                    
                    adaptive_probability[iterationPointer,breakPointer,injectPointer,remnantPointer] = probability[breakPointer,injectPointer,remnantPointer]

        # find peak of joint probability distribution
        max_probability = 1e+99
        max_break, max_inject, max_remnant = 0, 0, 0
        for breakPointer in range(0, max(1, n_breaks)):
            for injectPointer in range(0, max(1, n_injects)):
                for remnantPointer in range(0, max(1, n_remnants)):
                    if (probability[breakPointer,injectPointer,remnantPointer] < max_probability):
                        
                        max_probability = probability[breakPointer,injectPointer,remnantPointer]
                        normalisation = normalisation_array[breakPointer,injectPointer,remnantPointer]
                        
                        max_break = breakPointer
                        max_inject = injectPointer
                        max_remnant = remnantPointer

        # get the peak break frequency, peak injection index, and peak remnant ratio
        break_predict = break_frequency[max_break]
        inject_predict = injection_index[max_inject]
        remnant_predict = remnant_ratio[max_remnant]

        # store pointers in adaptive arrays
        adaptive_max_break[iterationPointer] = max_break
        adaptive_max_inject[iterationPointer] = max_inject
        adaptive_max_remnant[iterationPointer] = max_remnant

        # store peak estimates in adaptive arrays
        adaptive_break_peak[iterationPointer] = break_frequency[max_break]
        adaptive_inject_peak[iterationPointer] = injection_index[max_inject]
        adaptive_remnant_peak[iterationPointer] = remnant_ratio[max_remnant]

        # store the peak probability
        max_probability = probability[max_break,max_inject,max_remnant]
        

        # calculate standard deviation from marginal distributions
        sum_probability = 0
        dbreak_predict = 0
        if (n_breaks > 2):
            for breakPointer in range(0, max(1, n_breaks)):
                sum_probability = sum_probability + probability[breakPointer,max_inject,max_remnant]
                dbreak_predict = dbreak_predict + probability[breakPointer,max_inject,max_remnant]*(break_frequency[breakPointer] - break_predict)**2
            dbreak_predict = np.sqrt(dbreak_predict/sum_probability)

        sum_probability = 0
        dinject_predict = 0
        if (n_injects > 2):
            for injectPointer in range(0, max(1, n_injects)):
                sum_probability = sum_probability + probability[max_break,injectPointer,max_remnant]
                dinject_predict = dinject_predict + probability[max_break,injectPointer,max_remnant]*(injection_index[injectPointer] - inject_predict)**2
            dinject_predict = np.sqrt(dinject_predict/sum_probability)

        sum_probability = 0
        dremnant_predict = 0
        if (n_remnants > 2):
            for remnantPointer in range(0, max(1, n_remnants)):
                sum_probability = sum_probability + probability[max_break,max_inject,remnantPointer]
                dremnant_predict = dremnant_predict + probability[max_break,max_inject,remnantPointer]*(remnant_ratio[remnantPointer] - remnant_predict)**2
            dremnant_predict = np.sqrt(dremnant_predict/sum_probability)
        
        # update adaptive regions
        if (iterationPointer < n_iterations - 1):
            if (n_breaks > 2):
                dbreak_frequency = (break_frequency[-1] - break_frequency[0])/np.sqrt(n_breaks)
                # print(dbreak_frequency)
                break_frequency = np.linspace(max(break_range[0], break_predict - dbreak_frequency), min(break_range[1], break_predict + dbreak_frequency), n_breaks, endpoint=True)
                # break_frequency = np.linspace(break_predict - dbreak_frequency, break_predict + dbreak_frequency, n_breaks, endpoint=True)

            if (n_injects > 2):
                dinjection_index = (injection_index[-1] - injection_index[0])/np.sqrt(n_injects)
                injection_index = np.linspace(max(inject_range[0], inject_predict - dinjection_index), min(inject_range[1], inject_predict + dinjection_index), n_injects, endpoint=True)
                # injection_index = np.linspace(inject_predict - dinjection_index, inject_predict + dinjection_index, n_injects, endpoint=True)

            if (n_remnants > 2):
                dremnant_ratio = (remnant_ratio[-1] - remnant_ratio[0])/np.sqrt(n_remnants)
                remnant_ratio = np.linspace(max(remnant_range[0], remnant_predict - dremnant_ratio), min(remnant_range[1], remnant_predict + dremnant_ratio), n_remnants, endpoint=True)

    debug = False
    if debug == True:
        if save_prefix is not None:
            filename = '{}_injection_prob.pdf'.format(save_prefix)
        else: 
            filename = 'injection_prob.pdf'

        cm = pylab.get_cmap('rainbow')
        
        fig_inject = plt.figure(figsize=(10,10))
        ax_inject = fig_inject.add_axes([0.05, 0.05, 0.9, 0.9])
        for iterationPointer in range(0, max(1, n_iterations)):
            c = cm(1.*iterationPointer/max(1, n_iterations))
            max_break = int(adaptive_max_break[iterationPointer])
            max_remnant = int(adaptive_max_remnant[iterationPointer])
            injection_index = adaptive_injection_index[iterationPointer]
            test_prob = []
            for injectPointer in range(0, max(1, n_injects)):
                test_prob.append(adaptive_probability[iterationPointer,max_break,injectPointer,max_remnant])
            ax_inject.scatter(injection_index, test_prob, s=10, label='Iteration {}'.format(iterationPointer), color=c)
            ax_inject.plot(injection_index, test_prob, color=c)
            ax_inject.plot([adaptive_inject_peak[iterationPointer], adaptive_inject_peak[iterationPointer]],[0, 2], color=c)
        ax_inject.set_ylim([-0.05,1.05])
        ax_inject.legend(loc='upper right')
        plt.savefig('/home/sputnik/data/home/ben/resolved_sed_paper/fits_cutouts/final/{}'.format(filename),dpi=200)

        if save_prefix is not None:
            filename = '{}_break_prob.pdf'.format(save_prefix)
        else: 
            filename = 'break_prob.pdf'

        fig_break = plt.figure(figsize=(10,10))
        ax_break = fig_break.add_axes([0.05, 0.05, 0.9, 0.9])
        for iterationPointer in range(0, max(1, n_iterations)):
            c = cm(1.*iterationPointer/max(1, n_iterations))
            max_inject = int(adaptive_max_inject[iterationPointer])
            max_remnant = int(adaptive_max_remnant[iterationPointer])
            break_frequency = adaptive_break_frequency[iterationPointer]
            test_prob = []
            for breakPointer in range(0, max(1, n_breaks)):
                test_prob.append(adaptive_probability[iterationPointer,breakPointer,max_inject,max_remnant])
            ax_break.scatter(break_frequency, test_prob, s=10, label='Iteration {}'.format(iterationPointer),color=c)
            ax_break.plot(break_frequency, test_prob, color=c)
            ax_break.plot([adaptive_break_peak[iterationPointer], adaptive_break_peak[iterationPointer]],[0, 2], color=c)
        ax_break.set_ylim([-0.05,1.05])
        ax_break.legend(loc='upper right')
        plt.savefig('/home/sputnik/data/home/ben/resolved_sed_paper/fits_cutouts/final/{}'.format(filename),dpi=200)

    # return set of best-fit parameters with uncertainties
    espace='                      '
    colorstring = color_text("Free parameters optimised for {} model:".format(fit_type), Colors.DodgerBlue)
    logger.info(colorstring)
    colorstring = color_text(" {} s = {} +\- {} \n {} log_break_freq = {} +\- {} \n {} remnant_fraction = {} +\- {} ".format(espace, inject_predict, dinject_predict, espace, break_predict, dbreak_predict, espace, remnant_predict, dremnant_predict), Colors.Green)
    print(colorstring)

    # write fitting outputs to file
    if write_model is True:
        if fit_type is not None:
            filename = "{}_model_params.dat".format(fit_type)
        else:
            filename = "model_params.dat"
        if save_prefix is not None:
            filename = '{}_{}'.format(save_prefix, filename)
        else: 
            filename = filename
        if work_dir is not None:
            savename = "{}/{}".format(work_dir, filename)
        else:
            savename = filename
        colorstring = color_text("Writing fitting outputs to: {}".format(savename), Colors.DodgerBlue)
        logger.info(colorstring)
    
        file = open(savename, "w")
        file.write("log_break_freq, unc_log_break_freq, inj_index, unc_inj_index, remnant_fraction, unc_remnant_fraction, normalisation \n")
        file.write("{}, {}, {}, {}, {}, {}, {} \n".format(break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation))
        file.close()
    
    params = fit_type, break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation, max_probability
    debug_output = adaptive_probability, adaptive_break_frequency, adaptive_injection_index, adaptive_remnant_ratio, adaptive_max_break, adaptive_max_inject, adaptive_max_remnant
    
    if debug == True:
        return params, debug_output
    else:
        return params 


def spectral_model(fit_type, mesh_parameters, normalisation_vector, frequency, probability_vector, b_field=None, redshift=None, pcrit=1):

    break_frequencies = mesh_parameters[:,0][np.where(probability_vector>=pcrit)]
    injection_indices = mesh_parameters[:,1][np.where(probability_vector>=pcrit)]
    remnant_ratios = mesh_parameters[:,2][np.where(probability_vector>=pcrit)]
    normalisation_vector = normalisation_vector[np.where(probability_vector>=pcrit)]
    args = []
    for i in range(len(normalisation_vector)):
        break_frequency = break_frequencies[i]
        injection_index = injection_indices[i]
        remnant_ratio = remnant_ratios[i]
        normalisation = normalisation_vector[i]

        args.append([fit_type, frequency, 10**break_frequency, injection_index, remnant_ratio, normalisation, b_field, redshift])
    ncpus = 6
    with multiprocessing.Pool(processes=ncpus, maxtasksperchild=None) as pool:
        sf_stats_matrix = pool.starmap(__simulate_sf, args)
        luminosityArray = np.asarray(sf_stats_matrix)

    return luminosityArray


def spectral_model_(params : tuple, frequency : (list, np.ndarray), mc_length=500, err_width=2, \
    b_field=None, redshift=None, work_dir=None, write_model=False, save_prefix=None):
    """
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
    """
    
    # logger.info(color_text('Entered function.',Colors.MediumSpringGreen))
    
    # check data is the correct type
    if len(params) != 8:
        raise Exception(color_text("len(params) needs to be 8.",Colors.Red))
    fit_type, break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation = params
    if fit_type not in ['KP', 'TKP', 'JP', 'TJP', 'CI', 'TCI']:
        raise Exception(color_text("fit_type needs to be one of: KP, TKP, JP, TJP, CI, TCI",Colors.Red))
    if not isinstance(break_predict, float) or break_predict< 0:
        raise Exception(color_text('break_predict needs to be a float and greater than zero',Colors.Red))
    if not isinstance(dbreak_predict, (float, int)) or dbreak_predict < 0:
        raise Exception(color_text('dbreak_predict needs to be a float/int and greater than zero',Colors.Red))
    if not isinstance(inject_predict, float) or inject_predict < 0:
        raise Exception(color_text('inject_predict needs to be a float and greater than zero',Colors.Red))
    if not isinstance(dinject_predict, (float, int)) or dinject_predict < 0:
        raise Exception(color_text('dinject_predict needs to be a float/int and greater than zero',Colors.Red))
    if not isinstance(remnant_predict, (float,int)) or remnant_predict < 0:
        raise Exception(color_text('remnant_predict needs to be a float and greater than zero',Colors.Red))
    if not isinstance(dremnant_predict, (float, int)) or dremnant_predict < 0:
        raise Exception(color_text('dremnant_predict needs to be a float/int and greater than zero',Colors.Red))
    if not isinstance(normalisation, float) or normalisation < 0:
        raise Exception(color_text('normalisation needs to be a float and greater than zero',Colors.Red))
    if fit_type in ['TKP', 'TJP', 'TCI']:
        if redshift is None or b_field is None:
            raise Exception(color_text('{} requires a redshift and magnetic field strength.'.format(fit_type),Colors.Red))
    if b_field is not None and (not isinstance(b_field, (float, int)) or b_field < 0):
        raise Exception(color_text('Magnetic field strenth needs to be a float/int and greater than zero',Colors.Red))
    if redshift is not None and (not isinstance(redshift, (float, int)) or redshift < 0):
        raise Exception(color_text('Redshift needs to be a float/int and greater than zero',Colors.Red))
    
    # Evaluate the model over a list of frequencies
    if fit_type in ['JP', 'KP', 'CI']:
        model_data = __spectral_models_standard(frequency,np.zeros(len(frequency)), fit_type, \
            10**break_predict, inject_predict, remnant_predict, normalisation)[0]
    elif fit_type in ['TJP', 'TKP', 'TCI']:
        model_data = __spectral_models_tribble(frequency, np.zeros(len(frequency)), fit_type, \
            b_field, redshift, 10**break_predict, inject_predict, remnant_predict, normalisation)[0]
    
    # simulate a distribution of injection indices, break frequencies and quiescent fractions assuming gaussian errors.
    break_predict_vec = np.random.normal(break_predict, dbreak_predict, mc_length)
    inject_predict_vec = np.random.normal(inject_predict, dinject_predict, mc_length)
    remnant_predict_vec = np.random.normal(remnant_predict, dremnant_predict, mc_length)

    # write distributions to file
    if write_model is not None and write_model == True:
        free_param = [break_predict_vec, inject_predict_vec, remnant_predict_vec]
        free_param_name = ['log_break_frequency', 'injection_index', 'remnant_ratio']
        for i in range(0,len(free_param)):
            fig = plt.figure(figsize=[5,5])
            ax = fig.add_axes([0.1,0.1,0.85,0.85])
            ax.hist(free_param[i], bins=20)
            ax.set_xlabel(free_param_name[i], fontsize=10)
            if fit_type is not None:
                filename = '{}_{}_distribution.pdf'.format(fit_type,free_param_name[i])
            else:
                filename = '{}_distribution.pdf'.format(free_param_name[i])
            if save_prefix is not None:
                filename = '{}_{}'.format(save_prefix, filename)
            else: 
                filename = filename       
            if work_dir is not None:
                savename = '{}/{}'.format(work_dir, filename)
            else:
                savename = filename
            plt.savefig(savename,dpi=200)
    
    # instantiate vectors to store the uncertainties in the model at the corresponding frequency
    err_model_data=np.zeros([len(frequency)])
    model_data_min=np.zeros([len(frequency)])
    model_data_max=np.zeros([len(frequency)])

    if mc_length is not None:
        # instantiate array to store Monte-Carlo simulated spectra
        luminosityArray = np.zeros([mc_length,len(frequency)])
        frequencyArray = np.zeros([mc_length,len(frequency)])

        # evaluate and store the model spectrum for each set of free parameters
        colorstring = color_text("Estimating model errors from {} Monte-Carlo iterations".format(mc_length), Colors.DodgerBlue)
        logger.info(colorstring)
        for mcPointer in range(0,mc_length):
            if fit_type in ['JP', 'KP', 'CI']:
                fitmc, normmc = __spectral_models_standard(frequency, np.zeros(len(frequency)), fit_type, 10**break_predict_vec[mcPointer], \
                    inject_predict_vec[mcPointer], remnant_predict_vec[mcPointer], normalisation)
            elif fit_type in ['TJP', 'TKP', 'TCI']:
                fitmc, normmc = __spectral_models_tribble(frequency, np.zeros(len(frequency)), fit_type, b_field, redshift, 10**break_predict_vec[mcPointer], \
                inject_predict_vec[mcPointer], remnant_predict_vec[mcPointer], normalisation)
            luminosityArray[mcPointer] = (np.asarray(fitmc))
            frequencyArray[mcPointer] = (np.asarray(frequency))

                # take the std dev in the model at each frequency to evaluate a model uncertainty
        for freqPointer in range(0,len(frequency)):
            err_model_data[freqPointer] = np.std(luminosityArray.T[freqPointer])
            model_data_min[freqPointer] = model_data[freqPointer] - 0.5*err_width*np.std(luminosityArray.T[freqPointer])
            model_data_max[freqPointer] = model_data[freqPointer] + 0.5*err_width*np.std(luminosityArray.T[freqPointer])


    # write fitting outputs to file
    if write_model is not None and write_model == True:
        if fit_type is not None:
            filename = '{}_model_spectrum.dat'.format(fit_type)
        else:
            filename = 'model_spectrum.dat'
        if save_prefix is not None:
            filename = '{}_{}'.format(save_prefix, filename)
        else: 
            filename = filename
        if work_dir is not None:
            savename = '{}/{}'.format(work_dir, filename)
        else:
            savename = filename
        colorstring = color_text("Writing model spectrum to: {}".format(savename), Colors.DodgerBlue)
        logger.info(colorstring)
    
        file = open(savename, "w")
        file.write("Frequency, {0} Model, unc {0} Model, {0} Model +- {1} sigma, {0} Model +- {1} sigma \n".format(fit_type, err_width))
        for i in range(0,len(frequency)):
            file.write("{}, {}, {}, {}, {} \n".format(frequency[i], model_data[i], err_model_data[i], model_data_min[i], model_data_max[i]))
        file.close()
    
    # logger.info(color_text('Exitting function.',Colors.MediumSpringGreen))
    return(model_data, err_model_data, model_data_min, model_data_max)


def spectral_plotter(observed_data : tuple, model_data=None, plotting_data=None, work_dir=None, fit_type=None, err_model_width=2, save_prefix=None):
    """
    (usage) Plots the data and optimised model fit to figure, and writes figure to file. 
    
    parameters
    ----------
    observed_data : a tuple containing the observed spectral data (frequency_obs, luminosity_obs, err_luminosity_obs)
                    frequency_obs      : list of observed frequencies
                    luminosity_obs     : list of observed flux densities
                    err_luminosity_obs : list of flux density uncertainties

    model_data    : the model evaluated over the observed frequencies

    plotting_data : a tuple containing the model spectral data evaluated over a list of plotting frequencies 
                    (frequency_plt, luminosity_plt, err_luminosity_plt)
                    frequency_plt      : list of plotting frequencies
                    luminosity_plt     : list of model flux densities evaluated over frequency_plt
                    err_luminosity_plt : list of model flux density uncertainties
                    luminosity_plt_min : list of the lower-bound model flux densities
                    luminosity_plt_max : list of the lower-bound model flux densities

    work_dir      : Directory to write plot to

    fit_type      : Print model type on figure

    err_width     : The width of the uncertainty envelope in the model

    Returns 
    -------
    None

    Comments
    --------
    It is assumed that the units of the observed data and model are the same. 
    """
    # logger.info(color_text('Entered function.',Colors.MediumSpringGreen))
    
    # check observed_data has the correct type
    if not isinstance(observed_data, tuple) or len(observed_data) != 3:
        raise Exception(color_text('observed_data needs to be a tuple of length 3',Colors.Red))

    # unpack and check observed_data
    frequency_obs, luminosity_obs, err_luminosity_obs = observed_data
    if not isinstance(frequency_obs, (list, np.ndarray)):
        raise Exception(color_text('frequency_obs needs to be a list or np.ndarray',Colors.Red))
    if not isinstance(luminosity_obs, (list, np.ndarray)):
        raise Exception(color_text('luminosity_obs needs to be a list or np.ndarray',Colors.Red))
    if not isinstance(err_luminosity_obs, (list, np.ndarray)):
        raise Exception(color_text('err_luminosity_obs needs to be a list or np.ndarray',Colors.Red))
    if len(frequency_obs) != len(luminosity_obs) or len(frequency_obs) != len(err_luminosity_obs):
        raise Exception(color_text('frequency_obs, luminosity_obs and err_luminosity_obs must be same length',Colors.Red))
    if isinstance(frequency_obs, list):
        frequency_obs = np.asarray(frequency_obs)
    if isinstance(luminosity_obs, list):
        luminosity_obs = np.asarray(luminosity_obs)
    if isinstance(err_luminosity_obs, list):
        err_luminosity_obs = np.asarray(err_luminosity_obs)
    
    # check model data is correct type
    if model_data is not None and not isinstance(model_data, (list, np.ndarray)):
        raise Exception(color_text('model_data needs to be a list or np.ndarray',Colors.Red))
        if len(model_data) != len(frequency_obs):
            raise Exception(color_text('model_data must have same length as frequency_obs',Colors.Red))
        if isinstance(model_data, list):
            model_data = np.asarray(model_data)

    # unpack and check plotting_data
    if plotting_data is not None and (len(plotting_data) != 5 or not isinstance(plotting_data, tuple)):
        raise Exception(color_text('plotting_data needs to be a tuple of length 3',Colors.Red))
    else:
        frequency_plt, luminosity_plt, err_luminosity_plt, luminosity_plt_min, luminosity_plt_max  = plotting_data
        if not isinstance(frequency_plt, (list, np.ndarray)):
            raise Exception(color_text('frequency_plt needs to be a list or np.ndarray',Colors.Red))
        if not isinstance(luminosity_plt, (list, np.ndarray)):
            raise Exception(color_text('luminosity_plt needs to be a list or np.ndarray',Colors.Red))
        if not isinstance(err_luminosity_plt, (list, np.ndarray)):
            raise Exception(color_text('err_luminosity_plt needs to be a list or np.ndarray',Colors.Red))
        if not isinstance(luminosity_plt_min, (list, np.ndarray)):
            raise Exception(color_text('luminosity_plt_min needs to be a list or np.ndarray',Colors.Red))
        if not isinstance(luminosity_plt_max, (list, np.ndarray)):
            raise Exception(color_text('luminosity_plt_max needs to be a list or np.ndarray',Colors.Red))
        if len(frequency_plt) != len(luminosity_plt) or len(frequency_plt) != len(err_luminosity_plt):
            raise Exception(color_text('frequency_plt, luminosity_plt and err_luminosity_plt must be same length',Colors.Red))
    
    # instantiate figure
    fig = plt.figure(figsize=(12,12))
    if model_data is not None:
        # make a difference sub-plot
        ax_main = fig.add_axes([0.1,0.15,0.86,0.81])
        ax_main.axes.get_xaxis().set_visible(False)
        
        ax_sub = fig.add_axes([0.1, 0.06, 0.86, 0.08])
        ax_sub.axes.get_yaxis().set_ticks([1])
        ax_sub.set_xscale('log')
        ax_sub.set_xlim([10**(math.floor(np.min(np.log10(frequency_obs)))),10**(math.ceil(np.max(np.log10(frequency_obs))))])
        ax_sub.set_xlabel('Frequency / Hz', fontsize=20)
        ax_sub.scatter(frequency_obs, (luminosity_obs/model_data), marker='.', c='black', zorder=1)
        ax_sub.tick_params(axis='both', labelsize=20, which='both', direction='in', length=5, width=2)
        ax_sub.plot([10**(math.floor(np.min(np.log10(frequency_obs)))),10**(math.ceil(np.max(np.log10(frequency_obs))))],[1, 1], c='red')
        ax_sub.set_ylabel('$\\frac{data}{model}$', fontsize=25)
    else:
        ax_main = fig.add_axes([0.1,0.1,0.86,0.86])
        ax_main.set_xlabel('Frequency / Hz', fontsize=20)
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_ylabel('Integrated flux density / Jy', fontsize=20)
    ax_main.set_xlim([10**(math.floor(np.min(np.log10(frequency_obs)))),10**(math.ceil(np.max(np.log10(frequency_obs))))])
    ax_main.set_ylim([0.2*np.min(luminosity_obs),5*np.max(luminosity_obs)])
    ax_main.tick_params(axis='both', labelsize=20, which='both', direction='in', length=5, width=2)

    # plot the observed data
    ax_main.scatter(frequency_obs, luminosity_obs, marker='.', c='black', label='Data', zorder=1)
    ax_main.errorbar(frequency_obs, luminosity_obs, xerr=0, yerr=err_luminosity_obs, color='black', capsize=1, linestyle='None', fmt='none',alpha=0.9)

    # overlay the simulated model
    if plotting_data is not None:
        if fit_type is None:
            fit_type = ''
        ax_main.plot(frequency_plt, luminosity_plt, c='C0', label='{} Model fit'.format(fit_type), zorder=2)
        ax_main.fill_between(frequency_plt, luminosity_plt_min, luminosity_plt_max, color='purple', alpha=0.15)

    # work out the appropriate save name
    extension='.pdf'
    if fit_type is not None:
        plotname='{}_model_fit{}'.format(fit_type,extension)
    else:
        plotname='model_fit{}'.format(extension)
    if save_prefix is not None:
        plotname = '{}_{}'.format(save_prefix, plotname)
    else: 
        plotname = plotname
    if work_dir is not None:
        savename='{}/{}'.format(work_dir,plotname)
    else:
        savename=plotname

    # write figure to file
    colorstring = color_text("Writing figure to: {}".format(savename), Colors.DodgerBlue)
    logger.info(colorstring)
    plt.legend(loc='upper right', fontsize=20)
    plt.savefig(savename,dpi=200)


def spectral_ages(params : tuple, b_field : float, redshift : float):
    """
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
    """
    # logger.info(color_text('Entered function.',Colors.MediumSpringGreen))

    # unpack values
    fit_type, break_frequency, remnant_fraction = params

    # check data are of the correct type
    if not isinstance(break_frequency, (float)) or break_frequency < 0:
        raise Exception(color_text('Break frequency needs to be a float and greater than zero.', Colors.Red))
    if not isinstance(remnant_fraction, (float)) or remnant_fraction < 0:
        raise Exception(color_text('Quiescent fraction needs to be a float and greater than zero.', Colors.Red))
    if not isinstance(b_field, (float)) or b_field <= 0:
        raise Exception(color_text('Magnetic field strength needs to be a float and greater than zero.', Colors.Red))
    if not isinstance(redshift, (float)) or redshift <= 0:
        raise Exception(color_text('Redshift needs to be a float and greater than zero.', Colors.Red))
    if not isinstance(params, tuple) or len(params) != 3:
        raise Exception(color_text('params needs to be a tuple with len(params)=3', Colors.Red))
    if not isinstance(fit_type, str) or fit_type not in ['KP', 'TKP', 'JP', 'TJP', 'CI', 'TCI']:
        raise Exception(color_text('fit_type needs to be one of: KP, TKP, JP, TJP, CI, TCI', Colors.Red))

    espace='                    '
    colorstring = color_text("Spectral ages are calculated assuming the following parameters:", Colors.DodgerBlue)
    logger.info(colorstring)
    colorstring = color_text(" {} Model = {} \n {} Break frequency = {} Hz \n {} Quiescent fraction = {} \n {} Magnetic field strength = {} nT \n {} Redshift = {}".format(espace, fit_type, espace, break_frequency, espace, remnant_fraction, espace, b_field*1e+9, espace, redshift), Colors.Green)
    print(colorstring)

    # define important variables
    c, me, mu0, e = Const.c, Const.me, Const.mu0, Const.e
    Bic = 0.318*((1+redshift)**2)*1e-9
    
    # get the right value for v
    if fit_type in ['JP', 'TJP', 'CI', 'TCI']:
        v = ((243*np.pi*(me**5)*(c**2))/(4*(mu0**2)*(e**7)))**(0.5)
    elif fit_type in ['KP', 'TKP']:
        v = (1/2.25)*(((243*np.pi*(me**5)*(c**2))/(4*(mu0**2)*(e**7)))**(0.5))
    
    # evaluate the spectral age
    tau = ((v*(b_field**(0.5)))/((b_field**2)+(Bic**2)))*((break_frequency*(1+redshift))**(-0.5))

    # convert into Myr
    tau = tau/(3.154e+13) # Myr
    t_on = tau*(1-remnant_fraction)
    t_off = tau - t_on

    # print values to screen
    if fit_type in ['CI', 'TCI']:
        logger.info(color_text("Spectral ages estimated for {} model:".format(fit_type), Colors.DodgerBlue))
        print(color_text(" {} Total age = {} Myr \n {} Active duration = {} Myr \n {} Remnant duration = {} Myr"\
            .format(espace, tau, espace, t_on, espace, t_off), Colors.Green))
    elif fit_type in ['JP', 'TJP', 'KP', 'TKP']:
        logger.info(color_text("Spectral age estimated for {} model:".format(fit_type), Colors.DodgerBlue))
        print(color_text(" {} Total age = {} Myr".format(espace, tau), Colors.Green))
    
    # logger.info(color_text('Exitting function.',Colors.MediumSpringGreen))
    spectral_ages = (tau, t_on, t_off)
    return(spectral_ages)