#!/usr/bin/env python

__author__ = "Benjamin Quici, Ross J. Turner"
__date__ = "25/02/2021"

import math
import numpy as np
from numba import jit
from scipy.stats import chi2
from matplotlib import pyplot as plt
import logging

logging.basicConfig(format="%(levelname)s (%(funcName)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Colors:
    DogderBlue = (30, 144, 255)
    Green = (0,200,0)
    Orange = (255, 165, 0)
    Red = (255, 0, 0)
    MediumSpringGreen = (0, 250, 154)

def _join(*values):
    return ";".join(str(v) for v in values)

def color_text(s, c, base=30):
    template = "\x1b[{0}m{1}\x1b[0m"
    t = _join(base+8, 2, _join(*c))
    return template.format(t, s)

def besselK53():
    """
    (usage) Returns the modified Bessel function of order 5/3, evaluated over x = [1e-05, 60.25].
    Note, x represents a dimensionless function of frequency, e.g. equation (3) of Turner 2017b. 
    
    returns
    -------
    bessel_x : 1darray
        Values of x
    bessel_F : 1darray
        The value of the Bessel function evaluated at bessel_x
    """
    bessel_x = [1e-05, 1.122018e-05, 1.258925e-05, 1.412538e-05, 1.584893e-05, 1.778279e-05, 1.995262e-05, 2.238721e-05, 2.511886e-05, 2.818383e-05, 3.162278e-05, 3.548134e-05, 3.981072e-05, 4.466836e-05, 5.011872e-05, 5.623413e-05, 6.309573e-05, 7.079458e-05, 7.943282e-05, 8.912509e-05, 0.0001, 0.0001122018, 0.0001258925, 0.0001412538, 0.0001584893, 0.0001778279, 0.0001995262, 0.0002238721, 0.0002511886, 0.0002818383, 0.0003162278, 0.0003548134, 0.0003981072, 0.0004466836, 0.0005011872, 0.0005623413, 0.0006309573, 0.0007079458, 0.0007943282, 0.0008912509, 0.001, 0.001122018, 0.001258925, 0.001412538, 0.001584893, 0.001778279, 0.001995262, 0.002238721, 0.002511886, 0.002818383, 0.003162278, 0.003548134, 0.003981072, 0.004466836, 0.005011872, 0.005623413, 0.006309573, 0.007079458, 0.007943282, 0.008912509, 0.01, 0.01047129, 0.01096478, 0.01148154, 0.01202264, 0.01258925, 0.01318257, 0.01380384, 0.0144544, 0.01513561, 0.01584893, 0.01659587, 0.01737801, 0.01819701, 0.01905461, 0.01995262, 0.02089296, 0.02187762, 0.02290868, 0.02398833, 0.02511886, 0.02630268, 0.02754229, 0.02884032, 0.03019952, 0.03162278, 0.03311311, 0.03467369, 0.03630781, 0.03801894, 0.03981072, 0.04168694, 0.04365158, 0.04570882, 0.04786301, 0.05011872, 0.05248075, 0.05495409, 0.05754399, 0.06025596, 0.06309573, 0.06606934, 0.0691831, 0.0724436, 0.07585776, 0.07943282, 0.08317638, 0.08709636, 0.09120108, 0.09549926, 0.1, 0.1047129, 0.1096478, 0.1148154, 0.1202264, 0.1258925, 0.1318257, 0.1380384, 0.144544, 0.1513561, 0.1584893, 0.1659587, 0.1737801, 0.1819701, 0.1905461, 0.1995262, 0.2089296, 0.2187762, 0.2290868, 0.2398833, 0.2511886, 0.2630268, 0.2754229, 0.2884032, 0.3019952, 0.3162278, 0.3311311, 0.3467369, 0.3630781, 0.3801894, 0.3981072, 0.4168694, 0.4365158, 0.4570882, 0.4786301, 0.5011872, 0.5248075, 0.5495409, 0.5754399, 0.6025596, 0.6309573, 0.6606934, 0.691831, 0.724436, 0.7585776, 0.7943282, 0.8317638, 0.8709636, 0.9120108, 0.9549926, 1.0, 1.047129, 1.096478, 1.148154, 1.202264, 1.258925, 1.318257, 1.380384, 1.44544, 1.513561, 1.584893, 1.659587, 1.737801, 1.819701, 1.905461, 1.995262, 2.089296, 2.187762, 2.290868, 2.398833, 2.511886, 2.630268, 2.754229, 2.884032, 3.019952, 3.162278, 3.311311, 3.467369, 3.630781, 3.801894, 3.981072, 4.168694, 4.365158, 4.570882, 4.786301, 5.011872, 5.248075, 5.495409, 5.754399, 6.025596, 6.309573, 6.606934, 6.91831, 7.24436, 7.585776, 7.943282, 8.317638, 8.709636, 9.120108, 9.549926, 10.0, 10.47129, 10.96478, 11.48154, 12.02264, 12.58925, 13.18257, 13.80384, 14.4544, 15.13561, 15.84893, 16.59587, 17.37801, 18.19701, 19.05461, 19.95262, 20.89296, 21.87762, 22.90868, 23.98833, 25.11886, 26.30268, 27.54229, 28.84032, 30.19952, 31.62278, 33.11311, 34.67369, 36.30781, 38.01894, 39.81072, 41.68694, 43.65158, 45.70882, 47.86301, 50.11872, 52.48075, 54.95409, 57.54399, 60.25596]
    bessel_F = [0.04629204, 0.04810159, 0.04998175, 0.05193526, 0.05396496, 0.05607381, 0.05826488, 0.06054133, 0.06290648, 0.06536375, 0.06791669, 0.070569, 0.07332448, 0.07618712, 0.07916102, 0.08225045, 0.08545982, 0.08879372, 0.09225689, 0.09585424, 0.09959088, 0.1034721, 0.1075033, 0.1116901, 0.1160385, 0.1205543, 0.1252439, 0.1301138, 0.1351705, 0.1404209, 0.1458721, 0.1515314, 0.1574063, 0.1635045, 0.1698341, 0.176403, 0.1832197, 0.1902929, 0.1976311, 0.2052435, 0.2131391, 0.2213272, 0.2298174, 0.2386191, 0.247742, 0.257196, 0.2669908, 0.2771361, 0.2876418, 0.2985174, 0.3097725, 0.3214162, 0.3334574, 0.3459047, 0.3587661, 0.3720487, 0.3857592, 0.3999031, 0.4144848, 0.4295075, 0.4449725, 0.4512824, 0.4576629, 0.4641139, 0.4706351, 0.4772263, 0.483887, 0.4906169, 0.4974154, 0.5042819, 0.5112157, 0.5182161, 0.5252821, 0.5324129, 0.5396072, 0.546864, 0.554182, 0.5615596, 0.5689953, 0.5764874, 0.5840341, 0.5916334, 0.599283, 0.6069808, 0.614724, 0.6225102, 0.6303363, 0.6381993, 0.6460958, 0.6540225, 0.6619754, 0.6699506, 0.6779438, 0.6859505, 0.6939659, 0.7019849, 0.7100021, 0.7180117, 0.7260077, 0.7339838, 0.7419331, 0.7498485, 0.7577226, 0.7655474, 0.7733146, 0.7810154, 0.7886406, 0.7961807, 0.8036253, 0.810964, 0.8181855, 0.8252783, 0.8322301, 0.8390283, 0.8456595, 0.8521099, 0.8583651, 0.8644101, 0.8702292, 0.8758063, 0.8811245, 0.8861664, 0.8909142, 0.895349, 0.8994518, 0.9032027, 0.9065815, 0.9095673, 0.9121388, 0.914274, 0.9159508, 0.9171463, 0.9178378, 0.9180018, 0.9176149, 0.9166536, 0.9150942, 0.9129131, 0.910087, 0.906593, 0.9024084, 0.8975113, 0.8918805, 0.8854959, 0.8783384, 0.8703902, 0.8616355, 0.8520599, 0.8416513, 0.8303998, 0.8182985, 0.8053429, 0.7915322, 0.7768687, 0.7613587, 0.7450125, 0.7278449, 0.7098753, 0.6911279, 0.6716322, 0.6514228, 0.6305401, 0.6090297, 0.5869434, 0.5643382, 0.541277, 0.5178279, 0.4940644, 0.4700649, 0.445912, 0.4216926, 0.3974965, 0.3734163, 0.3495463, 0.3259814, 0.3028163, 0.2801445, 0.2580567, 0.23664, 0.2159765, 0.1961424, 0.1772062, 0.1592282, 0.1422594, 0.1263403, 0.1115008, 0.0977593, 0.08512255, 0.07358577, 0.06313278, 0.05373656, 0.04536004, 0.03795704, 0.03147351, 0.02584889, 0.02101761, 0.01691069, 0.01345729, 0.01058629, 0.008227734, 0.006314173, 0.004781787, 0.003571302, 0.002628682, 0.001905562, 0.00135946, 0.0009537575, 0.0006574939, 0.0004450058, 0.0002954477, 0.0001922383, 0.0001224694, 7.63148e-05, 4.646519e-05, 2.761261e-05, 1.599737e-05, 9.024603e-06, 4.951062e-06, 2.638069e-06, 1.3633e-06, 6.823153e-07, 3.302248e-07, 1.543035e-07, 6.94964e-08, 3.011714e-08, 1.253542e-08, 5.001613e-09, 1.909237e-09, 6.957938e-10, 2.41558e-10, 7.970527e-11, 2.493648e-11, 7.37861e-12, 2.059504e-12, 5.407576e-13, 1.331809e-13, 3.067388e-14, 6.585815e-15, 1.31379e-15, 2.426677e-16, 4.1351460000000005e-17, 6.476066e-18, 9.284197e-19, 1.2133319999999999e-19, 1.439201e-20, 1.542358e-21, 1.48625e-22, 1.281342e-23, 9.831652e-25, 6.677119999999999e-26]
    return(np.array(bessel_x), np.array(bessel_F))

def spectral_fitter(frequency, luminosity, dluminosity, fit_type, n_breaks=31, break_range=[8,11], n_injects=31, inject_range=[2.01,2.99], n_remnants=31, remnant_range=[0,1], n_iterations=3, bfield=None, z=None, options=None):
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
        The type of model to fit (JP, KP, CI)
    n_breaks : int
        Number of break frequencies used in adaptive grid
    break_range : list
        Bounds for the log(break frequency) range
    n_injects : int
        Number of injection indices used in adaptive grid
    inject_range : list
        bounds for the injection index range
    n_remnants : int
        Number of remnant ratios used in adaptive grid
    remnant_range : list
        Bounds for the remnant ratio range
    n_iterations : int
        Number of iterations
    options : Namespace()
        Custom options parsed through argparse. This is only relevant if synchrofit is executed from main.

    returns
    -------
    params : tuple
        Contains the fitted model (fit_type) and the free parameters constrained by the fitting (and their uncertainties)
    """
    logger.info(color_text('Entered function.',Colors.MediumSpringGreen))
    
    # read in custom configuration options from argparse
    if options is not None:
        n_breaks = options.n_breaks
        break_range = options.break_range
        n_injects = options.n_injects
        inject_range = options.inject_range
        n_remnants = options.n_remnants
        remnant_range = options.remnant_range
        n_iterations = options.n_iterations
        work_dir = options.work_dir
        write_model = options.write_model
        bfield = options.bfield
        z = options.z
        # if a single value is parsed as rather than a range, convert the list containing 1 item into a float
        if len(break_range) == 1:
            break_range = break_range[0]
        if len(inject_range) == 1:
            inject_range = inject_range[0]
        if len(remnant_range) == 1:
            remnant_range = remnant_range[0]
    
    # check inputs are of correct data types
    if not isinstance(frequency, (list, np.ndarray)) or not isinstance(luminosity, (list, np.ndarray)) or not isinstance(dluminosity, (list, np.ndarray)) or not len(luminosity) == len(frequency) or not len(dluminosity) == len(frequency):
        raise Exception(color_text('Frequency, luminosity and uncertainty arrays must be lists or numpy arrays of the same length.'),Colors.Red)
    if not isinstance(fit_type, str) or fit_type not in ['JP','TJP','KP','TKP','CI','TCI']: #(fit_type == 'CI' or fit_type == 'JP' or fit_type == 'KP'):
        raise Exception(color_text('Spectral fit must be either \'CI\', \'JP\' or \'KP\' model.'),Colors.Red)
    if not isinstance(n_breaks, (int, float)) or not isinstance(n_injects, (int, float)) or not isinstance(n_remnants, (int, float)):
        raise Exception(color_text('Number of break frequencies, injection indices and remnant ratios must be integers.'),Colors.Red)
    if isinstance(n_breaks, float):
        n_breaks = int(n_breaks)
    if isinstance(n_injects, float):
        n_injects = int(n_injects)
    if isinstance(n_remnants, float):
        n_remnants = int(n_remnants)
    if not (isinstance(break_range, (float, int)) or isinstance(break_range, (list, np.ndarray)) and len(break_range) == 2):
        raise Exception(color_text('Break frequency must be a float, or a two element list or numpy array.'),Colors.Red)
    if isinstance(break_range, (float, int)):
        break_range = [break_range, break_range]
        n_breaks = 1
    if not (isinstance(inject_range, (float, int)) or isinstance(inject_range, (list, np.ndarray)) and len(inject_range) == 2):
        raise Exception(color_text('Injection index must be a float, or a two element list or numpy array.'),Colors.Red)
    if isinstance(inject_range, (float, int)):
        inject_range = [inject_range, inject_range]
        n_injects = 1
    if isinstance(remnant_range, (float, int)):
        remnant_range = [remnant_range, remnant_range]
        n_remnants = 1
    if not (isinstance(remnant_range, (float, int)) or isinstance(remnant_range, (list, np.ndarray)) and len(remnant_range) == 2):
        raise Exception(color_text('Remnant ratio must be a float, or a two element list or numpy array.'),Colors.Red)
    if fit_type in ['TKP', 'TJP', 'TCI']:
        if z is None or bfield is None:
            raise Exception(color_text('{} requires a redshift and magnetic field strength.'.format(fit_type),Colors.Red))
    if bfield is not None and (not isinstance(bfield, (float, int)) or bfield < 0):
        raise Exception(color_text('Magnetic field strenth needs to be a float/int and greater than zero',Colors.Red))
    if z is not None and (not isinstance(z, (float, int)) or z < 0):
        raise Exception(color_text('Redshift needs to be a float/int and greater than zero',Colors.Red))

    # simplify the grid fitting if the following models are specified
    if fit_type in ['JP', 'TJP', 'KP', 'TKP']:
        n_remnants = 1
        remnant_range = [0, 0]

    # print accepted parameters
    espace='                      '
    colorstring = color_text("Fitting options accepted:", Colors.DogderBlue)
    logger.info(colorstring)
    colorstring=color_text(" {} fit_type = {} \n {} inject_range = {} \n {} n_injects = {} \n {} n_breaks = {} \n {} break_range = {} \n {} n_remnants = {} \n {} remnant_range = {}".format(espace, fit_type,espace,inject_range, espace, n_injects, espace, n_breaks, espace, break_range, espace, n_remnants, espace, remnant_range), Colors.Green)
    print(colorstring)
    
    # convert luminosities and uncertainties to log-scale
    log_luminosity = np.log10(luminosity + 1e-307)
    dlog_luminosity = np.zeros_like(log_luminosity)
    for freqPointer in range(0, len(frequency)):
        if (luminosity[freqPointer] - dluminosity[freqPointer] > 0):
            dlog_luminosity[freqPointer] = (np.log10(luminosity[freqPointer] + dluminosity[freqPointer] + 1e-307) - np.log10(luminosity[freqPointer] - dluminosity[freqPointer] + 1e-307))/2
        else:
            dlog_luminosity[freqPointer] = 1e-307
    
    bessel_x, bessel_F = besselK53()
    
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
        
    # instantiate array to store probabilities from chi-squared statistics
    probability = np.zeros((max(1, n_breaks), max(1, n_injects), max(1, n_remnants)))
    normalisation_array = np.zeros((max(1, n_breaks), max(1, n_injects), max(1, n_remnants)))
    # find chi-squared statistic for each set of parameters, and iterate through with adaptive mesh
    for iterationPointer in range(0, max(1, n_iterations)):
        for breakPointer in range(0, max(1, n_breaks)):
            for injectPointer in range(0, max(1, n_injects)):
                for remnantPointer in range(0, max(1, n_remnants)):
                    
                    # find spectral fit for current set of parameters
                    normalisation = 0 # specify that fit needs to be scaled
                    if fit_type in ['JP', 'KP', 'CI']:
                        luminosity_predict, normalisation = spectral_models(frequency, luminosity, fit_type, \
                            10**break_frequency[breakPointer], injection_index[injectPointer], \
                            remnant_ratio[remnantPointer], normalisation, bessel_x, bessel_F)
                    elif fit_type in ['TJP', 'TKP', 'TCI']:
                        luminosity_predict, normalisation = spectral_models_tribble(frequency, luminosity, fit_type, \
                            bfield, z, 10**break_frequency[breakPointer], injection_index[injectPointer], \
                            remnant_ratio[remnantPointer], normalisation, bessel_x, bessel_F)
                    normalisation_array[breakPointer,injectPointer,remnantPointer] = normalisation

                    # calculate chi-squared statistic and probability */                    
                    probability[breakPointer,injectPointer,remnantPointer] = np.nansum(((log_luminosity - np.log10(luminosity_predict + 1e-307))/dlog_luminosity)**2/2.)
                    probability[breakPointer,injectPointer,remnantPointer] = 1 - chi2.cdf(probability[breakPointer,injectPointer,remnantPointer], dof)
       
        # find peak of joint probability distribution
        max_probability = 0.
        max_break, max_inject, max_remnant = 0, 0, 0
        for breakPointer in range(0, max(1, n_breaks)):
            for injectPointer in range(0, max(1, n_injects)):
                for remnantPointer in range(0, max(1, n_remnants)):
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
                break_frequency = np.linspace(max(break_range[0], break_predict - dbreak_frequency), min(break_range[1], break_predict + dbreak_frequency), n_breaks, endpoint=True)
            if (n_injects > 2):
                dinjection_index = (injection_index[-1] - injection_index[0])/np.sqrt(n_injects)
                injection_index = np.linspace(max(inject_range[0], inject_predict - dinjection_index), min(inject_range[1], inject_predict + dinjection_index), n_injects, endpoint=True)
            if (n_remnants > 2):
                dremnant_ratio = (remnant_ratio[-1] - remnant_ratio[0])/np.sqrt(n_remnants)
                remnant_ratio = np.linspace(max(remnant_range[0], remnant_predict - dremnant_ratio), min(remnant_range[1], remnant_predict + dremnant_ratio), n_remnants, endpoint=True)

    # return set of best-fit parameters with uncertainties
    colorstring = color_text("Free parameters optimised for {} model:".format(fit_type), Colors.DogderBlue)
    logger.info(colorstring)
    colorstring = color_text(" {} s = {} +\- {} \n {} log_break_freq = {} +\- {} \n {} remnant_fraction = {} +\- {} ".format(espace, inject_predict, dinject_predict, espace, break_predict, dbreak_predict, espace, remnant_predict, dremnant_predict), Colors.Green)
    print(colorstring)

    # write fitting outputs to file
    if write_model is not None:
        if fit_type is not None:
            filename = "{}_model_params.dat".format(fit_type)
        else:
            filename = "model_params.dat"
        if work_dir is not None:
            savename = "{}/{}".format(work_dir, filename)
        else:
            savename = filename
        colorstring = color_text("Writing fitting outputs to: {}".format(savename), Colors.DogderBlue)
        logger.info(colorstring)
    
        file = open(savename, "w")
        file.write("log_break_freq, unc_log_break_freq, inj_index, unc_inj_index, remnant_fraction, unc_remnant_fraction, normalisation \n")
        file.write("{}, {}, {}, {}, {}, {}, {} \n".format(break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation))
        file.close()
    
    params = fit_type, break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation
    logger.info(color_text('Exitting function.',Colors.MediumSpringGreen))
    return(params)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def spectral_models(frequency, luminosity, fit_type, break_frequency, injection_index, remnant_ratio, normalisation, bessel_x, bessel_F):
    """
    (usage) The standard forms of the JP, KP and CI models.  
    
    parameters
    ----------
    frequency       : input frequencies
                      (type = 1darray)
    luminosity      : input flux densities
                      (type = 1darray)
    fit_type        : 
    break_frequency :
    injection_index :
    remnant_ratio   :
    normalisation   :
    bessel_x        :
    bessel_F        :
        
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
                elif fit_type in ['TKP', 'TJP', 'TCI']:
                        raise Exception('Tribble form supplied to spectral_models')
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

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def spectral_models_tribble(frequency, luminosity, fit_type, bfield, redshift, break_frequency, injection_index, remnant_ratio, normalisation, bessel_x, bessel_F):
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
    err_model_width : int
        range of the model uncertainty envelope in sigma
    n_remnants : int
        The remnant ratio range
    n_model_freqs : int
        The number of plotting frequencies
    mc_length : int
        Number of MC iterations
        
    returns
    -------
    luminosity_predict : 1darray
        fitted flux density for given frequency list
    normalisation : float
        normalisation factor for correct scaling
    """
    # define constants (SI units)
    c = 2.99792458e+8       # light speed
    me = 9.10938356e-31 # electron mass
    mu0 = 4*np.pi*1e-7  # magnetic permeability of free space
    e = 1.60217662e-19  # charge on electron
    sigmaT = 6.6524587158e-29 # electron cross-section
    bfield = bfield * 1e-9

    if fit_type == 'TJP' or fit_type == 'TKP':
        remnant_ratio = 0
    nalpha, nfields, nenergiesJP, nenergiesCI = 32, 32, 32, 32 # can be increased for extra precision
    nenergies = nenergiesJP + nenergiesCI
    
    # calculate the best fit to frequency-luminosity data
    luminosity_sum, predict_sum, nfreqs = 0., 0., 0
    luminosity_predict = np.zeros(len(frequency))
    
    # calculate the synchrotron age if B field provided as model parameter
    const_a = bfield/np.sqrt(3)
    const_synage = np.sqrt(243*np.pi)*me**(5./2)*c/(2*mu0*e**(7./2))
    Bic = 0.318*((1 + redshift)**2)*1e-9
    t_syn = const_synage*bfield**0.5/(bfield**2 + Bic**2)/np.sqrt(break_frequency*(1 + redshift))
    
    for freqPointer in range(0, len(frequency)):
        if (normalisation > 0 or luminosity[freqPointer] > 0): # either valid luminosity data or not fitting
            
            luminosity_predict[freqPointer] = 0. # should already be zero
            
            # integrate over magnetic field (as B)
            B_min = np.log10(const_a) - 4
            B_max = np.log10(const_a) + 4
            for i in range(0, nfields):
            
                # set up numerical integration
                B = (10**((i + 1)*(B_max - B_min)/nfields + B_min) + 10**(i*(B_max - B_min)/nfields + B_min))/2
                dB = 10**((i + 1)*(B_max - B_min)/nfields + B_min) - 10**(i*(B_max - B_min)/nfields + B_min)
                            
                # integral over pitch angle
                alpha_min = 0
                alpha_max = np.pi/2 # should be pi, but sin(alpha) is symmetric about pi/2
                for j in range(0, nalpha):
                    
                    # set up numerical integration
                    alpha = ((j + 0.5)/nalpha)*(alpha_max - alpha_min) + alpha_min
                    dalpha = (alpha_max - alpha_min)/nalpha
                    
                    # integrate over energy (as E): TRIBBLE
                    if (fit_type == 'TCI' or fit_type == 'TJP'):
                        const_losses = 4*sigmaT*(B**2 + Bic**2)/(3*me**2*c**3)/(2*mu0)
                    elif (fit_type == 'TKP'):
                        const_losses = 4*sigmaT*((B*np.sin(alpha))**2 + Bic**2)/(3*me**2*c**3)/(2*mu0)
                    elif fit_type in ['KP', 'JP', 'CI']:
                        raise Exception('Standard form supplied to spectral_models_tribble')
                    else:
                        raise Exception('Spectral fit must be either \'TCI\', \'TJP\' or \'TKP\' model.')
                    E_crit = np.log10(1./(const_losses*t_syn))
                    E_min = E_crit - 4 # can be increased for extra precision away from break
                    E_max = E_crit + 4
                                
                    for k in range(0, nenergies):
                        
                        # set up numerical integration, allowing for different spacing above and below x_crit
                        if (k < nenergiesJP):
                            E = (10**((k + 1)*(E_crit - E_min)/nenergiesJP + E_min) + 10**(k*(E_crit - E_min)/nenergiesJP + E_min))/2 # trapezoidal rule
                            dE = 10**((k + 1)*(E_crit - E_min)/nenergiesJP + E_min) - 10**(k*(E_crit - E_min)/nenergiesJP + E_min)
                        else:
                            E = (10**((k + 1 - nenergiesJP)*(E_max - E_crit)/nenergiesCI + E_crit) + 10**((k - nenergiesJP)*(E_max - E_crit)/nenergiesCI + E_crit))/2
                            dE = 10**((k + 1 - nenergiesJP)*(E_max - E_crit)/nenergiesCI + E_crit) + 10**((k - nenergiesJP)*(E_max - E_crit)/nenergiesCI + E_crit)
                    
                        # calculate x, dx, x_crit and x_crit_star
                        x = 4*np.pi*me**3*(c**4)*frequency[freqPointer]/(3*e*E**2*B*np.sin(alpha))
                        dx = (8*np.pi*me**3*c**4*frequency[freqPointer]/(3*e*E**3*B*np.sin(alpha))*dE)*( 4*np.pi*me**3*c**4*frequency[freqPointer]/(3*e*E**2*B**2*np.sin(alpha))*dB)
                        
                        if (fit_type == 'TCI' or fit_type == 'TJP'):
                            x_crit = np.log10(frequency[freqPointer]/(break_frequency*np.sin(alpha)))
                        elif (fit_type == 'TKP'):
                            x_crit = np.log10(frequency[freqPointer]*np.sin(alpha)**3/(break_frequency))
                        else:
                            raise Exception('Spectral fit must be either \'TCI\', \'TJP\' or \'TKP\' model.')
                        if remnant_ratio > 0:
                            x_crit_star = np.log10(frequency[freqPointer]/(break_frequency*np.sin(alpha))*remnant_ratio**2)
                        else:
                            x_crit_star = -307. # close to zero in log space
                    
                        # calculate the spectrum for JP, KP or CI/off models
                        if (x > 10**x_crit):
                            if (fit_type == 'TCI'):
                                if remnant_ratio > 0:
                                    N_x = x**(-1./2)*((np.sqrt(x) - 10**(x_crit_star/2))**(injection_index - 1) - (np.sqrt(x) - 10**(x_crit/2))**(injection_index - 1))
                                else:
                                    N_x = x**((injection_index - 2)/2.)*(1 - x**((1 - injection_index)/2.)*(np.sqrt(x) - 10**(x_crit/2))**(injection_index - 1))
                            elif (fit_type == 'TJP' or fit_type == 'TKP'):
                                N_x = x**(-1./2)*(np.sqrt(x) - 10**(x_crit/2))**(injection_index - 2)
                        elif (x > 10**x_crit_star): # only CI-off model should meet this condition
                            if (fit_type == 'TCI'):
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
                        if (fit_type == 'TCI'):
                            luminosity_predict[freqPointer] = luminosity_predict[freqPointer] + frequency[freqPointer]**(-injection_index/2.)*np.sin(alpha)**((injection_index + 4)/2.)*F_x*N_x*dx*dalpha *B**2*np.exp(-B**2/(2*const_a))
                        elif (fit_type == 'TJP'):
                            luminosity_predict[freqPointer] = luminosity_predict[freqPointer] + frequency[freqPointer]**((1 - injection_index)/2.)*np.sin(alpha)**((injection_index + 3)/2.)*F_x*N_x*dx*dalpha *B**2*np.exp(-B**2/(2*const_a))
                        elif (fit_type == 'TKP'):
                            luminosity_predict[freqPointer] = luminosity_predict[freqPointer] + frequency[freqPointer]**((1 - injection_index)/2.)*np.sin(alpha)**((3*injection_index + 1)/2.)*F_x*N_x*dx*dalpha *B**2*np.exp(-B**2/(2*const_a))

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

def spectral_data(params, frequency_observed=None, n_model_freqs=100, mc_length=10, err_model_width=2, bfield=None, z=None, work_dir=None, write_model=None, options=None):
    """
    (usage) Uses the optimized parameters to return a 1darray of model flux densities for a given frequency list. An uncertainty envelope on the model is calculated following an MC approach. 
    
    parameters
    ----------
    params : tuple
        Contains the fit_type, break_predict, dbreak_predict, inject_predict, dinject_predict, remnant_predict, dremnant_predict, normalisation
    frequency_observed : 1darray
        The list of observed frequencies. If None, model is evaluated between 50 MHz to 50 GHz. 
    n_model_freqs : int
        The number of frequencies at which to evaluate the model
    mc_length : int
        Number of MC iterations used for uncertainty estimation
    err_model_width : int
        Width of the uncertainty level corresponding to the model
    work_dir : str
        If not None, writes outputs to this directory
    write_model : Bool
        If True, writes outputs
        
    returns
    -------
    spectral_model_plot_data : matrix : np.shape(matrix) = (5 x n_model_freqs)
        contains the plotting frequency, model evaluated at the plotting frequency, error in the model, model lower bound, model upper bound. 
    luminosity_model_observed : 1darray
        If frequency_observed != None, luminosity_model_observed gives the model evaluated over the observed frequencies
    """
    if options is not None:
        bfield = options.bfield
        z = options.z
        work_dir = options.work_dir
        write_model = options.write_model
    logger.info(color_text('Entered function.',Colors.MediumSpringGreen))
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
    if not isinstance(remnant_predict, float) or remnant_predict < 0:
        raise Exception(color_text('remnant_predict needs to be a float and greater than zero',Colors.Red))
    if not isinstance(dremnant_predict, (float, int)) or dremnant_predict < 0:
        raise Exception(color_text('dremnant_predict needs to be a float/int and greater than zero',Colors.Red))
    if not isinstance(normalisation, float) or normalisation < 0:
        raise Exception(color_text('normalisation needs to be a float and greater than zero',Colors.Red))
    if fit_type in ['TKP', 'TJP', 'TCI']:
        if z is None or bfield is None:
            raise Exception(color_text('{} requires a redshift and magnetic field strength.'.format(fit_type),Colors.Red))
    if bfield is not None and (not isinstance(bfield, (float, int)) or bfield < 0):
        raise Exception(color_text('Magnetic field strenth needs to be a float/int and greater than zero',Colors.Red))
    if z is not None and (not isinstance(z, (float, int)) or z < 0):
        raise Exception(color_text('Redshift needs to be a float/int and greater than zero',Colors.Red))

    bessel_x, bessel_F = besselK53()

    # Determine whether to evaluate the model at the observed frequencies 
    if frequency_observed is not None:
        # evaluate the model at the observed frequencies, and use the observed frequencies to constrain the bounds of the plotting_frequency
        frequency_model_simulated = np.geomspace(10**(math.floor(np.min(np.log10(frequency_observed)))),10**(math.ceil(np.max(np.log10(frequency_observed)))), num=n_model_freqs)
        if fit_type in ['JP', 'KP', 'CI']:
            luminosity_model_observed = spectral_models(frequency_observed, np.zeros(len(frequency_observed)), fit_type, 10**break_predict, inject_predict, remnant_predict, normalisation, bessel_x, bessel_F)[0]
        elif fit_type in ['TJP', 'TKP', 'TCI']:
            luminosity_model_observed = spectral_models_tribble(frequency_observed, np.zeros(len(frequency_observed)), fit_type, bfield, z, 10**break_predict, inject_predict, remnant_predict, normalisation, bessel_x, bessel_F)[0]
    else:
        # default the plotting frequencies between 50 MHz and 50 GHz
        luminosity_model_observed = None
        frequency_model_simulated = np.geomspace(5e+7, 5e+10, num=n_model_freqs)
    
    # Evaluate the model over a list of simulated plotting frequencies
    if fit_type in ['JP', 'KP', 'CI']:
        luminosity_model_simulated = spectral_models(frequency_model_simulated, np.zeros(len(frequency_model_simulated)), fit_type, 10**break_predict, inject_predict, remnant_predict, normalisation, bessel_x, bessel_F)[0]
    elif fit_type in ['TJP', 'TKP', 'TCI']:
        luminosity_model_simulated = spectral_models_tribble(frequency_model_simulated, np.zeros(len(frequency_model_simulated)), fit_type, bfield, z, 10**break_predict, inject_predict, remnant_predict, normalisation, bessel_x, bessel_F)[0]
    
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
            ax.hist(free_param[i])
            ax.set_xlabel(free_param_name[i], fontsize=10)
            if fit_type is not None:
                filename = '{}_{}_distribution.pdf'.format(fit_type,free_param_name[i])
            else:
                filename = '{}_distribution.pdf'.format(free_param_name[i])
            if work_dir is not None:
                savename = '{}/{}'.format(work_dir, filename)
            else:
                savename = filename
            plt.savefig(savename,dpi=200)
    
    # instantiate array to store Monte-Carlo simulated spectra
    luminosityArray = np.zeros([mc_length,len(frequency_model_simulated)])

    # evaluate and store the model spectrum for each set of free parameters
    for mcPointer in range(0,mc_length):
        print('MC iteration: {}'.format(mcPointer))
        if fit_type in ['JP', 'KP', 'CI']:
            fitmc, normmc = spectral_models(frequency_model_simulated, np.zeros(len(frequency_model_simulated)), fit_type, 10**break_predict_vec[mcPointer], inject_predict_vec[mcPointer], remnant_predict_vec[mcPointer], normalisation, besselK53()[0], besselK53()[1])
        elif fit_type in ['TJP', 'TKP', 'TCI']:
            fitmc, normmc = spectral_models_tribble(frequency_model_simulated, np.zeros(len(frequency_model_simulated)), fit_type, bfield, z, 10**break_predict_vec[mcPointer], inject_predict_vec[mcPointer], remnant_predict_vec[mcPointer], normalisation, bessel_x, bessel_F)
        luminosityArray[mcPointer] = (np.asarray(fitmc))
    
    # instantiate vectors to store the uncertainties in the model at the corresponding frequency
    err_luminosity_model_simulated=np.zeros([len(frequency_model_simulated)])
    luminosity_model_simulated_min=np.zeros([len(frequency_model_simulated)])
    luminosity_model_simulated_max=np.zeros([len(frequency_model_simulated)])
    
    # take the std dev in the model at each frequency to evaluate a model uncertainty
    colorstring = color_text("Estimating model errors from {} Monte-Carlo iterations".format(mc_length), Colors.DogderBlue)
    logger.info(colorstring)
    for plotfreqPointer in range(0,len(frequency_model_simulated)):
        err_luminosity_model_simulated[plotfreqPointer] = np.std(luminosityArray.T[plotfreqPointer])
        luminosity_model_simulated_min[plotfreqPointer] = luminosity_model_simulated[plotfreqPointer] - 0.5*err_model_width*np.std(luminosityArray.T[plotfreqPointer])
        luminosity_model_simulated_max[plotfreqPointer] = luminosity_model_simulated[plotfreqPointer] + 0.5*err_model_width*np.std(luminosityArray.T[plotfreqPointer])
    
        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_axes([0.1,0.1,0.85,0.85])
        # ax.hist((luminosityArray.T[plotfreqPointer]))
        # std = np.std((luminosityArray.T[plotfreqPointer]))
        # ax.set_xlabel('{} {}'.format(frequency_model_simulated[plotfreqPointer], std))
        # plt.savefig('/home/sputnik/Documents/{}.pdf'.format(frequency_model_simulated[plotfreqPointer]))

    # stack the model frequency, model luminosity, error in model luminosity and the upper/lower bounds of the model in an array
    spectral_model_plot_data = np.empty([5,len(frequency_model_simulated)])
    spectral_model_plot_data[0] = frequency_model_simulated
    spectral_model_plot_data[1] = luminosity_model_simulated
    spectral_model_plot_data[2] = err_luminosity_model_simulated
    spectral_model_plot_data[3] = luminosity_model_simulated_min
    spectral_model_plot_data[4] = luminosity_model_simulated_max

    # write fitting outputs to file
    if write_model is not None and write_model == True:
        if fit_type is not None:
            filename = '{}_model_spectrum.dat'.format(fit_type)
        else:
            filename = 'model_spectrum.dat'
        if work_dir is not None:
            savename = '{}/{}'.format(work_dir, filename)
        else:
            savename = filename
        colorstring = color_text("Writing model spectrum to: {}".format(savename), Colors.DogderBlue)
        logger.info(colorstring)
    
        file = open(savename, "w")
        file.write("Frequency, {0} Model, unc {0} Model, {0} Model +- {1} sigma, {0} Model +- {1} sigma \n".format(fit_type, err_model_width))
        for i in range(0,len(frequency_model_simulated)):
            file.write("{}, {}, {}, {}, {} \n".format(frequency_model_simulated[i], luminosity_model_simulated[i], err_luminosity_model_simulated[i], luminosity_model_simulated_min[i], luminosity_model_simulated_max[i]))
        file.close()
    
    logger.info(color_text('Exitting function.',Colors.MediumSpringGreen))
    return(spectral_model_plot_data, luminosity_model_observed)

def spectral_plotter(frequency, luminosity, dluminosity, spectral_model_plot_data=None, luminosity_model_observed=None, work_dir=None, fit_type=None, err_model_width=None):
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
    spectral_model_plot_data : 
        output of spectral_data() function
    work_dir : str
        Directory to write plot to
    fit_type : str
        Print model type on figure
    err_model_width : int
        Print width of uncertainty envelope on figure
    """
    logger.info(color_text('Entered function.',Colors.MediumSpringGreen))
    # check inputs are correct type and shape
    if len(frequency) != len(luminosity) or len(frequency) != len(dluminosity) or len(luminosity) != len(dluminosity):
        raise Exception(color_text('The observed frequency, flux density, and flux density errors must be equal length', Colors.Red))
    if luminosity_model_observed is not None and len(luminosity_model_observed) != len(luminosity):
        raise Exception(color_text('Observed and model flux densities need to be same length',Colors.Red))
    # instantiate figure
    fig = plt.figure(figsize=(12,12))
    if luminosity_model_observed is not None:
        ax = fig.add_axes([0.1,0.15,0.86,0.81])
        ax.axes.get_xaxis().set_visible(False)
        ax0 = fig.add_axes([0.1, 0.06, 0.86, 0.08])
        ax0.axes.get_yaxis().set_ticks([1])
        ax0.set_xscale('log')
        ax0.set_xlim([10**(math.floor(np.min(np.log10(frequency)))),10**(math.ceil(np.max(np.log10(frequency))))])
        ax0.set_xlabel('Frequency / Hz', fontsize=20)
        ax0.scatter(frequency, (luminosity/luminosity_model_observed), marker='.', c='black', zorder=1)
        ax0.tick_params(axis='both', labelsize=20, which='both', direction='in', length=5, width=2)
        ax0.plot([10**(math.floor(np.min(np.log10(frequency)))),10**(math.ceil(np.max(np.log10(frequency))))],[1, 1], c='red')
        ax0.set_ylabel('$\\frac{data}{model}$', fontsize=25)
    else:
        ax = fig.add_axes([0.1,0.1,0.86,0.86])
        ax.set_xlabel('Frequency / Hz', fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Integrated flux density / Jy', fontsize=20)
    ax.set_xlim([10**(math.floor(np.min(np.log10(frequency)))),10**(math.ceil(np.max(np.log10(frequency))))])
    ax.set_ylim([0.2*np.min(luminosity),5*np.max(luminosity)])
    ax.tick_params(axis='both', labelsize=20, which='both', direction='in', length=5, width=2)

    # plot the observed data
    ax.scatter(frequency, luminosity, marker='.', c='black', label='Data', zorder=1)
    ax.errorbar(frequency, luminosity, xerr=0, yerr=dluminosity, color='black', capsize=1, linestyle='None',hold=True, fmt='none',alpha=0.9)

    # if supplied, unpack and overlay the simulated model
    if spectral_model_plot_data is not None:
        frequency_model_simulated = spectral_model_plot_data[0]
        luminosity_model_simulated = spectral_model_plot_data[1]
        err_luminosity_model_simulated = spectral_model_plot_data[2]
        luminosity_model_simulated_min = spectral_model_plot_data[3]
        luminosity_model_simulated_max = spectral_model_plot_data[4]
        if fit_type is not None:
            ax.plot(frequency_model_simulated, luminosity_model_simulated, c='C0', label='{} Model'.format(fit_type), zorder=2)
        else:
            ax.plot(frequency_model_simulated, luminosity_model_simulated, c='C0', label='Model', zorder=2)
        ax.fill_between(frequency_model_simulated, luminosity_model_simulated_min.T, luminosity_model_simulated_max.T, color='purple', alpha=0.15)

    # work out the appropriate save name
    extension='.pdf'
    if fit_type is not None:
        plotname='{}_model_fit{}'.format(fit_type,extension)
    else:
        plotname='model_fit{}'.format(extension)
    if work_dir is not None:
        savename='{}/{}'.format(work_dir,plotname)
    else:
        savename=plotname

    # write figure to file
    colorstring = color_text("Writing figure to: {}".format(savename), Colors.DogderBlue)
    logger.info(colorstring)
    plt.legend(loc='upper right', fontsize=20)
    plt.savefig(savename,dpi=200)
    
    logger.info(color_text('Exitting function.',Colors.MediumSpringGreen))

def spectral_ages(params, B, z):
    """
    (usage) Derives the total, active and inactive spectral age using the break frequency, quiescent fraction, magnetic field strength and redshift.
    
    parameters
    ----------
    params: tuple
        Contains the fitted model (fit_type), break frequency (vb), and quiescent fraction (T), each constrained by spectral_fitter.
    B : float
        The magnetic field strength (nT)
    z : float
        The cosmological redshift. (dimensionless)

    returns
    -------
    tau : float
        The spectral age of the radio emission in Myr
    t_on : float
        Duration of active phase in Myr
    t_off : float
        Duration of remnant phase in Myr
    """
    logger.info(color_text('Entered function.',Colors.MediumSpringGreen))

    fit_type, vb, T = params # unpack values
    if not isinstance(vb, (float)) or vb < 0:
        raise Exception(color_text('Break frequency needs to be a float and greater than zero.', Colors.Red))
    if not isinstance(T, (float)) or T < 0:
        raise Exception(color_text('Quiescent fraction needs to be a float and greater than zero.', Colors.Red))
    if not isinstance(B, (float)) or B <= 0:
        raise Exception(color_text('Magnetic field strength needs to be a float and greater than zero.', Colors.Red))
    if not isinstance(z, (float)) or z <= 0:
        raise Exception(color_text('Redshift needs to be a float and greater than zero.', Colors.Red))
    if not isinstance(params, tuple) or len(params) != 3:
        raise Exception(color_text('params needs to be a tuple with len(params)=3', Colors.Red))
    if not isinstance(fit_type, str) or fit_type not in ['KP', 'TKP', 'JP', 'TJP', 'CI', 'TCI']:
        raise Exception(color_text('fit_type needs to be one of: KP, TKP, JP, TJP, CI, TCI', Colors.Red))

    espace='                    '
    colorstring = color_text("Spectral ages are calculated assuming the following parameters:", Colors.DogderBlue)
    logger.info(colorstring)
    colorstring = color_text(" {} Model = {} \n {} Break frequency = {} Hz \n {} Quiescent fraction = {} \n {} Magnetic field strength = {} nT \n {} Redshift = {}".format(espace, fit_type, espace, vb, espace, T, espace, B, espace, z), Colors.Green)
    print(colorstring)

    # define constants (SI units)
    c = 2.99792458e+8       # light speed
    me = 9.10938356e-31 # electron mass
    mu0 = 4*np.pi*1e-7  # magnetic permeability of free space
    e = 1.60217662e-19  # charge on electron
    
    Bic = 0.318*((1+z)**2)*1e-9
    B = B*1e-9
    
    # get the right constant
    if fit_type in ['CI', 'JP']:
        v = ((243*np.pi*(me**5)*(c**2))/(4*(mu0**2)*(e**7)))**(0.5)
    elif fit_type == 'KP':
        v = (1/2.25)*(((243*np.pi*(me**5)*(c**2))/(4*(mu0**2)*(e**7)))**(0.5))
    
    tau = ((v*(B**(0.5)))/((B**2)+(Bic**2)))*((vb*(1+z))**(-0.5)) # seconds
    tau = tau/(3.154e+13) # Myr
    t_on = tau*(1-T)
    t_off = tau - t_on

    if fit_type == 'CI':
        colorstring = color_text("Spectral ages estimated for {} model:".format(fit_type), Colors.DogderBlue)
        logger.info(colorstring)
        colorstring = color_text(" {} Total age = {} Myr \n {} Active duration = {} Myr \n {} Remnant duration = {} Myr".format(espace, tau, espace, t_on, espace, t_off), Colors.Green)
        print(colorstring)
    elif fit_type in ['JP', 'KP']:
        colorstring = color_text("Spectral age estimated for {} model:".format(fit_type), Colors.DogderBlue)
        logger.info(colorstring)
        colorstring = color_text(" {} Total age = {} Myr".format(espace, tau), Colors.Green)
        print(colorstring)
    
    logger.info(color_text('Exitting function.',Colors.MediumSpringGreen))
    return(tau, t_on, t_off)

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
        colorstring = color_text("Input frequency registered as {}. No scaling required.".format(unit), Colors.DogderBlue)
        logger.info(colorstring)
        return(data)
    if unit == 'MHz':
        colorstring = color_text("Input frequency registered as {}. Scaling by 10^6.".format(unit), Colors.DogderBlue)
        logger.info(colorstring)
        return(1e+6*data)
    if unit == 'GHz':
        colorstring = color_text("Input frequency registered as {}. Scaling by 10^9.".format(unit), Colors.DogderBlue)
        logger.info(colorstring)
        return(1e+9*data)
    if unit == 'Jy':
        colorstring = color_text("Input flux density registered as {}. No scaling required.".format(unit), Colors.DogderBlue)
        logger.info(colorstring)
        return(data)
    if unit == 'mJy':
        colorstring = color_text("Input flux density registered as {}. Scaling by 10^(-3).".format(unit), Colors.DogderBlue)
        logger.info(colorstring)
        return(1e-3*data)
    if unit == 'uJy':
        colorstring = color_text("Input flux density registered as {}. Scaling by 10^(-6).".format(unit), Colors.DogderBlue)
        logger.info(colorstring)
        return(1e-6*data)
