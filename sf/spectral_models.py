#!/usr/bin/env python

__author__ = "Benjamin Quici, Ross J. Turner"
__date__ = "29/05/2021"

import numpy as np
from numba import jit, njit

"""
Functions describing the standard and tribble forms of the JP, KP and CI models.
"""
#TODO: investigate parallel=True mode in numba

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

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __spectral_models_standard(frequency, 
    luminosity, 
    fit_type, 
    break_frequency,
    injection_index, 
    remnant_ratio, 
    normalisation, 
    besselK53=besselK53()):
    """
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
    """ 
    # unpack besselK53 function
    bessel_x, bessel_F = besselK53

    if fit_type == 'JP' or fit_type == 'KP':
        remnant_ratio = 0

    nalpha, nenergiesJP, nenergiesCI = 64, 64, 64 # can be increased for extra precision
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
    return(luminosity_predict, normalisation)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __spectral_models_tribble(frequency,
    luminosity, 
    fit_type, 
    b_field, 
    redshift, 
    break_frequency, 
    injection_index, 
    remnant_ratio, 
    normalisation, 
    besselK53=besselK53()):
    """
    (usage) The Tribble forms of the JP, KP and CI models.  
    
    Parameters
    ----------
    frequency       : an input list of frequencies (Hz).

    luminosity      : an input list of flux densities (Jy).

    fit_type        : the spectral model, must be one of [TKP, TJP, TCI].

    b_field         : the magnetic field strength (T)

    redshift        : the cosmological redshift (dimensionless)

    break_frequency : the break frequency (Hz).

    injection_index : the energy injection index, s. (dimensionless).

    remnant_ratio   : the remnant fraction, e.g. the fractional inactive time (dimensionless).

    normalisation   : the normalisation factor (dimensionless).
        
    Returns
    -------
    luminosity_predict : fitted flux density for given frequency list

    normalisation      : normalisation factor for correct scaling of 
    """ 
    
    # define constants (SI units)
    c = 2.99792458e+8         # light speed
    me = 9.10938356e-31       # electron mass
    mu0 = 4*np.pi*1e-7        # magnetic permeability of free space
    e = 1.60217662e-19        # charge on electron
    sigmaT = 6.6524587158e-29 # electron cross-section

    # unpack besselK53 function
    bessel_x, bessel_F = besselK53

    if fit_type == 'TJP' or fit_type == 'TKP':
        remnant_ratio = 0
    nalpha, nfields, nenergiesJP, nenergiesCI = 32, 32, 32, 32 # can be increased for extra precision
    nenergies = nenergiesJP + nenergiesCI
    
    # calculate the best fit to frequency-luminosity data
    luminosity_sum, predict_sum, nfreqs = 0., 0., 0
    luminosity_predict = np.zeros(len(frequency))
    
    # calculate the synchrotron age if B field provided as model parameter
    const_a = b_field/np.sqrt(3)
    const_synage = np.sqrt(243*np.pi)*me**(5./2)*c/(2*mu0*e**(7./2))
    Bic = 0.318*((1 + redshift)**2)*1e-9
    t_syn = const_synage*b_field**0.5/(b_field**2 + Bic**2)/np.sqrt(break_frequency*(1 + redshift))
    
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
    return(luminosity_predict, normalisation)