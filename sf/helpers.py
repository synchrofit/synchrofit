#!/usr/bin/env python

__author__ = "Benjamin Quici, Ross J. Turner"
__date__ = "25/02/2021"

"""
Helper functions and classes for synchrofit's core modules.
"""

import logging
import numpy as np

logging.basicConfig(format="%(levelname)s (%(funcName)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Colors:
    DodgerBlue = (30, 144, 255)
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

class Const:
    # define constants (SI units)
    c   = 2.99792458e+8  # light speed
    me  = 9.10938356e-31 # electron mass
    mu0 = 4*np.pi*1e-7   # magnetic permeability osf free space
    e   = 1.60217662e-19 # charge on electron

class CheckFunctionInputs:
    def spectral_fitter(frequency,
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
        redshift):

        # check input spectrum is a list or numpy array
        if (not isinstance(frequency, (list, np.ndarray))) or (not isinstance(luminosity, (list, np.ndarray))) or (not isinstance(dluminosity, (list, np.ndarray))):
            raise Exception(color_text('Expected <class "list"> or <class "numpy.ndarray"> for input frequency, flux density and uncertainty.',Colors.Red))

        # check lengths are equal
        if len(frequency) != len(luminosity) or len(frequency) != len(dluminosity):
            raise Exception(color_text('Expected equal length for frequency, flux density, and flux density error lists.',Colors.Red))
        
        # check fit_type is correct
        if not isinstance(fit_type, str):
            raise Exception(color_text('Expected string type for "fit_type".',Colors.Red))
        else:
            if fit_type not in ['KP', 'TKP', 'JP', 'TJP', 'CI', 'TCI']:
                raise Exception(color_text('fit_type needs to be a string and one of: "KP", "TKP", "JP", "TJP", "CI", "TCI"',Colors.Red))

        # simplify sampling the parameter grid if fitting KP or JP models
        if fit_type in ['KP', 'TKP', 'JP', 'TJP']:
            n_remnants = 1
            remnant_range = [0, 0]

        # ensure sensible values for n_injects
        if not isinstance(n_injects, int):
            raise Exception(color_text('Expected <class "int"> for n_injects.',Colors.Red))
        else:
            if n_injects < 1:
                raise Exception(color_text('n_injects cannot be less than 1.',Colors.Red))
        
        # ensure sensible values for n_breaks
        if not isinstance(n_breaks, int):
            raise Exception(color_text('Expected <class "int"> for n_breaks.',Colors.Red))
        else:
            if n_breaks < 1:
                raise Exception(color_text('n_breaks cannot be less than 1.',Colors.Red))

        # ensure sensible values for n_remnants
        if not isinstance(n_remnants, int):
            raise Exception(color_text('Expected <class "int"> for n_remnants.',Colors.Red))
        else:
            if n_remnants < 1:
                raise Exception(color_text('n_remnants cannot be less than 1.',Colors.Red))
        
        # ensure injection index range is sensible
        if isinstance(inject_range, (list, np.ndarray)):
            if len(inject_range) == 1:
                logger.warning(color_text('Single value received for inject_range: assuming the injection index is a known or fixed quantity.', Colors.Orange))
                if not isinstance(inject_range[0], (float, int)):
                    raise Exception(color_text('Expected <class "float"> or <class "int"> for inject_range.',Colors.Red))
                elif inject_range[0] < 0:
                    raise Exception(color_text('Value for inject_range cannot be negative.',Colors.Red))
                else:
                    inject_range = [inject_range[0], inject_range[0]]
                    n_injects = 1
            else:
                for inject in inject_range:
                    if not isinstance(inject, (float, int)):
                        raise Exception(color_text('Elements within inject_range must be float/int and cannot be negative',Colors.Red))
                    elif inject < 0:
                        raise Exception(color_text('Elements within inject_range cannot be negative.',Colors.Red))      
        elif isinstance(inject_range, (float, int)):
            logger.warning(color_text('Single value received for inject_range: assuming the injection index is a known or fixed quantity.', Colors.Orange))
            if inject_range < 0:
                raise Exception(color_text('Value for inject_range cannot be negative.',Colors.Red))
            else:
                inject_range = [inject_range, inject_range]
                n_injects = 1
        else:
            raise Exception(color_text('Unrecognized type for inject_range. Please refer to documentation.',Colors.Red))
        # check whether values are consistent with theoretical limits
        for inject in inject_range:
            if inject < 2.01:
                logger.warning(color_text('An injection index of {} is below what is expected from theory.'.format(inject), Colors.Orange))
            if inject > 2.99:
                logger.warning(color_text('An injection index of {} is above what is expected from theory.'.format(inject), Colors.Orange))

        # ensure break frequency range is sensible
        if isinstance(break_range, (list, np.ndarray)):
            if len(break_range) == 1:
                logger.warning(color_text('Single value received for break_range: assuming the injection index is a known or fixed quantity.', Colors.Orange))
                if not isinstance(break_range[0], (float, int)):
                    raise Exception(color_text('Expected <class "float"> or <class "int"> for break_range.',Colors.Red))
                elif break_range[0] < 0:
                    raise Exception(color_text('Value for break_range cannot be negative.',Colors.Red))
                else:
                    break_range = [break_range[0], break_range[0]]
                    n_breaks = 1
            else:
                for inject in break_range:
                    if not isinstance(inject, (float, int)):
                        raise Exception(color_text('Elements within break_range must be float/int and cannot be negative',Colors.Red))
                    elif inject < 0:
                        raise Exception(color_text('Elements within break_range cannot be negative.',Colors.Red))      
        elif isinstance(break_range, (float, int)):
            logger.warning(color_text('Single value received for break_range: assuming the injection index is a known or fixed quantity.', Colors.Orange))
            if break_range < 0:
                raise Exception(color_text('Value for break_range cannot be negative.',Colors.Red))
            else:
                break_range = [break_range, break_range]
                n_breaks = 1
        else:
            raise Exception(color_text('Unrecognized type for break_range. Please refer to documentation.',Colors.Red))

        # ensure remnant fraction range is sensible
        if isinstance(remnant_range, (list, np.ndarray)):
            if len(remnant_range) == 1:
                logger.warning(color_text('Single value received for remnant_range: assuming the injection index is a known or fixed quantity.', Colors.Orange))
                if not isinstance(remnant_range[0], (float, int)):
                    raise Exception(color_text('Expected <class "float"> or <class "int"> for remnant_range.',Colors.Red))
                elif remnant_range[0] < 0:
                    raise Exception(color_text('Value for remnant_range cannot be negative.',Colors.Red))
                else:
                    remnant_range = [remnant_range[0], remnant_range[0]]
                    n_remnants = 1
            else:
                for inject in remnant_range:
                    if not isinstance(inject, (float, int)):
                        raise Exception(color_text('Elements within remnant_range must be float/int and cannot be negative',Colors.Red))
                    elif inject < 0:
                        raise Exception(color_text('Elements within remnant_range cannot be negative.',Colors.Red))      
        elif isinstance(remnant_range, (float, int)):
            logger.warning(color_text('Single value received for remnant_range: assuming the injection index is a known or fixed quantity.', Colors.Orange))
            if remnant_range < 0:
                raise Exception(color_text('Value for remnant_range cannot be negative.',Colors.Red))
            else:
                remnant_range = [remnant_range, remnant_range]
                n_remnants = 1
        else:
            raise Exception(color_text('Unrecognized type for remnant_range. Please refer to documentation.',Colors.Red))
        
        # convert all lists into arrays, if not already
        frequency = np.asarray(frequency)
        luminosity = np.asarray(luminosity)
        dluminosity = np.asarray(dluminosity)
        inject_range = np.asarray(inject_range)
        break_range = np.asarray(break_range)
        remnant_range = np.asarray(remnant_range)

        # if remnant_range is zero, assume CI-on model (e.g. no need to sample remnant_range parameter space)
        if fit_type == 'CI' and all([x == 0 for x in remnant_range]):
            logger.info(color_text('Recieved zero for remnant range, assuming CI-on model.', Colors.DodgerBlue))
            n_remnants = 1
        
        # n_iterations
        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise Exception(color_text('n_iterations needs to be an integer and greater than 0',Colors.Red))

        # ensure sensible inputs for Tribble models
        if fit_type in ['TKP', 'TJP', 'TCI']:
            if redshift is None:
                raise Exception(color_text('{} requires a redshift.'.format(fit_type),Colors.Red))
            else:
                if redshift < 0:
                   raise Exception(color_text('Redshift cannot be negative.'.format(fit_type),Colors.Red)) 
            if b_field is None:
                raise Exception(color_text('{} requires a magnetic field strength.'.format(fit_type),Colors.Red))
            else:
                if b_field < 0:
                   raise Exception(color_text('Magnetic field cannot be negative.'.format(fit_type),Colors.Red))

        # print accepted parameters
        espace='                      '
        colorstring = color_text("Fitting options accepted:", Colors.DodgerBlue)
        logger.info(colorstring)
        colorstring=color_text(" {} fit_type = {} \n {} inject_range = {} \n {} n_injects = {} \n {} n_breaks = {} \n {} break_range = {} \n {} n_remnants = {} \n {} remnant_range = {}".format(espace, fit_type,espace,inject_range, espace, n_injects, espace, n_breaks, espace, break_range, espace, n_remnants, espace, remnant_range), Colors.Green)
        print(colorstring)

        return(frequency, luminosity, dluminosity, inject_range, break_range, remnant_range, n_injects, n_breaks, n_remnants)
    
    def spectral_model():
        pass
    def spectral_plotter():
        pass
