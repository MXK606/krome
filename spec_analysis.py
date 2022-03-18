#!/usr/bin/env python
# coding: utf-8

"""
spec_analysis.py: This python module contains supplementary functions used by the index_calc.py module along with other function for further analysis.

"""

__author__ = "Mukul Kumar"
__email__ = "Mukul.k@uaeu.ac.ae, MXK606@alumni.bham.ac.uk"
__date__ = "10-03-2022"
__version__ = "1.2"

import numpy as np
import pandas as pd
from astropy.io import fits
import csv
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as log_progress
from PyAstronomy import pyasl
from astropy.timeseries import LombScargle


## Defining a function to find the index of a line containing a certain string

def find_string_idx(out_file_path, string):
    
    """
    
    Finds the index of the line containing the given string in the given file.
    
    Parameters:
    -----------
    out_file_path: str
    Path of the .out file
    
    string: str
    String to check for in the .out file
    
    Returns:
    --------
    Line index: int
    Index of the line in the .out file that contains the given string
    
    
    """
    
    file = open(out_file_path, "r")
    
    flag = 0
    index = 0
    
    for line in file:
        index += 1
        
        if string in line:
            
            flag = 1
            break
    
    if flag == 0:
        print('String', string , 'not found in .out file')
        print('----------------------------------------------------------------------------------------------------------------')
        
        return float('nan')
    else:
        return index-1
    
    file.close()
    

## Defining a function to extract important object parameters from the .out file

def obj_params(out_file_path):
    
    """
    
    Extracts useful object parameters from the .out file.
    
    Parameters:
    -----------
    out_file_path: str
    Path of the .out file
    
    Returns:
    --------
    object parameters: dict
    Dictionary containing the usefule object parameters
    
    """
    
    str_list = ['Number of stellar exposures in the sequence (2 or 4) :',
                'Detector gain (e/ADU) and read-out noise (e) :',
                '   Coordinates of object :',
                '   Time of observations :',
                '   Total exposure time :',
                '   Heliocentric Julian date (UTC) :',
                '         >>> vmag/teff estimate from sn curve (mag/K):']

    file = open(out_file_path).readlines() 
    
    idx = []

    for string in str_list:
        idx.append(find_string_idx(out_file_path, string))

    obj_parameters = {}

    try:
        obj_parameters['NUM_EXP'] = float(file[idx[0]][-2:-1])
    except TypeError:
        print('Object parameter for "{}" not found in the .out file'.format(str_list[0])) 
        print('----------------------------------------------------------------------------------------------------------------')
        
    try:
        obj_parameters['GAIN'] = float(file[idx[1]][-12:-1].split()[0])
    except TypeError:
        print('Object parameter for "{}" not found in the .out file'.format(str_list[1])) 
        print('----------------------------------------------------------------------------------------------------------------')
        
    try:
        obj_parameters['RON'] = float(file[idx[1]][-12:-1].split()[1])
    except TypeError:
        print('Object parameter for "{}" not found in the .out file'.format(str_list[1]))
        print('----------------------------------------------------------------------------------------------------------------')
        
    try:
        obj_parameters['RA'] = file[idx[2]][-26:-15].replace(' ', '')
    except TypeError:
        print('Object parameter for "{}" not found in the .out file'.format(str_list[2]))
        print('----------------------------------------------------------------------------------------------------------------')
        
    try:
        obj_parameters['Dec'] = file[idx[2]][-11:-1]
    except TypeError:
        print('Object parameter for "{}" not found in the .out file'.format(str_list[2]))
        print('----------------------------------------------------------------------------------------------------------------')
        
    try:
        obj_parameters['AIRMASS'] = float(file[idx[3]][-8:-3])
    except TypeError:
        print('Object parameter for "{}" not found in the .out file'.format(str_list[3]))
        print('----------------------------------------------------------------------------------------------------------------')
        
    try:
        obj_parameters['T_EXP'] = float(file[idx[4]][25:31])
    except TypeError:
        print('Object parameter for "{}" not found in the .out file'.format(str_list[4]))
        print('----------------------------------------------------------------------------------------------------------------')
        
    try:
        obj_parameters['HJD'] = float(file[idx[5]][-14:-1])
    except TypeError:
        print('Object parameter for "{}" not found in the .out file'.format(str_list[5]))
        print('----------------------------------------------------------------------------------------------------------------')
        
    try:
        obj_parameters['V_mag'] = float(file[idx[6]][-11:-1].split()[0])
    except TypeError:
        print('Object parameter for "{}" not found in the .out file'.format(str_list[6]))
        print('----------------------------------------------------------------------------------------------------------------')
        
    try:
        obj_parameters['T_eff'] = float(file[idx[6]][-11:-1].split()[1])
    except TypeError:
        print('Object parameter for "{}" not found in the .out file'.format(str_list[6]))
        print('----------------------------------------------------------------------------------------------------------------')
        

    return obj_parameters

## Defining a function to retrieve spectral orders from NARVAL

def extract_orders(wav,
                   flx,
                   flx_err,
                   show_plot=False):
    
    """
    Extracts the overlapping spectral orders from the NARVAL .s files
    
    Parameters:
    -----------
    wav: array
    Spectrum wavelength 
    
    flx: array
    Spectrum flux 
    
    flx_err: array
    Spectrum flux error
    
    show_plot: bool, default: False
    Overplots all of the extracted spectral orders in different colors
    
    Returns:
    --------
    Returns a list containing each extracted order which includes the wav, flx and flx_err.
    
    """
    
    # Checking that the shapes of each input array are the same!
    if len(wav) == len(flx) == len(flx_err):
    
        jump_index = []
        
        # The loop below looks for disruption between two distinct data points in the increasing wavelength axis 
        # and appends their index to the empty list 'jump_index'
        #
        # For ex. wav = [1,2,3,4,5,2,3,4,5,6]
        # order_1 = [1,2,3,4,5]
        # order_2 = [2,3,4,5,6]
        
        for i in range(len(wav)-1):
            if wav[i+1] < wav[i]:
                jump_index.append(i)
                
        
        # Creating a loop for cases where wavelength axis does not contain individual spectral orders        
        if len(jump_index) > 1:
        
            spec_orders = []
            
            # For when certain files have nan as the flux errors;
            if np.isnan(np.sum(flx_err)):
                
                print('Flux errors array contains NaN values. Returning orders without errors for all!')
                
                order_0 = [wav[:jump_index[0]], flx[:jump_index[0]]]
                spec_orders.append(order_0)
                
                for i in range(len(jump_index)-1):
                    
                    order = [wav[jump_index[i]+1:jump_index[i+1]], flx[jump_index[i]+1:jump_index[i+1]]]
                    spec_orders.append(order)
                    
            else:
                
                # Creating the first spectral order which ends at jump_index[0]
                order_0 = [wav[:jump_index[0]], flx[:jump_index[0]], flx_err[:jump_index[0]]]
                spec_orders.append(order_0)
                
                # The loop below creates each spectral order form the jump_index list above and appends them to the spec_orders
                # list
                for i in range(len(jump_index)-1):
                    
                    order = [wav[jump_index[i]+1:jump_index[i+1]], 
                             flx[jump_index[i]+1:jump_index[i+1]], 
                             flx_err[jump_index[i]+1:jump_index[i+1]]]
                    spec_orders.append(order)
                
            if show_plot:
                
                plt.figure(figsize=(10,4))
                for i in range(len(spec_orders)):
                    # Plots each spectral order with different colors 
                    plt.plot(spec_orders[i][0], spec_orders[i][1])
                    
                plt.xlabel(r'$\lambda$(nm)')
                plt.ylabel('Normalised Flux')
                plt.title('{} spectral orders overplotted'.format(len(spec_orders)))
                plt.minorticks_on()
                plt.tick_params(direction='in', axis='both', which='both')
                
            return spec_orders
                
        else:
            print('No individual spectral orders found. The wavelength is linear.')
            
    else:
        raise ValueError("Input arrays must have the same shape.")
        
        
## Defining a function to reach check which orders the given lines are found in 
    
def check_lines(spectral_orders, line_names, lines, bandwidths):
    
        
    """
    
    Looks for which spectral order contains the given line ± the bandwidth/2, i.e. the region required for flux calculation.
        
    Parameters
    ----------
    
    spectral_orders: array
    Array containing all of the extracted spectral orders

    line_names: list
    List containing the names of each line
    
    lines: list
    List containing the lines to check
    
    bandwidths: list
    List conatining the bandwidths of the given lines
    
    Returns:
    --------
    
    List containing the index of the order(s) which contain the given line regions
    
    """
    
    idx = []
    
    for i in range(len(lines)):
        ln_idx = []
        
        for j in range(len(spectral_orders)):
            if spectral_orders[j][0].min() < (lines[i] - bandwidths[i]/2) and spectral_orders[j][0].max() > (lines[i] + bandwidths[i]/2):
                ln_idx.append(j)
        idx.append(ln_idx)

    for i in range(len(line_names)):
        
        print('The {} line is found within {} spectral order(s)'.format(line_names[i], len(idx[i])))
        for j in idx[i]:
            print('Order: #{}'.format(61-j)) # The orders begin from # 61 so to get the # number, we index as 61-idx.
        print('----------------------------------------------------------------------------------------------------------------')
        
    return idx

## Defining a function to read the given data

def read_data(file_path,
              Instrument,
              out_file_path=None,
              ccf_file_path=None,
              print_stat=True,
              show_plots=True):
    
    """
    
    Reads the data contained within the .s file and extract useful information from the .out file.
    
    Parameters:
    ----------
    
    file_path: str
    File path of the .s or .fits file
    
    Instrument: str
    Instrument type used. Available options: ['NARVAL', 'HARPS', 'HARPS-N']
    
    out_file_path: str, default=None
    File path of the .out file
    
    ccf_file_path: str, default=None
    File path of the CCF file
    
    print_stat: bool, default=True
    Prints the status of each process within the function.
    
    show_plots: bool, default=True
    Plots all overlapping spectral orders in one plot for NARVAL and plots the whole spectrum for the others.
    
    Returns:
    --------
    
    object parameters: dict (Only if out_file_path given for NARVAL)
    
    For NARVAL; spectral orders: list of pandas df
    
    For Others; spectrum: list
    
    """
    
    if Instrument=='NARVAL':
        
        if print_stat:
            print('Reading the data from the .s file: {}'.format(file_path))
            print('----------------------------------------------------------------------------------------------------------------')
        
        # Checking if its a Stokes V or I .s file using pandas
    
        df = pd.read_fwf(file_path, skiprows=2) # skipping first 2 rows of .s file
        
        # Defining the column names for both Stove V and I files
        col_names_V = ['Wavelength', 'Intensity', 'Polarized', 'N1', 'N2', 'I_err'] 
        col_names_I = ['Wavelength', 'Intensity', 'I_err']
        
        if len(df.columns)==6:
            data_spec = pd.read_csv(file_path, names=col_names_V, skiprows=2, sep=' ', skipinitialspace=True) 
            if print_stat:
                print('Stokes Profile: [V]')
                print('----------------------------------------------------------------------------------------------------------------')
        elif len(df.columns)==3:
            data_spec = pd.read_csv(file_path, names=col_names_V, skiprows=2, sep=' ', skipinitialspace=True)
            if print_stat:
                print('Stokes Profile: [I]')
                print('----------------------------------------------------------------------------------------------------------------')
        else:
            raise InputError('Input .s file contains unrecognisable number of columns. Recognised numbers are 3 (I profile) and 6 (V profile).')
            
        if print_stat:    
            print('----------------------------------------------------------------------------------------------------------------')
            print('Extracting all overlapping spectral orders')
            print('----------------------------------------------------------------------------------------------------------------')
        
        spec_orders = extract_orders(data_spec['Wavelength'],
                                     data_spec['Intensity'],
                                     data_spec['I_err'],
                                     show_plot=show_plots)
        
        if out_file_path != None:
            
            if print_stat:
                print('Extracting useful object parameters from the .out file: {}'.format(out_file_path))
                print('----------------------------------------------------------------------------------------------------------------')
            
            object_parameters = obj_params(out_file_path)
            
            keys = [i for i in object_parameters.keys()]
            vals = [j for j in object_parameters.values()]
            
            for i in range(len(keys)):
                print('{}: {}'.format(keys[i], vals[i]))
        
            return object_parameters, spec_orders
        
        else:
            return spec_orders
    
    elif Instrument=='HARPS':
        
        if print_stat:
            print('Reading the data from the .fits file: {}'.format(file_path))
            print('----------------------------------------------------------------------------------------------------------------')
        
        # Opening the FITS file using 'astropy.io.fits'
        # NOTE: The format of this FITS file must be ADP which contains the reduced spectrum with the wav, 
        # flux and flux_err in three columns
        
        file = fits.open(file_path)
        
        #Extracting useful information from the fits file header
        
        if print_stat:
            print('Extracting useful object parameters from the .fits file header')
            print('----------------------------------------------------------------------------------------------------------------')
        
        object_parameters = {}
        
        object_parameters['MJD'] = file[0].header['MJD-OBS'] # Modified Julian Date
        object_parameters['BJD'] = file[0].header['HIERARCH ESO DRS BJD'] # Barycentric Julian Date
        object_parameters['BERV'] = file[0].header['HIERARCH ESO DRS BERV'] # Barycentric Earth Radial Velocity  km/s 
        object_parameters['EXPTIME'] = file[0].header['EXPTIME'] # Exposure time in s
        object_parameters['OBS_DATE'] = file[0].header['DATE-OBS'] # Observation Date
        object_parameters['PROG_ID'] = file[0].header['PROG_ID'] # Program ID
        object_parameters['SNR'] = file[0].header['SNR'] # Signal to Noise ratio
        object_parameters['SIGDET'] = np.round(file[0].header['HIERARCH ESO DRS CCD SIGDET'], 3)  #CCD Readout Noise [e-]
        object_parameters['CONAD'] = file[0].header['HIERARCH ESO DRS CCD CONAD'] #CCD conversion factor [e-/ADU]; from e- to ADU
        object_parameters['RON'] = np.round((object_parameters['SIGDET'] * object_parameters['CONAD']), 3) #CCD Readout Noise [ADU]
        
        if ccf_file_path:
            if print_stat:
                print('Extracting RV from the CCF fits file: {}'.format(ccf_file_path))
                print('----------------------------------------------------------------------------------------------------------------')
            ccf_file = fits.open(ccf_file_path) # Opening the CCF FITS file to extract the RV
            object_parameters['RV'] = ccf_file[0].header['HIERARCH ESO DRS CCF RV']*1000 # Radial velocity converted from km/s to m/s
            
        else:
            object_parameters['RV'] = float('nan')
        
        keys = [i for i in object_parameters.keys()]
        vals = [j for j in object_parameters.values()]
    
        if print_stat:
            for i in range(len(keys)):
                print('{}: {}'.format(keys[i], vals[i]))
            print('----------------------------------------------------------------------------------------------------------------')
        
        # Defining each wavelength, flux and flux error arrays from the FITS file!
        
        wvl = file[1].data[0][0] # Å 
        flx = file[1].data[0][1] # Flux in ADU
        flx_err = file[1].data[0][2]
        
        spectrum = [wvl, flx, flx_err]
        
        # Plotting the spectrum
        
        if show_plots:
            
            f, ax  = plt.subplots(figsize=(10,4)) 
            ax.plot(spectrum[0], spectrum[1], '-k')  
            ax.set_xlabel('$\lambda$ (Å)')
            ax.set_ylabel("Flux (adu)")
            plt.minorticks_on()
            ax.tick_params(direction='in', which='both')
            plt.tight_layout()
            plt.show()
            
        return object_parameters, spectrum
    
    elif Instrument=='HARPS-N':
        
        if print_stat:
            print('Reading the data from the .fits file: {}'.format(file_path))
            print('----------------------------------------------------------------------------------------------------------------')
        
        # Opening the FITS file using 'astropy.io.fits'
        # NOTE: The format of this FITS file must be s1d which only contains flux array. 
        # The wavelength array is constructed using the starting point (CRVAL1), length of spectral axis (NAXIS1) 
        # and wavelength step (CDELT1) from the FITS file header.
        
        file = fits.open(file_path)
        
        #Extracting useful information from the fits file header
        
        if print_stat:
            print('Extracting useful object parameters from the .fits file header')
            print('----------------------------------------------------------------------------------------------------------------')
        
        object_parameters = {}
        
        object_parameters['MJD'] = file[0].header['MJD-OBS'] # Modified Julian Date
        object_parameters['EXPTIME'] = file[0].header['EXPTIME'] # Exposure time in seconds
        object_parameters['OBS_DATE'] = file[0].header['DATE-OBS'] # Observation Date
        object_parameters['PROG_ID'] = file[0].header['PROGRAM'] # Program ID
        
        if ccf_file_path:
            if print_stat:
                print('Extracting RV from the CCF fits file: {}'.format(ccf_file_path))
                print('----------------------------------------------------------------------------------------------------------------')
            ccf_file = fits.open(ccf_file_path)  # Opening the CCF FITS file to extract the RV
            object_parameters['RV'] = ccf_file[0].header['HIERARCH TNG DRS CCF RV']*1000 # Radial velocity converted from km/s to m/s
            
        else:
            object_parameters['RV'] = float('nan')
        
        keys = [i for i in object_parameters.keys()]
        vals = [j for j in object_parameters.values()]
        
        if print_stat:
            for i in range(len(keys)):
                print('{}: {}'.format(keys[i], vals[i]))
            print('----------------------------------------------------------------------------------------------------------------')
        
        # Defining each wavelength and flux arrays from the FITS file!
        # NOTE: No error column provided in the .fits file and no ReadOut noise as well to construct our own 
        # flux_err array!
        
        # constructing the spectral axis using start point, delta and axis length from file header
        wvl = file[0].header['CRVAL1'] + file[0].header['CDELT1']*np.arange(0, file[0].header['NAXIS1']) # Å
        flx = file[0].data # Flux in ADU
        
        spectrum = [wvl, flx]
        
        # Plotting the spectrum
        
        if show_plots:
            
            f, ax  = plt.subplots(figsize=(10,4)) 
            ax.plot(spectrum[0], spectrum[1], '-k')  
            ax.set_xlabel('$\lambda$ (Å)')
            ax.set_ylabel("Flux (adu)")
            plt.minorticks_on()
            ax.tick_params(direction='in', which='both')
            plt.tight_layout()
            plt.show()
            
        return object_parameters, spectrum
    
## Defining a function to calculate the LombScargle periodogram!
        
def LS_periodogram(x,
                   y,
                   dy,
                   minimum_frequency,
                   maximum_frequency,
                   samples_per_peak=10,
                   nterms=1, 
                   method='chi2',
                   normalization='model',
                   fap_method='bootstrap',
                   probabilities=None,
                   sampling_window_func=True,
                   show_plot=True,
                   save_fig=False,
                   fig_name=None):
    
    """
    Calculates and plots the astropy.timeseries.LombScargle periodogram showcasing the trial periods in log scale. 
    See https://docs.astropy.org/en/stable/timeseries/lombscargle.html for more info on default parameters.
    
    Parameters:
    -----------
    
    x: array
    Observation timestamps
    
    y: array
    Observation values
    
    dy: array
    Error on observation values
    
    minimum_frequency: int
    Minimum frequency to test
    
    maximum_frequency: int
    Minimum frequency to test
    
    samples_per_peak: int, default=10
    Number of sample frequencies to test per peak
    
    nterms: int, default=1
    Number of Fourier terms to use for the LombScargle model
    
    method: str, default='chi2'
    LombScargle implementation method. 
    
    normalization: str, default='model'
    Periodogram normalization method.
    
    fap_method: str, default='bootstrap'
    False Alarm Probability (FAP) calculation method.
    
    probabilities: str, default=None
    Probabilities to determine the False Alarm Levels (FALs) for.
    
    sampling_window_func: bool, default=True
    Calculates and plots the observation sampling windoe function periodogram following https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    
    show_plot: bool, default=True
    Plots the resulting periodogram
    
    save_fig: bool, default=False
    Saves the periodogram plot
    
    fig_name: str, default=None
    Name with which to save the plot
    
    Returns:
    --------
    
    frequency grid, periodogram power, array with index of periods with highest powers in decending order & if True; False Alarm Level and False Alarm Probabilities
    
    All of these are numpy.ndarray. 
    
    """
    
    # NOTE: All frequencies in LombScargle are not angular frequencies, but rather frequencies of oscillation (i.e., number of cycles per unit time).
    
    # The frequency limit on the low end is relatively easy: for a set of observations spanning a length of time T, 
    # a signal with frequency 1/T will complete exactly one oscillation cycle, and so this is a suitable minimum 
    # frequency to choose. 
    # The max. freq. is determined by the precision of the time measurements; min(delta_t)
    
    # The number of samples per peak is taken to be n0 = 10 (Debosscher et al. 2007; Richards et al. 2012) as it 
    # is common in literature. 
    
    # NOTE: setting nterms = 2 and method = 'fastchi2' might result in peak power greater than 1 and period with negative powers, which should not happen.  
    # This is because the method 'fastchi2' is an approximation that speeds the algorithm from O[N^2] to O[N log N], at the expense of not producing exact results. 
    # For values close to zero, this approximation can lead to negative powers. Changing the method to 'chi2', which supports multiple Fourier terms fixes this issue.
    # See the GitHub astropy issues page, https://github.com/astropy/astropy/issues/8454, for more info!
    
    # The frequencies are sampled by the autofrequency() method such that the delta_f = 1/n0*T 
    # The length of the freq. array is then given as N_evals = n0*T*f_max ! 
    
    if np.isnan(np.sum(dy)):
        raise ValueError('Error array "dy" contains NaN values')
    
    ls = LombScargle(x, y, dy, nterms=nterms, normalization=normalization)
    
    freq, power = ls.autopower(minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak, method=method)
    
    print('----------------------------------------------------------------------------------------------------------------')
    print('Total number of frequencies tested: {}'.format(len(freq)))
    print('----------------------------------------------------------------------------------------------------------------')
    print('Frequency grid resolution: {}'.format(np.diff(freq)[0]))
    print('----------------------------------------------------------------------------------------------------------------')
    
    sort_idx = np.argsort(-power) # minus symbol to sort the array from biggest to smallest. np.argsort resturns the indices that would sort the given array. Very useful!
    
    if nterms==1: # FAL and FAP's are not provided for multiple Fourier term periodograms!
        
        print('Calculating False Alarm Probabilities/Levels (FAPs/FALs) using the {} method'.format(fap_method))
        print('----------------------------------------------------------------------------------------------------------------')
        
        if probabilities != None:
            probabilities = probabilities 
        else:
            probabilities = [0.5, 0.2, 0.1]
        
        fal = ls.false_alarm_level(probabilities, method=fap_method, minimum_frequency=minimum_frequency, 
                                   maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak) 
        
        fap = ls.false_alarm_probability(power=power, method=fap_method, minimum_frequency=minimum_frequency, 
                                         maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak)
        
    print('Frequency at max. power is: {} which is {}d'.format(np.round(freq[sort_idx[0]], 4), np.round(1/freq[sort_idx[0]], 4)))
    print('----------------------------------------------------------------------------------------------------------------')
        
    if show_plot:
        
        plt.figure(figsize=(10,4))
        plt.plot(1/freq, power, color='black')
        plt.xlabel('Period (days)')
        plt.ylabel('Power') # Note here that the Lomb-Scargle power is always a unitless quantity, because it is related to the 𝜒2 of the best-fit periodic model at each frequency.
        plt.title('LombScargle Periodogram with x-axis as logarithmic period')
        plt.tick_params(axis='both', direction='in', which='both')
        plt.xscale('log')
        plt.xlim(1/maximum_frequency-0.1,1/minimum_frequency+0.5)
        
        
        if nterms == 1:
            for i in range(len(fal)):
                plt.plot([1/maximum_frequency,1/minimum_frequency], [fal[i]]*2, '--',
                         label="FAP = %4.1f%%" % (probabilities[i]*100))
            
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        if save_fig:
            plt.savefig('{}.pdf'.format(fig_name), format='pdf')
            
    if sampling_window_func:
        
        y = np.ones_like(x)
        
        ls_samp = LombScargle(x, y, fit_mean=False, center_data=False, normalization=normalization) # making sure to not pre-center the data or use a floating mean model.
        
        freq_samp, power_samp = ls_samp.autopower(minimum_frequency=minimum_frequency,
                                                  maximum_frequency=maximum_frequency,
                                                  samples_per_peak=samples_per_peak,
                                                  method=method)
        
        print('----------------------------------------------------------------------------------------------------------------')
        print('Sampling window period is: {}d'.format(np.round(1/freq_samp[np.argmax(power_samp)], 4)))
        print('----------------------------------------------------------------------------------------------------------------')
        
        if show_plot:
            
            plt.figure(figsize=(10,4))
            plt.plot(1/freq_samp, power_samp, color='black')
            plt.xlabel('Period (days)')
            plt.ylabel('Power') 
            plt.title('Sampling Window Function Periodogram')
            plt.tick_params(axis='both', direction='in', which='both')
            plt.xscale('log')
            plt.xlim(1/maximum_frequency-0.1,1/minimum_frequency+0.5)
            plt.tight_layout()
            plt.show()
            
            if save_fig:
                plt.savefig('{}_Sampling_Window.pdf'.format(fig_name), format='pdf')
            
            
    if nterms==1:
        return freq, power, sort_idx, freq_samp, power_samp, fal.value, fap
    else:
        return freq, power, sort_idx, freq_samp, power_samp
    
## Defining a function to calculate the best period model fit using LombScargle and plot it on either the BJD plot or the phase-folded plot!
    
def period_fit(x,
               y,
               dy,
               period,
               fit,
               normalization='model',
               ylabel=None,
               multi_term=False,
               save_fig=False,
               save_name=None):
    
    """
    Type Function Docstring Here!
    
    Parameters:
    -----------
    
    x: array
    Observation timestaps
    
    y: array
    Observation values
    
    dy: array
    Error on observation values
    
    period: int
    Orbital period of the model to fit
    
    fit: str
    Fit type; available options are 
    'BJD' which fits the model over the enitre observation timespan
    'phase' which phase folds the data before fitting the model form 0 - 1
    
    normalization: str, default='model'
    Periodogram normalization method. 
    See https://docs.astropy.org/en/stable/timeseries/lombscargle.html for more info.
    
    ylabel: str, default=None
    y-axis label for the period fit figure
    
    multi_term: bool, default=False
    Plots additional model fits obtained from periodograms consiting 2 and 3 Fourier terms.
    See https://docs.astropy.org/en/stable/timeseries/lombscargle.html for more info.
    
    save_fig: bool, default=False
    Saves the model fit plot as a PDF in the working directory
    
    save_name: str, default=None
    Name with which to save the plot PDF.
    
    Returns:
    --------
    If fit = 'BJD';
    model fit timestamps, model fit y values (model fit y values of the multi-term models if multi_term is True)
    
    If fit = 'phase':
    phase folded x values, model fit timestamps, model fit y values (model fit y values of the multi-term models if multi_term is True)
    
    All of these are type numpy.ndarray. 
    
    """
    
    
    if fit == 'BJD':
        
        t_fit = np.linspace(x.min() - 2450000, x.max() - 2450000, 10000)
        ls_1 = LombScargle(x - 2450000, y, dy, nterms=1, normalization=normalization)
        ls_2 = LombScargle(x - 2450000, y, dy, nterms=2, normalization=normalization)
        ls_3 = LombScargle(x - 2450000, y, dy, nterms=3, normalization=normalization)
        y_fit_1 = ls_1.model(t_fit, 1/period)
        y_fit_2 = ls_2.model(t_fit, 1/period)
        y_fit_3 = ls_3.model(t_fit, 1/period)
        
        plt.figure(figsize=(10,4))
        plt.errorbar(x - 2450000, y, yerr=dy, fmt='.k', capsize=5)
        plt.plot(t_fit, y_fit_1, '-r', label='Fundamental')
        
        if multi_term:
            plt.plot(t_fit, y_fit_2, '-b', label='Fundamental + 1st Harmonic')
            plt.plot(t_fit, y_fit_3, '-g', label='Fundamental + First 2 Harmonics')
            plt.legend()
        
        plt.xlabel('BJD - 2450000')
        if ylabel != None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel('Index')
        
        plt.tick_params(axis='both', direction='in', which='both')
        plt.tight_layout()
        plt.show()
        
        if save_fig:
            plt.savefig('{}.pdf'.format(save_name), format='pdf')
            
        if multi_term:
            return t_fit, y_fit_1, y_fit_2, y_fit_3
        else:
            return t_fit, y_fit_1
            
    elif fit == 'phase':
        
        t_fit = np.linspace(0.0, 1/best_frequency, 10000)
        ls_1 = LombScargle(x, y, dy, nterms=1, normalization=normalization)
        ls_2 = LombScargle(x, y, dy, nterms=2, normalization=normalization)
        ls_3 = LombScargle(x, y, dy, nterms=3, normalization=normalization)
        y_fit_1 = ls_1.model(t_fit, 1/period)
        y_fit_2 = ls_2.model(t_fit, 1/period)
        y_fit_3 = ls_3.model(t_fit, 1/period)
        
        phase_folded_x = pyasl.foldAt(x, period)
        
        plt.figure(figsize=(10,4))
        plt.errorbar(phase_folded_x, y, yerr=dy, fmt='ok', capsize=5)
        plt.plot(t_fit/period, y_fit_1, '-r', label='Fundamental')
        
        if multi_term:
            plt.plot(t_fit/period, y_fit_2, '-b', alpha=0.5, label='Fundamental + 1st Harmonic')
            plt.plot(t_fit/period, y_fit_3, '-g', alpha=0.5, label='Fundamental + First 2 Harmonics')
            plt.legend()
        
        plt.xlabel('Period Phase')
        
        if ylabel != None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel('Index')
            
        plt.tick_params(axis='both', direction='in', which='both')
        plt.tight_layout()
        plt.show()
        if save_fig:
            plt.savefig('{}.pdf'.format(save_name), format='pdf')
            
        if multi_term:
            return phase_folded_x, t_fit, y_fit_1, y_fit_2, y_fit_3
        else:
            return phase_folded_x, t_fit, y_fit_1
        
        
def find_nearest(array,
                 value):
    
    """
    Simple function that returns the index of an array giving the value closest to the given value.
    
    Parameters:
    -----------
    
    array: array
    Array from which to find to a closest value.
    
    value: float
    Floating value.
    
    Returns:
    --------
    Index, int.
    
    """
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Defining a function to calculate the true anomaly, i.e. orbital phase angle from the observation HJD

# T_e = 2455959 JD taken from http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2014AcA....64..323M
# converting T_e from JD to HJD using https://doncarona.tamu.edu/apps/jd/

# NOTE: These are all default parameters for the star GJ 436 and should be changed when using for another star

def ephemerides(file_path,
                P_orb=2.644,
                T_e=2455959.0039936211,
                e=0.152,
                P_rot=44.09,
                phase_start=2457464.49670, 
                print_stat=True,
                save_results=False,
                save_name=None):
    
    """
    
    Calculates the orbital and rotational phases for a star. 
    NOTE: The default parameters within this function are for the star GJ 436.
    
    Parameters:
    -----------
    
    file_path: str
    List of paths of the .out/.fits file containing the OBS_HJD/OBS_BJD
    
    P_orb: int, default=2.644
    Planetary orbital period in days. 
    Default value for GJ436b taken from http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2018A&A...609A.117T
    
    T_e: int, default=2455959.0039936211
    Epoch of periastron in HJD since the dates in .out files are HJD. Input in BJD instead if the given file_path is .fits. 
    Default value for GJ436b taken from http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2014AcA....64..323M
    
    e: int, default=0.152
    Orbital eccentricity. 
    Default value for GJ436b taken from http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2018A&A...609A.117T
    
    P_rot: int, default=44.09
    Stellar rotation period in days.
    Default value for GJ436 taken from https://ui.adsabs.harvard.edu/abs/2018Natur.553..477B/abstract
    
    phase_start: int, default=2457464.49670
    Starting point for the rotational phase. This ideally should be the first JD of your observation.
    Default value for GJ436b taken for the NARVAL 2016 observations downloaded from Polarbase.
    
    print_stat: bool, default=True
    Prints the status of each process within the function.
    
    save_results: bool, default=False
    Saves the results as a csv file.
    
    save_name: str, default=None
    Name with which to save the results file with.
    
    Returns:
    --------
    
    HJD/BJD, Number of orbits done since T_e, Mean anomaly, Eccentric anomaly, True anomaly, orbital phase, rotational phase
    
    All of these are floating points.
    
    """
    
    results = [] # Empty list to which the run results will be appended
    
    # Creating a loop to go through each given file_path in the list of file paths
    
    # Using the tqdm function 'log_progress' to provide a neat progress bar in Jupyter Notebook which shows the total number of
    # runs, the run time per iteration and the total run time for all files!
    
    for i in log_progress(range(len(file_path)), desc='Calculating System Ephemerides'):
        
        if file_path[i][-4:] == '.out':
            
            file = open(file_path[i]).readlines() # Opening the .out file and reading each line as a string
                        
            string = '   Heliocentric Julian date (UTC) :' # Creating a string variable that matches the string in the .out file
            
            idx = find_string_idx(file_path[i], string) # Using the 'find_string_idx' function to find the index of the line that contains the above string. 
            
            JD = float(file[idx][-14:-1]) # Using the line index found above, the HJD is extracted by indexing just that from the line.
            
        elif file_path[i][-4:] == 'fits':
            
            hdu = fits.open(file_path[i])
            JD = hdu[0].header['HIERARCH ESO DRS BJD'] # Barycentric Julian Date
        
        # Calculating the mean anomaly M
        
        n = 2*np.pi/P_orb # mean motion in radians 
        
        # Total orbits done since last periastron of 2455959.0039936211
        N = int((JD - T_e)/P_orb)
        
        if print_stat:
            print('----------------------------------------------------------------------------------------------------------------')
            print('Total number of orbits since the given periastron {}: {}'.format(T_e, N))
            print('----------------------------------------------------------------------------------------------------------------')
        
        t = T_e + (N*P_orb) # time of last periastron RIGHT before our HJD!
        
        mean_an = (JD - t)*n # mean anomaly; (t - T)*n 
        
        if print_stat:
            print('Mean Anomaly: {}'.format(mean_an))
            print('----------------------------------------------------------------------------------------------------------------')
        
        # Solving for eccentric anomaly from the mean anomaly as M = E - e*sin(E) = (t - T)*n using pyasl.MarkleyKESolver()
        
        # Instantiate the solver
        ks = pyasl.MarkleyKESolver()
        
        # Solves Kepler's Equation for a set
        # of mean anomaly and eccentricity.
        # Uses the algorithm presented by
        # Markley 1995.
        
        M = mean_an
        E = ks.getE(M, e)
        
        if print_stat:
            print("Eccentric Anomaly: {}".format(E))
            print('----------------------------------------------------------------------------------------------------------------')
        
        f = 2*np.arctan2(1, 1/(np.sqrt((1+e)/(1-e))*np.tan(E/2))) 
        # using np.arctan2 instead of np.arctan to retrive values from the positive quadrant of tan(x) values 
        # see https://stackoverflow.com/questions/16613546/using-arctan-arctan2-to-plot-a-from-0-to-2π
        
        orb_phase = f/(2*np.pi) # converting f to orbital phase by dividing it with 2pi radians!
        
        rot_phase = (JD - phase_start)/P_rot - int((JD - phase_start)/P_rot)
        
        if print_stat:
            print('True Anomaly: {}'.format(f))
            print('----------------------------------------------------------------------------------------------------------------')
            print('Orbital Phase: {}'.format(orb_phase))
            print('----------------------------------------------------------------------------------------------------------------')
            print('Rotational Phase: {}'.format(rot_phase))
            print('----------------------------------------------------------------------------------------------------------------')
        
        
        
        res = JD, N, M, E, f, orb_phase, rot_phase
        
        results.append(res)
        
    # Saving the results in a csv file format  
    if save_results:
        
        if print_stat:
            
            print('Saving results in the working directory in file: {}.csv'.format(save_name))
            print('----------------------------------------------------------------------------------------------------------------')
            
        header = ['JD', 'Number_of_orbits_since_T_e', 'Mean_Anomaly', 'Eccentric_Anomaly', 'True_Anomaly', 'Orbital_Phase', 'Rotational_Phase']
            
        with open('{}.csv'.format(save_name), 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow(header)
            for row in results:
                writer.writerow(row)  
            
    return results
