#!/usr/bin/env python
# coding: utf-8

"""
spec_analysis.py: This python module contains supplementary functions used by the index_calc.py module along with other function for further analysis.

"""

__author__ = "Mukul Kumar"
__email__ = "Mukul.k@uaeu.ac.ae, MXK606@alumni.bham.ac.uk"
__date__ = "26-05-2022"
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

def find_string_idx(out_file_path,
                    string,
                    print_stat=True):
    
    """
    
    Finds the index of the line containing the given string in the given file.
    
    Parameters:
    -----------
    out_file_path: str
    Path of the .out file
    
    string: str
    String to check for in the .out file
    
    print_stat: bool, default: True
    Prints the string name not found in the given file
    
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
        if print_stat:
            print('String', string , 'not found in .out file')
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        return float('nan')
    else:
        return index-1
    
    file.close()
    

## Defining a function to extract important object parameters from the .out file

def obj_params(file_path,
               Instrument,
               print_stat=True):
    
    """
    
    Extracts useful object parameters from the .out/.fits file.
    
    Parameters:
    -----------
    file_path: str
    Path of the .out/.fits file
    
    Instrument: str
    Instrument type used. Available options: ['NARVAL', 'HARPS', 'HARPS-N']
    
    print_stat: bool, default=True
    Prints the status of each process within the function.
    
    Returns:
    --------
    object parameters: dict
    Dictionary containing the usefule object parameters
    
    """
    
    # Creating an empty dictionary to append to 
    object_parameters = {}
    
    ## NARVAL
    
    if Instrument=='NARVAL':
        
        str_list = ['Number of stellar exposures in the sequence (2 or 4) :',
                    'Detector gain (e/ADU) and read-out noise (e) :',
                    '   Coordinates of object :',
                    '   Time of observations :',
                    '   Total exposure time :',
                    '   Heliocentric Julian date (UTC) :',
                    '         >>> vmag/teff estimate from sn curve (mag/K):']
    
        file = open(file_path).readlines() 
        
        idx = []
    
        for string in str_list:
            idx.append(find_string_idx(file_path, string, print_stat=print_stat))
            
        try:
            object_parameters['HJD'] = float(file[idx[5]][-14:-1])
        except TypeError:
            object_parameters['HJD'] = float('nan')
            if print_stat:
                print('Object parameter for "{}" not found in the .out file'.format(str_list[5]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        try:
            object_parameters['RA'] = file[idx[2]][-26:-15].replace(' ', '')
        except TypeError:
            object_parameters['RA'] = float('nan')
            if print_stat:
                print('Object parameter for "{}" not found in the .out file'.format(str_list[2]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        try:
            object_parameters['DEC'] = file[idx[2]][-11:-1]
        except TypeError:
            object_parameters['DEC'] = float('nan')
            if print_stat:
                print('Object parameter for "{}" not found in the .out file'.format(str_list[2]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        try:
            object_parameters['AIRMASS'] = float(file[idx[3]][-8:-3])
        except TypeError:
            object_parameters['AIRMASS'] = float('nan')
            if print_stat:
                print('Object parameter for "{}" not found in the .out file'.format(str_list[3]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
           
        try:
            object_parameters['T_EXP'] = float(file[idx[4]][25:31])
        except TypeError:
            object_parameters['T_EXP'] = float('nan')
            if print_stat:
                print('Object parameter for "{}" not found in the .out file'.format(str_list[4]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['NUM_EXP'] = float(file[idx[0]][-2:-1])
        except TypeError:
            object_parameters['NUM_EXP'] = float('nan')
            if print_stat:
                print('Object parameter for "{}" not found in the .out file'.format(str_list[0])) 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        try:
            object_parameters['GAIN'] = float(file[idx[1]][-12:-1].split()[0])
        except TypeError:
            object_parameters['GAIN'] = float('nan')
            if print_stat:
                print('Object parameter for "{}" not found in the .out file'.format(str_list[1])) 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        try:
            object_parameters['RON'] = float(file[idx[1]][-12:-1].split()[1])
        except TypeError:
            object_parameters['RON'] = float('nan')
            if print_stat:
                print('Object parameter for "{}" not found in the .out file'.format(str_list[1]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')    
             
        try:
            object_parameters['V_mag'] = float(file[idx[6]][-11:-1].split()[0])
        except TypeError:
            object_parameters['V_mag'] = float('nan')
            if print_stat:
                print('Object parameter for "{}" not found in the .out file'.format(str_list[6]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        try:
            object_parameters['T_eff'] = float(file[idx[6]][-11:-1].split()[1])
        except TypeError:
            object_parameters['T_eff'] = float('nan')
            if print_stat:
                print('Object parameter for "{}" not found in the .out file'.format(str_list[6]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
    elif Instrument=='HARPS':
        
        file = fits.open(file_path)
        
        try:
            object_parameters['BJD'] = file[0].header['HIERARCH ESO DRS BJD'] # Barycentric Julian Date
        except KeyError:
            object_parameters['BJD'] = float('nan')
            if print_stat:
                print('Object parameter for "HIERARCH ESO DRS BJD" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        try:
            object_parameters['RA'] = file[0].header['RA'] # Barycentric Julian Date
        except KeyError:
            object_parameters['RA'] = float('nan')
            if print_stat:
                print('Object parameter for "RA" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['DEC'] = file[0].header['DEC'] # Barycentric Julian Date
        except KeyError:
            object_parameters['DEC'] = float('nan')
            if print_stat:
                print('Object parameter for "DEC" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['AIRMASS'] = file[0].header['AIRMASS'] # Barycentric Julian Date
        except KeyError:
            object_parameters['AIRMASS'] = float('nan')
            if print_stat:
                print('Object parameter for "AIRMASS" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        try:
            object_parameters['EXPTIME'] = file[0].header['EXPTIME'] # Exposure time in s
        except KeyError:
            object_parameters['EXPTIME'] = float('nan')
            if print_stat:
                print('Object parameter for "EXPTIME" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['BERV'] = file[0].header['HIERARCH ESO DRS BERV'] # Barycentric Earth Radial Velocity  km/s
        except KeyError:
            object_parameters['BERV'] = float('nan')
            if print_stat:
                print('Object parameter for "HIERARCH ESO DRS BERV" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        
        try:
            object_parameters['OBS_DATE'] = file[0].header['DATE-OBS'] # Observation Date
        except KeyError:
            object_parameters['OBS_DATE'] = float('nan')
            if print_stat:
                print('Object parameter for "DATE-OBS" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['PROG_ID'] = file[0].header['PROG_ID'] # Program ID
        except KeyError:
            object_parameters['PROG_ID'] = float('nan')
            if print_stat:
                print('Object parameter for "PROG_ID" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['SNR'] = file[0].header['SNR'] # Signal to Noise ratio
        except KeyError:
            object_parameters['SNR'] = float('nan')
            if print_stat:
                print('Object parameter for "SNR" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        try:
            object_parameters['SIGDET'] = np.round(file[0].header['HIERARCH ESO DRS CCD SIGDET'], 3)  #CCD Readout Noise [e-]
        except KeyError:
            object_parameters['SIGDET'] = float('nan')
            if print_stat:
                print('Object parameter for "HIERARCH ESO DRS CCD SIGDET" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['CONAD'] = file[0].header['HIERARCH ESO DRS CCD CONAD'] #CCD conversion factor [e-/ADU]; from e- to ADU
        except KeyError:
            object_parameters['CONAD'] = float('nan')
            if print_stat:
                print('Object parameter for "HIERARCH ESO DRS CCD CONAD" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['RON'] = np.round((object_parameters['SIGDET'] * object_parameters['CONAD']), 3) #CCD Readout Noise [ADU]
        except KeyError:
            object_parameters['RON'] = float('nan')
            pass
        
    elif Instrument=='HARPS-N':
        
        file = fits.open(file_path)
        
        try:
            object_parameters['BJD'] = file[0].header['HIERARCH TNG DRS BJD'] # Barycentric Julian Date
        except KeyError:
            object_parameters['BJD'] = float('nan')
            if print_stat:
                print('Object parameter for "HIERARCH TNG DRS BJD" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        try:
            object_parameters['RA'] = file[0].header['RA'] # Barycentric Julian Date
        except KeyError:
            object_parameters['RA'] = float('nan')
            if print_stat:
                print('Object parameter for "RA" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['DEC'] = file[0].header['DEC'] # Barycentric Julian Date
        except KeyError:
            object_parameters['DEC'] = float('nan')
            if print_stat:
                print('Object parameter for "DEC" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['AIRMASS'] = file[0].header['AIRMASS'] # Barycentric Julian Date
        except KeyError:
            object_parameters['AIRMASS'] = float('nan')
            if print_stat:
                print('Object parameter for "AIRMASS" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['EXPTIME'] = file[0].header['EXPTIME'] # Exposure time in seconds
        except KeyError:
            object_parameters['EXPTIME'] = float('nan')
            if print_stat:
                print('Object parameter for "EXPTIME" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['OBS_DATE'] = file[0].header['DATE-OBS'] # Observation Date
        except KeyError:
            object_parameters['OBS_DATE'] = float('nan')
            if print_stat:
                print('Object parameter for "DATE-OBS" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        try:
            object_parameters['PROG_ID'] = file[0].header['PROGRAM'] # Program ID
        except KeyError:
            object_parameters['PROG_ID'] = float('nan')
            if print_stat:
                print('Object parameter for "PROGRAM" not found in the fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
    
    else:
        raise ValueError('Instrument type not recognised. Available options are "NARVAL", "HARPS" and "HARPS-N"')
        
    return object_parameters
    

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
    spectral orders: list
    Returns a list of lists with each containing the extracted order as; [wav, flx and flx_err].
    
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
        # 
        # OR, in some cases;
        #
        # wav = [1,2,3,4,7,8,9,10]
        # order_1 = [1,2,3,4]
        # order_2 = [7,8,9,10]
        
        wav_res = np.mean(np.diff(wav)) # Mean spectral axis resolution
        
        for i in range(len(wav)-1):
            if wav[i+1] < wav[i]:
                jump_index.append(i)
            elif wav[i+1] - wav[i] > 100*wav_res: # Using 100*wav_res since the spectral resolution isn't constant throughout
                jump_index.append(i)
                
                
        # Creating a condition for cases where wavelength axis does not contain individual spectral orders        
        if len(jump_index) > 1:
        
            spec_orders = []
            
            # For when certain files have nan as the flux errors;
            if np.isnan(np.sum(flx_err)):
                
                print('Flux errors array contains NaN values. Returning orders without errors for all!')
                
                # Creating the first spectral order which ends at jump_index[0]
                order_0 = [wav[:jump_index[0]], flx[:jump_index[0]]]
                spec_orders.append(order_0)
                
                # The loop below creates each spectral order form the jump_index list above and appends them to the spec_orders
                # list
                for i in range(len(jump_index)-1):
                    
                    order = [wav[jump_index[i]+1:jump_index[i+1]], flx[jump_index[i]+1:jump_index[i+1]]]
                    spec_orders.append(order)
                    
                # Creating the last spectral order
                order_last = [wav[jump_index[-1]:], flx[jump_index[-1]:], flx_err[jump_index[-1]:]]
                spec_orders.append(order_last)
                    
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
                    
                # Creating the last spectral order
                order_last = [wav[jump_index[-1]:], flx[jump_index[-1]:], flx_err[jump_index[-1]:]]
                spec_orders.append(order_last)
                
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
                plt.tight_layout()
                
            return spec_orders
                
        else:
            print('No individual spectral orders found. The wavelength axis is linear.')
            
    else:
        raise ValueError("Input arrays must have the same shape.")
        
        
## Defining a function to reach check which orders the given lines are found in 
    
def check_lines(spectral_orders,
                line_names,
                lines,
                bandwidths):
    
        
    """
    
    Looks for which spectral order contains the given line ± the bandwidth/2, i.e. the region required for flux calculation.
        
    Parameters
    ----------
    spectral_orders: list
    List containing all of the extracted spectral orders

    line_names: list
    List containing the names of each line
    
    lines: list
    List containing the lines to check
    
    bandwidths: list
    List conatining the bandwidths of the given lines
    
    Returns:
    --------
    Line index: list
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
        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
    return idx

## Defining a function to read the given data

def read_data(file_path,
              Instrument,
              out_file_path=None,
              ccf_file_path=None,
              print_stat=True,
              show_plots=True):
    
    """
    
    Reads the data contained within the .s/.fits file and extract useful information from the .out/.fits file.
    
    Parameters:
    ----------
    file_path: str
    File path of the .s/.fits file
    
    Instrument: str
    Instrument type used. Available options: ['NARVAL', 'HARPS', 'HARPS-N']
    
    out_file_path: str, default=None
    File path of the .out file to extract object parameters from
    
    ccf_file_path: str, default=None
    File path of the CCF file to extract the object radial velocity from
    
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
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        # Checking if its a Stokes V or I .s file using pandas
    
        df = pd.read_fwf(file_path, skiprows=2) # skipping first 2 rows of .s file
        
        # Defining the column names for both Stove V and I files
        col_names_V = ['Wavelength', 'Intensity', 'Polarized', 'N1', 'N2', 'I_err'] 
        col_names_I = ['Wavelength', 'Intensity', 'I_err']
        
        if len(df.columns)==6:
            data_spec = pd.read_csv(file_path, names=col_names_V, skiprows=2, sep=' ', skipinitialspace=True) 
            if print_stat:
                print('Stokes Profile: [V]')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        elif len(df.columns)==3:
            data_spec = pd.read_csv(file_path, names=col_names_V, skiprows=2, sep=' ', skipinitialspace=True)
            if print_stat:
                print('Stokes Profile: [I]')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        else:
            raise ValueError('Input .s file contains unrecognisable number of columns. Recognised numbers are 3 (I profile) and 6 (V profile).')
            
        if print_stat:    
            print('Extracting all overlapping spectral orders')
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        spec_orders = extract_orders(data_spec['Wavelength'],
                                     data_spec['Intensity'],
                                     data_spec['I_err'],
                                     show_plot=show_plots)
        
        if out_file_path != None:
            
            if print_stat:
                print('Extracting useful object parameters from the .out file: {}'.format(out_file_path))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            object_parameters = obj_params(out_file_path, Instrument=Instrument, print_stat=print_stat)
            
            if print_stat:
                keys = [i for i in object_parameters.keys()]
                vals = [j for j in object_parameters.values()]
                
                for i in range(len(keys)):
                    print('{}: {}'.format(keys[i], vals[i]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
            return object_parameters, spec_orders
        
        else:
            return spec_orders
    
    elif Instrument=='HARPS':
        
        if print_stat:
            print('Reading the data from the .fits file: {}'.format(file_path))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        # Opening the FITS file using 'astropy.io.fits'
        # NOTE: The format of this FITS file must be ADP which contains the reduced spectrum with the wav, 
        # flux and flux_err in three columns
        
        file = fits.open(file_path)
        
        #Extracting useful information from the fits file header
        
        if print_stat:
            print('Extracting useful object parameters from the .fits file header')
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------') 
        
        object_parameters = obj_params(file_path, Instrument=Instrument, print_stat=print_stat)
        
        if ccf_file_path:
            if print_stat:
                print('Extracting RV from the CCF fits file: {}'.format(ccf_file_path))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            ccf_file = fits.open(ccf_file_path) # Opening the CCF FITS file to extract the RV
            
            try:
                object_parameters['RV'] = ccf_file[0].header['HIERARCH ESO DRS CCF RV']*1000 # Radial velocity converted from km/s to m/s
            except KeyError:
                object_parameters['RV'] = float('nan')
                print('Object parameter for "HIERARCH ESO DRS CCF RV" not found in the CCF fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
    
        if print_stat:
            keys = [i for i in object_parameters.keys()]
            vals = [j for j in object_parameters.values()]
            
            for i in range(len(keys)):
                print('{}: {}'.format(keys[i], vals[i]))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        # Defining each wavelength, flux and flux error arrays from the FITS file!
        
        wvl = file[1].data[0][0]/10 # nm 
        flx = file[1].data[0][1] # Flux in ADU
        flx_err = file[1].data[0][2]
        
        spectrum = [wvl, flx, flx_err]
        
        # Plotting the spectrum
        
        if show_plots:
            
            f, ax  = plt.subplots(figsize=(10,4)) 
            ax.plot(spectrum[0], spectrum[1], '-k')  
            ax.set_xlabel('$\lambda$ (nm)')
            ax.set_ylabel("Flux (adu)")
            plt.minorticks_on()
            ax.tick_params(direction='in', which='both')
            plt.tight_layout()
            plt.show()
            
        return object_parameters, spectrum
    
    elif Instrument=='HARPS-N':
        
        if print_stat:
            print('Reading the data from the .fits file: {}'.format(file_path))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        # Opening the FITS file using 'astropy.io.fits'
        # NOTE: The format of this FITS file must be s1d which only contains flux array. 
        # The wavelength array is constructed using the starting point (CRVAL1), length of spectral axis (NAXIS1) 
        # and wavelength step (CDELT1) from the FITS file header.
        
        file = fits.open(file_path)
        
        #Extracting useful information from the fits file header
        
        if print_stat:
            print('Extracting useful object parameters from the .fits file header')
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        object_parameters = obj_params(file_path, Instrument=Instrument, print_stat=print_stat)
        
        if ccf_file_path:
            if print_stat:
                print('Extracting RV from the CCF fits file: {}'.format(ccf_file_path))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            ccf_file = fits.open(ccf_file_path)  # Opening the CCF FITS file to extract the RV
            
            try:
                object_parameters['RV'] = ccf_file[0].header['HIERARCH TNG DRS CCF RV']*1000 # Radial velocity converted from km/s to m/s
            except KeyError:
                object_parameters['RV'] = float('nan')
                print('Object parameter for "HIERARCH TNG DRS CCF RV" not found in the CCF fits file header') 
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        if print_stat:
            keys = [i for i in object_parameters.keys()]
            vals = [j for j in object_parameters.values()]
            
            for i in range(len(keys)):
                print('{}: {}'.format(keys[i], vals[i]))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        # Defining each wavelength and flux arrays from the FITS file!
        # NOTE: No error column provided in the .fits file and no ReadOut noise as well to construct our own 
        # flux_err array!
        
        # constructing the spectral axis using start point, delta and axis length from file header
        wvl = file[0].header['CRVAL1'] + file[0].header['CDELT1']*np.arange(0, file[0].header['NAXIS1'])
        wvl = wvl/10 # nm
        flx = file[0].data # Flux in ADU
        
        spectrum = [wvl, flx]
        
        # Plotting the spectrum
        
        if show_plots:
            
            f, ax  = plt.subplots(figsize=(10,4)) 
            ax.plot(spectrum[0], spectrum[1], '-k')  
            ax.set_xlabel('$\lambda$ (nm)')
            ax.set_ylabel("Flux (adu)")
            plt.minorticks_on()
            ax.tick_params(direction='in', which='both')
            plt.tight_layout()
            plt.show()
            
        return object_parameters, spectrum
    
## Defining a function to do the calculation part for each index function given the regions!

def calc_ind(regions,
             index_name,
             print_stat,
             CaI_index=False,
             hfv=None):
    
    """
    
    Calculates the indices for the index_calc functions given the regions.
    
    Parameters:
    -----------
    regions: list
    List containing the appropriate regions required for the calculation
    
    index_name: str
    Name of the index to calculate. Available options are; ['HaI', 'NaI', 'CaIIH']
    
    print_stat: bool
    print_stat argument from within the index_calc function
    
    CaI_index: bool, default=False
    CaI index calculation condition. Used when index_name='HaI'
    
    hfv: int, default=None
    hfv parameter from NaI_index function from index_calc. Used when index_name='NaI'
    
    
    Returns:
    --------
    For 'HaI'; I_HaI, I_HaI_err, I_CaI, I_CaI_err (NOTE: CaI index values will be returned as NaN if CaI_index=False)
    For 'NaI'; I_NaI, I_NaI_err, F1_mean, F2_mean
    For 'CaIIH'; I_CaIIH, I_CaIIH_err 
    
    All returned values are type float()
    
    """
    
    ## HaI
    
    if index_name=='HaI':
    
        F_H_alpha_region = regions[0]
        F1_region = regions[1]
        F2_region = regions[2]
            
        # Mean of the flux within this region is calculated using np.mean and rounded off to 4 decimal places
        F_H_alpha_mean = np.round(np.mean(F_H_alpha_region.flux.value), 4)
        
        # The error on the mean flux is calculated as the standard error of the mean
        F_H_alpha_sum_err = [i**2 for i in F_H_alpha_region.uncertainty.array]
        F_H_alpha_mean_err = np.round((np.sqrt(np.sum(F_H_alpha_sum_err))/len(F_H_alpha_sum_err)), 4)
        
        # Same thing repeated for the F1 and F2 regions
        F1_mean = np.round(np.mean(F1_region.flux.value), 4)
        F1_sum_err = [i**2 for i in F1_region.uncertainty.array]
        F1_mean_err = np.round((np.sqrt(np.sum(F1_sum_err))/len(F1_sum_err)), 4)
        
        F2_mean = np.round(np.mean(F2_region.flux.value), 4)
        F2_sum_err = [i**2 for i in F2_region.uncertainty.array]
        F2_mean_err = np.round((np.sqrt(np.sum(F2_sum_err))/len(F2_sum_err)), 4)
                   
        if print_stat:
            print('H alpha region used ranges from {}nm to {}nm'.format(np.round(F_H_alpha_region.spectral_axis[0].value, 3),
                                                                        np.round(F_H_alpha_region.spectral_axis[-1].value, 3)))
            print('F1 region used ranges from {}nm to {}nm'.format(np.round(F1_region.spectral_axis[0].value, 3),
                                                                   np.round(F1_region.spectral_axis[-1].value, 3)))
            print('F2 region used ranges from {}nm to {}nm'.format(np.round(F2_region.spectral_axis[0].value, 3),
                                                                   np.round(F2_region.spectral_axis[-1].value, 3)))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        # H alpha index is computed using the calculated mean fluxes.
        
        Hai_from_mean = np.round((F_H_alpha_mean/(F1_mean + F2_mean)), 4)
        
        # Continuum flux error is calculated as explained at the start of the tutorial Jupyter Notebook!
        
        sigma_F12_from_mean = np.sqrt((np.square(F1_mean_err) + np.square(F2_mean_err)))
        
        # Error on this index is calculated as explained at the start of the tutorial Jupyter notebook!
        
        sigma_Hai_from_mean = np.round((Hai_from_mean*np.sqrt(np.square(F_H_alpha_mean_err/F_H_alpha_mean) + np.square(sigma_F12_from_mean/(F1_mean+F2_mean)))), 4)
        
        if print_stat:
    
            print('Mean of {} flux points in H alpha: {}±{}'.format(len(F_H_alpha_region.flux), F_H_alpha_mean, F_H_alpha_mean_err))
            print('Mean of {} flux points in F1: {}±{}'.format(len(F1_region.flux), F1_mean, F1_mean_err))
            print('Mean of {} flux points in F2: {}±{}'.format(len(F2_region.flux), F2_mean, F2_mean_err))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('Index from mean of flux points in each band: {}±{}'.format(Hai_from_mean, sigma_Hai_from_mean))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        ## CaI_index 
        
        if CaI_index:
            
            if print_stat:
                print('Calculating CaI Index')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            F_CaI_region = regions[3]
            F_CaI_mean = np.round(np.mean(F_CaI_region.flux.value), 4) # Calculating mean of the flux within this region
            
            # The error on the mean flux is calculated as the standard error of the mean
            F_CaI_sum_err = [i**2 for i in F_CaI_region.uncertainty.array]
            F_CaI_mean_err = np.round((np.sqrt(np.sum(F_CaI_sum_err))/len(F_CaI_sum_err)), 4)
            
            # Calculating the CaI index using the mean fluxes calculated above
            CaI_from_mean = np.round((F_CaI_mean/(F1_mean + F2_mean)), 4)
            
            # Index error calculated in the same way as that for H alpha index above
            sigma_CaI_from_mean = np.round((CaI_from_mean*np.sqrt(np.square(F_CaI_mean_err/F_CaI_mean) + np.square(sigma_F12_from_mean/(F1_mean+F2_mean)))), 4)
            
            if print_stat:
                print('CaI region used ranges from {}nm to {}nm'.format(F_CaI_region.spectral_axis[0].value, 
                                                                         F_CaI_region.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('Mean of {} flux points in CaI: {}±{}'.format(len(F_CaI_region.flux), F_CaI_mean, F_CaI_mean_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('Index from mean of flux points in each band: {}±{}'.format(CaI_from_mean, sigma_CaI_from_mean))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
        else:
            
            CaI_from_mean = float('nan')
            sigma_CaI_from_mean = float('nan')
        
        return Hai_from_mean, sigma_Hai_from_mean, CaI_from_mean, sigma_CaI_from_mean
    
    ## NaI
    
    elif index_name=='NaI':
            
        NaID1_region = regions[0]
        NaID2_region = regions[1]
        F1_region = regions[2]
        F2_region = regions[3]
        
        # Definig the pseudo-continuum
        
        # Sorting the flux in F1 region from lowest to highest and using only the given number of highest flux values, (hfv), for the mean.
        
        F1_sorted_flux = F1_region.flux[np.argsort(-F1_region.flux)[:hfv]] 
        F1_mean = np.round(np.mean(F1_sorted_flux), 4)
        F1_err = F1_region.uncertainty.array[np.argsort(-F1_region.flux)[:hfv]]
        
        # The error on this mean is calculated using error propagation
        
        F1_sum_err = [i**2 for i in F1_err]
        F1_err = np.round((np.sqrt(np.sum(F1_sum_err))/len(F1_sum_err)), 4)
        
        # Same process for F2 region
        
        F2_sorted_flux = F2_region.flux[np.argsort(-F2_region.flux)[:hfv]]
        F2_mean = np.round(np.mean(F2_sorted_flux), 4)
        F2_err = F2_region.uncertainty.array[np.argsort(-F2_region.flux)[:hfv]]
        
        F2_sum_err = [i**2 for i in F2_err]
        F2_err = np.round((np.sqrt(np.sum(F2_sum_err))/len(F2_sum_err)), 4)
        
        # The pseudo-continuum is taken as the mean of the fluxes calculated abvove in F1 and F2 regions
        
        F_cont = np.round(((F1_mean+F2_mean)/2), 4) # This value is used for the index calculation
        F_cont_err = np.round((np.sqrt(F1_err**2 + F2_err**2)/2), 4) # Error calculated using error propagation
        
        # Calculating the mean flux in the D1 D2 lines
            
        NaID1_mean = np.round(np.mean(NaID1_region.flux.value), 4)
        
        # Error calculated using error propagation
        NaID1_sum_err = [i**2 for i in NaID1_region.uncertainty.array]
        NaID1_err = np.round((np.sqrt(np.sum(NaID1_sum_err))/len(NaID1_sum_err)), 4)
        
        NaID2_mean = np.round(np.mean(NaID2_region.flux.value), 4)
        NaID2_sum_err = [i**2 for i in NaID2_region.uncertainty.array]
        NaID2_err = np.round((np.sqrt(np.sum(NaID2_sum_err))/len(NaID2_sum_err)), 4)
        
        # Error on the sum of mean fluxes in D1 and D2
        sigma_D12 = np.sqrt(np.square(NaID1_err) + np.square(NaID2_err))
        
        # Calculating the index and rounding it up to 5 decimal places
        NaID_index = np.round(((NaID1_mean + NaID2_mean)/F_cont.value), 4)
        
        # Error calculated using error propagation and rounding it up to 5 decimal places
        sigma_NaID_index = np.round((NaID_index*np.sqrt(np.square(sigma_D12/(NaID1_mean + NaID2_mean)) + np.square(F_cont_err/F_cont.value))), 4)
        
        if print_stat:
            print('Using {} highest flux values in each continuum band for the pseudo-cont. calculation'.format(hfv))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('Mean of {} out of a total {} flux points in blue cont.: {}±{}'.format(len(F1_sorted_flux), len(F1_region.flux), F1_mean, F1_err))
            print('Mean of {} out of a total {} flux points in red cont.:  {}±{}'.format(len(F2_sorted_flux), len(F2_region.flux), F2_mean, F2_err))
            print('Mean cont. flux: {}±{}'.format(F_cont.value, F_cont_err))
            print('Mean of {} flux points in D1: {}±{}'.format(len(NaID1_region.flux), NaID1_mean, NaID1_err))
            print('Mean of {} flux points in D2: {}±{}'.format(len(NaID2_region.flux), NaID2_mean, NaID2_err))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('The NaI doublet index is: {}±{}'.format(NaID_index, sigma_NaID_index))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        return NaID_index, sigma_NaID_index, F1_mean, F2_mean
    
    ## CaIIH
    
    elif index_name=='CaIIH':
        
        F_CaIIH_region = regions[0]
        cont_R_region = regions[1]
        
        F_CaIIH_mean = np.round(np.mean(F_CaIIH_region.flux.value), 4) # Calculating mean of the flux within this bandwidth
        
        # Calculating the standard error on the mean flux calculated above.
        F_CaIIH_sum_err = [i**2 for i in F_CaIIH_region.uncertainty.array]
        F_CaIIH_mean_err = np.round((np.sqrt(np.sum(F_CaIIH_sum_err))/len(F_CaIIH_sum_err)), 4)
        
        # Doing the same for the cont R region!
        
        cont_R_mean = np.round(np.mean(cont_R_region.flux.value), 4)
        cont_R_sum_err = [i**2 for i in cont_R_region.uncertainty.array]
        cont_R_mean_err = np.round((np.sqrt(np.sum(cont_R_sum_err))/len(cont_R_sum_err)), 4)

        # Calculating the index from the mean fluxes calculated above
        CaIIH_from_mean = np.round((F_CaIIH_mean/cont_R_mean), 4)
        
        # Error on this index is calculated using error propagation!
        sigma_CaIIH_from_mean = np.round((CaIIH_from_mean*np.sqrt(np.square(F_CaIIH_mean_err/F_CaIIH_mean) + np.square(cont_R_mean_err/cont_R_mean))), 4)
        
        if print_stat:
            print('CaIIH region used ranges from {}nm to {}nm:'.format(F_CaIIH_region.spectral_axis[0].value, 
                                                                 F_CaIIH_region.spectral_axis[-1].value))
            print('Cont R region used ranges from {}nm to {}nm:'.format(cont_R_region.spectral_axis[0].value, 
                                                                 cont_R_region.spectral_axis[-1].value))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('Mean of {} flux points in CaIIH: {}±{}'.format(len(F_CaIIH_region.flux), F_CaIIH_mean, F_CaIIH_mean_err))
            print('Mean of {} flux points in cont R: {}±{}'.format(len(cont_R_region.flux), cont_R_mean, cont_R_mean_err))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('Index from mean of flux points in each band: {}±{}'.format(CaIIH_from_mean, sigma_CaIIH_from_mean))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        return CaIIH_from_mean, sigma_CaIIH_from_mean
    
## Defining a function to calculate the LombScargle periodogram!
        
def LS_periodogram(x,
                   y,
                   minimum_frequency,
                   maximum_frequency,
                   samples_per_peak=10,
                   nterms=1, 
                   method='chi2',
                   normalization='model',
                   fap_method='bootstrap',
                   dy=None,
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
    
    dy: array, default=None
    Error on observation values
    
    probabilities: list, default=None
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
    
    if dy != None:
        if np.isnan(np.sum(dy)):
            raise ValueError('Error array "dy" contains NaN values')
        
        ls = LombScargle(x, y, dy, nterms=nterms, normalization=normalization)
        
    else:
        ls = LombScargle(x, y, nterms=nterms, normalization=normalization)
    
    freq, power = ls.autopower(minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak, method=method)
    
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Total number of frequencies tested: {}'.format(len(freq)))
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Frequency grid resolution: {}'.format(np.round(np.diff(freq)[0], 6)))
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
    
    sort_idx = np.argsort(-power) # minus symbol to sort the array from biggest to smallest. np.argsort resturns the indices that would sort the given array. Very useful!
    
    if nterms==1: # FAL and FAP's are not provided for multiple Fourier term periodograms!
        
        print('Calculating False Alarm Probabilities/Levels (FAPs/FALs) using the {} method'.format(fap_method))
        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        if probabilities != None:
            probabilities = probabilities 
        else:
            probabilities = [0.5, 0.2, 0.1]
        
        fal = ls.false_alarm_level(probabilities, method=fap_method, minimum_frequency=minimum_frequency, 
                                   maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak) 
        
        fap = ls.false_alarm_probability(power=power, method=fap_method, minimum_frequency=minimum_frequency, 
                                         maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak)
        
    print('Frequency at max. power is: {} which is {}d'.format(np.round(freq[sort_idx[0]], 4), np.round(1/freq[sort_idx[0]], 4)))
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
    if show_plot:
        
        plt.figure(figsize=(10,4))
        plt.plot(1/freq, power, color='black')
        plt.vlines(1/freq[sort_idx[0]], ymin=-0.12, ymax=max(power), colors='red', linestyles='-', linewidth=3, label='P={}d'.format(np.round((1/freq[sort_idx[0]]), 3)))
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
        
        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('Sampling window period is: {}d'.format(np.round(1/freq_samp[np.argmax(power_samp)], 4)))
        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        if show_plot:
            
            plt.figure(figsize=(10,4))
            plt.plot(1/freq_samp, power_samp, color='black')
            plt.vlines(1/freq_samp[np.argmax(power_samp)], ymin=-0.12, ymax=max(power_samp), colors='red', linestyles='-', linewidth=3, label='P={}d'.format(np.round(1/freq_samp[np.argmax(power_samp)], 4)))
            plt.xlabel('Period (days)')
            plt.ylabel('Power') 
            plt.title('Sampling Window Function Periodogram')
            plt.tick_params(axis='both', direction='in', which='both')
            plt.xscale('log')
            plt.xlim(1/maximum_frequency-0.1,1/minimum_frequency+0.5)
            plt.legend()
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
               period,
               fit,
               normalization='model',
               dy=None,
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
    
    period: int
    Orbital period of the model to fit
    
    fit: str
    Fit type; available options are 
    'JD' which fits the model over the enitre observation timespan
    'phase' which phase folds the data before fitting the model form 0 - 1
    
    normalization: str, default='model'
    Periodogram normalization method. 
    See https://docs.astropy.org/en/stable/timeseries/lombscargle.html for more info.
    
    dy: list, default=None
    Error on observation values
    
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
    If fit = 'JD';
    model fit timestamps, model fit y values (model fit y values of the multi-term models if multi_term is True)
    
    If fit = 'phase':
    phase folded x values, model fit timestamps, model fit y values, model fit parameters (model fit y values and parameters of the multi-term models if multi_term is True) [See 'The Lomb-Scargle Model' section here, https://docs.astropy.org/en/stable/timeseries/lombscargle.html, for more info on model parameters]
    
    All of these are type numpy.ndarray. 
    
    """
    
    if dy !=None:
        
        ls_1 = LombScargle(x, y, dy, nterms=1, normalization=normalization)
        ls_2 = LombScargle(x, y, dy, nterms=2, normalization=normalization)
        ls_3 = LombScargle(x, y, dy, nterms=3, normalization=normalization)
        
    else:
        ls_1 = LombScargle(x, y, nterms=1, normalization=normalization)
        ls_2 = LombScargle(x, y, nterms=2, normalization=normalization)
        ls_3 = LombScargle(x, y, nterms=3, normalization=normalization)
    
    
    if fit == 'JD':
        
        t_fit = np.linspace(x.min(), x.max(), 10000)
        y_fit_1 = ls_1.model(t_fit, 1/period)
        y_fit_2 = ls_2.model(t_fit, 1/period)
        y_fit_3 = ls_3.model(t_fit, 1/period)
        
        plt.figure(figsize=(10,4))
        
        if dy !=None:
            plt.errorbar(x, y, yerr=dy, fmt='.k', capsize=5)
        else:
            plt.plot(x, y, '.k')
            
        plt.plot(t_fit, y_fit_1, '-r', label='Fundamental')
        
        if multi_term:
            plt.plot(t_fit, y_fit_2, '-b', label='Fundamental + 1st Harmonic')
            plt.plot(t_fit, y_fit_3, '-g', label='Fundamental + First 2 Harmonics')
            plt.legend()
        
        plt.xlabel('JD')
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
            theta = ls_1.model_parameters(1/period)
            return t_fit, y_fit_1, theta
            
    elif fit == 'phase':
        
        t_fit = np.linspace(0.0, period, 10000)
        y_fit_1 = ls_1.model(t_fit, 1/period)
        y_fit_2 = ls_2.model(t_fit, 1/period)
        y_fit_3 = ls_3.model(t_fit, 1/period)
        
        phase_folded_x = pyasl.foldAt(x, period)
        
        plt.figure(figsize=(10,4))
        
        if dy != None:
            plt.errorbar(phase_folded_x, y, yerr=dy, fmt='ok', capsize=5)
        else:
            plt.plot(phase_folded_x, y, 'ok')
            
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
            theta = [ls_1.model_parameters(1/period), ls_2.model_parameters(1/period), ls_3.model_parameters(1/period)]
            return phase_folded_x, t_fit, y_fit_1, y_fit_2, y_fit_3, theta
        else:
            theta = ls_1.model_parameters(1/period)
            return phase_folded_x, t_fit, y_fit_1, theta
        
## Defining a function to find the nearest value to the given value in an array
        
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

## Defining a function to calculate the true anomaly, i.e. orbital phase angle from the observation JD

# T_e = 2455959 JD taken from http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2014AcA....64..323M
# converting T_e from JD to HJD using https://doncarona.tamu.edu/apps/jd/

def ephemerides(file_path,
                P_orb=2.644,
                T_e=2455959.0039936211,
                e=0.152,
                P_rot=None,
                phase_start=None, 
                Rot_phase=False,
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
    
    P_rot: int, default=None
    Stellar rotation period in days.
    Used if Rot_phase is True
    
    phase_start: int
    Starting point for the rotational phase. This ideally should be the first JD of your observation.
    Used if Rot_phase is True
    
    Rot_phase: bool, default=False
    Calculates the stellar rotational phases and cycle using the given 'P_rot' and 'phase_start' parameters.
    
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
            
            try:
                JD = hdu[0].header['HIERARCH ESO DRS BJD']
            except:
                JD = hdu[0].header['HIERARCH TNG DRS BJD']
        
        # Calculating the mean anomaly M
        
        n = 2*np.pi/P_orb # mean motion in radians 
        
        # Total orbits done since last periastron
        N = int((JD - T_e)/P_orb)
        
        if print_stat:
            print('Total number of orbits since the given periastron {}: {}'.format(T_e, N))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        t = T_e + (N*P_orb) # time of last periastron RIGHT before our HJD!
        
        mean_an = (JD - t)*n # mean anomaly; (t - T)*n 
        
        # Solving for eccentric anomaly from the mean anomaly as M = E - e*sin(E) = (t - T)*n using pyasl.MarkleyKESolver()
        
        # Instantiate the solver
        ks = pyasl.MarkleyKESolver()
        
        # Solves Kepler's Equation for a set
        # of mean anomaly and eccentricity.
        # Uses the algorithm presented by
        # Markley 1995.
        
        M = np.round(mean_an, 5)
        
        if print_stat:
            print('Mean Anomaly: {}'.format(M))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        E = np.round((ks.getE(M, e)), 5)
        
        if print_stat:
            print("Eccentric Anomaly: {}".format(E))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        f = np.round((2*np.arctan2(1, 1/(np.sqrt((1+e)/(1-e))*np.tan(E/2)))), 5)
        # using np.arctan2 instead of np.arctan to retrive values from the positive quadrant of tan(x) values 
        # see https://stackoverflow.com/questions/16613546/using-arctan-arctan2-to-plot-a-from-0-to-2π
        
        orb_phase = np.round((f/(2*np.pi)), 5) # converting f to orbital phase by dividing it with 2pi radians!
        
        if Rot_phase:
            rot_cycle = np.round(((JD - phase_start)/P_rot), 5)
            rot_phase = np.round((rot_cycle - int(rot_cycle)), 5)
        
        if print_stat:
            print('True Anomaly: {}'.format(f))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('Orbital Phase: {}'.format(orb_phase))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            if Rot_phase:
                print('Rotational Phase: {}'.format(rot_phase))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        
        if Rot_phase:
            res = JD, N, M, E, f, orb_phase, rot_phase, rot_cycle
        else:
            res = JD, N, M, E, f, orb_phase
        
        results.append(res)
        
    # Saving the results in a csv file format  
    if save_results:
        
        if print_stat:
            
            print('Saving results in the working directory in file: {}.csv'.format(save_name))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        if Rot_phase:
            header = ['JD', 'Number_of_orbits_since_T_e', 'Mean_Anomaly', 'Eccentric_Anomaly', 'True_Anomaly', 'Orbital_Phase', 'Rotational_Phase', 'Rotational_Cycle']
        else:
            header = ['JD', 'Number_of_orbits_since_T_e', 'Mean_Anomaly', 'Eccentric_Anomaly', 'True_Anomaly', 'Orbital_Phase']
            
        with open('{}.csv'.format(save_name), 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow(header)
            for row in results:
                writer.writerow(row)  
            
    return results