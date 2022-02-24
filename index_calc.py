#!/usr/bin/env python
# coding: utf-8

"""
index_calc.py: This python module contains the CaIIH, NaI, and Hα (CaI within it) activity index calculation functions.

"""

__author__ = "Mukul Kumar"
__email__ = "Mukul.k@uaeu.ac.ae, MXK606@alumni.bham.ac.uk"
__date__ = "24-02-2022"
__version__ = "1.7"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as log_progress
import astropy.units as u
import astropy as ap
import csv
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from specutils.manipulation import extract_region
from astropy.modeling.polynomial import Chebyshev1D
from astropy.nddata import StdDevUncertainty
from astropy.io import fits
from spec_analysis import find_string_idx, find_nearest, extract_orders
    
## Defining a function for calculating the H alpha index following Boisse et al. 2009 (2009A&A...495..959B)

def H_alpha_index(file_path,
                  radial_velocity=9609,
                  degree=4,
                  H_alpha_line=656.2808,
                  H_alpha_band=0.16,
                  CaI_line=657.2795,
                  CaI_band=0.08,
                  F1_line=655.087,
                  F1_band=1.075,
                  F2_line=658.031, 
                  F2_band=0.875,
                  Instrument='NARVAL',
                  Stokes_profile=['V'],
                  norm_spec=False,
                  plot_fit=False, 
                  plot_spec=True,
                  print_stat=True,
                  save_results=False, 
                  results_file_name=None,
                  save_figs=False,
                  out_file_path=None,
                  ccf_file_path=None,
                  CaI_index=True):
    
    """
    Calculates the H alpha index following Boisse I., et al., 2009, A&A, 495, 959. In addition, it also 
    calculates the CaI index following Robertson P., Endl M., Cochran W. D., Dodson-Robinson S. E., 2013, ApJ, 764, 3. 
    
    This index uses the exact same reference continuums, F1 and F2, used for the H alpha index to serve as a 
    control against the significance of H alpha index variations!
    
    Parameters:
    
    -----------
    file_path: list, .s format (NARVAL), ADP..._.fits format (HARPS) or s1d_A.fits format (HARPS-N)
    List containng the paths of the spectrum files 
    
    radial_velocity: int, default: 9609 m/s
    Stellar radial velocity along the line-of-sight taken from GAIA DR2 for GJ 436.
    This value is used for doppler shifting the spectra to its rest frame.
    
    degree: int, default: 4
    The degree of the Chebyshev1D polynomial to fit to the continuum for normalisation.
    Normalisation done using Specutils. 
    For more info, 
    see https://specutils.readthedocs.io/en/stable/api/specutils.fitting.fit_generic_continuum.html#specutils.fitting.fit_generic_continuum
    
    H_alpha_line: int, default: 656.2808 nm
    H alpha line centre in nm.
    
    H_alpha_band: int, default: 0.16 nm
    Band width (nm) in which to calculate the mean flux.
    
    CaI_line: int, default: 657.2795 nm
    CaI line centre in nm.
    
    CaI_band: int, default: 0.08 nm
    Band width (nm) in which to calculate the mean flux.
    
    F1_line: int, default: 655.087 nm
    Line centre of the blue reference continuum.
    
    F1_band: int, default: 1.075 nm
    Band width (nm) in which to calculate the mean continuum flux.
    
    F2_line: int, default: 658.031 nm
    Line centre of the red reference continuum.
    
    F2_band: int, default: 0.875 nm
    Band width (nm) in which to calculate the mean continuum flux.
    
    Instrument: str, default: 'NARVAL'
    The instrument from which the data has been collected. Available options are 'NARVAL', 'HARPS' or 'HARPS-N'.
    
    Stokes_profile: str, default: ['V']
    The Stokes profile for the input data. 'V' for per night and 'I' for per sub-exposure per night. Used only when Instrument type is 'NARVAL'
    
    norm_spec: bool, default: False
    Normalizes the spectrum.
    
    plot_fit: bool, default: False
    Plots the continuum fitting normalization processes.
    
    plot_spec: bool, default: True
    Plots the final reduced spectrum.
    
    print_stat: bool, default: True
    Prints the status of each process within the function.
    
    save_results: bool, default: False
    Saves the run results in a .csv format in the working directory
    
    results_file_name: str, default: None
    Name of the file with which to save the results file
    
    save_figs: bool, default: False
    Save the plots in a pdf format in the working directory
    
    out_file_path: list, .out format (NARVAL), default: None
    List containing the paths of the .out files to extract the OBS_HJD. If None, HJD is returned as NaN. Used only when Instrument type is 'NARVAL'
    
    ccf_file_path: list, .fits format (HARPS/HARPS-N), default: None
    List containig the paths of the CCF FITS files to extract the radial velocity. If None, the given radial velocity argument is used for all files for doppler shift corrections
    
    CaI_index: bool, default=True
    Calculates the activity insensitive CaI index as well. If False, NaN values are returned instead.
    
    Returns:
    -----------
    NARVAL: HJD of observation, H alpha index, error on H alpha index, CaI index and error on CaI index.
    HARPS: MJD of observation, Observation date, H alpha index, error on H alpha index, CaI index, error on CaI index, radial velocity, exposure time (s), SNR, ReadOut noise and Program ID
    HARPS-N: MJD of observation, Observation date, H alpha index, error on H alpha index, CaI index, error on CaI index, radial velocity, exposure time (s) and Program ID
    
    All values are type float().
    
    """
    
    results = [] # Empty list to which the run results will be appended
    
    # Creating a loop to go through each given file_path in the list of file paths
    
    # Using the tqdm function 'log_progress' to provide a neat progress bar in Jupyter Notebook which shows the total number of
    # runs, the run time per iteration and the total run time for all files!
    
    for i in log_progress(range(len(file_path)), desc='Calculating Hα Index'):
        
        # Creating a loop for data from each Instrument;
        
        # NARVAL
        
        if Instrument == 'NARVAL':
            
            if out_file_path != None:
                
                file = open(out_file_path[i]).readlines() # Opening the .out file and reading each line as a string
                
                string = '   Heliocentric Julian date (UTC) :' # Creating a string variable that matches the string in the .out file
                
                idx = find_string_idx(out_file_path[i], string) # Using the 'find_string_idx' function to find the index of the line that contains the above string. 
                
                HJD = float(file[idx][-14:-1]) # Using the line index found above, the HJD is extracted by indexing just that from the line.
                
            else:
                if print_stat:
                    print('out_file_path not given as an argument. Returning NaN as HJD instead.')
                    print('----------------------------------------------------------------------------------------------------------------')
                HJD = float('nan')
        
            # Defining column names for pandas to read the file easily
            
            col_names_V = ['Wavelength', 'Intensity', 'Polarized', 'N1', 'N2', 'I_err'] # For Stokes V
            col_names_I = ['Wavelength', 'Intensity', 'I_err'] # For Stokes I
            
            # Reading data using pandas and skipping the first 2 rows

            if Stokes_profile==['V']:
                data_spec = pd.read_fwf(file_path[i], names=col_names_V, skiprows=2) 
            else:
                data_spec = pd.read_fwf(file_path[i], names=col_names_I, skiprows=2)
                
            # Extracting indivdidual spectral orders using 'extract_orders'
            # Orders #35 and #34 both contain the H alpha line for GJ 436 data. 
            # The #34th order is used since it has a higher SNR; (see .out file)
            
            if print_stat:
                print('Extracting spectral orders')
                print('----------------------------------------------------------------------------------------------------------------')
            
            orders = extract_orders(data_spec['Wavelength'].values, 
                                    data_spec['Intensity'].values, 
                                    flx_err=data_spec['I_err'].values, 
                                    show_plot=False)
            
            if print_stat:
                print('Total {} spectral orders extracted'.format(len(orders)))
                print('----------------------------------------------------------------------------------------------------------------')
                    
            
            order_34 = orders[61-34] # The orders begin from # 61 so to get # 34, we index as 61-34.
            
            if print_stat:
                print('The #34 order wavelength read from .s file using pandas is: {}'.format(order_34[0]))
                print('The #34 order intensity read from .s file using pandas is: {}'.format(order_34[1]))
                print('The #34 order intensity error read from .s file using pandas is: {}'.format(order_34[2]))
                print('----------------------------------------------------------------------------------------------------------------')
        
            
            # The spectra is now doppler shift corrected in the wavelength axis using the stellar radial velocity and the rest wavelength of H alpha line; delta_lambda = (v/c)*lambda
            
            shift = ((radial_velocity/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in the .s files!
            
            wvl = np.round((order_34[0] - shift), 4) # Subtracting the calculated doppler shift value from the wavelength axis since the stellar radial velocity is positive. If the stellar RV is negative, the shift value will be added instead.
            flx = order_34[1] # Indexing flux array from order_34
            flx_err = order_34[2] # Indexing flux_err array from order_34
            
            # Creating a spectrum object called 'spec1d' using 'Spectrum1D' from 'specutils'
            # Docs for 'specutils' are here; https://specutils.readthedocs.io/en/stable/ 
            
            # The spectral and flux axes are given units nm and Jy respectively using 'astropy.units'. 
            # The uncertainty has units Jy as well!
        
            spec1d = Spectrum1D(spectral_axis=wvl*u.nm, 
                                flux=flx*u.Jy, 
                                uncertainty=StdDevUncertainty(flx_err, unit=u.Jy)) 
            
            # Printing info
            
            if print_stat:
                print('The doppler shift size using RV {} m/s and the H alpha line of 656.2808nm is: {}nm'.format(radial_velocity, shift))
                print('The spectral order used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 4 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('----------------------------------------------------------------------------------------------------------------')
                
            # Fitting an nth order polynomial to the continuum for normalisation using specutils
            
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('----------------------------------------------------------------------------------------------------------------')
                
                # 'fit_generic_continuum' is a function imported from 'specutils' which fits a given polynomial model to the given spectrum.
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis) # Continuum fit y values are calculated by inputting the spectral axis x values into the polynomial fit equation 
                
                spec_normalized = spec1d / y_cont_fitted # Spectrum is normalised by diving it with the polynomial fit
                
                # Plots the polynomial fits
                if plot_fit:
                    f, ax1 = plt.subplots()  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalized Flux')
                    ax1.set_title("Continuum Fitting")
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('----------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(HJD), format='pdf')
                    
                    f, ax2 = plt.subplots()  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, color='blue', label='Re-Normalized', alpha=0.6)
                    ax2.plot(spec1d.spectral_axis, spec1d.flux, color='red', label='Pipeline Normalized', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec1d.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec1d.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized ")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(HJD), format='pdf')
                        
                spec = spec_normalized # Note the continuum normalized spectrum also has new uncertainty values!
                
            else:
                spec = spec1d
                
                
            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            if plot_spec:
                f, ax  = plt.subplots()
                ax.plot(spec.spectral_axis, spec.flux, '-k')  
                ax.set_xlabel('$\lambda (nm)$')
                ax.set_ylabel("Normalized Flux")
                plt.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Hα {}±{}nm'.format(H_alpha_line, H_alpha_band/2))
                plt.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='blue', label='Blue cont. {}±{}nm'.format(F1_line, F1_band/2))
                plt.vlines(F1_line+(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='blue')
                plt.vlines(F2_line-(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='red', label='Red cont. {}±{}nm'.format(F2_line, F2_band/2))
                plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='red')
                ax.set_xlim(F1_line-1.1, F2_line+1.1)
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                plt.minorticks_on()
                ax.tick_params(direction='in', which='both')
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_reduced_spec_plot.pdf'.format(HJD), format='pdf')
                
                # Plots the zoomed in regions around the H alpha line.
                f, ax1  = plt.subplots()
                ax1.plot(spec.spectral_axis, spec.flux)
                ax1.set_xlabel('$\lambda (nm)$')
                ax1.set_ylabel("Normalized Flux")
                plt.vlines(H_alpha_line, ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='green')
                plt.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Hα band width = {}nm'.format(H_alpha_band))
                plt.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax1.set_xlim(H_alpha_line-(H_alpha_band/2)-0.1, H_alpha_line+(H_alpha_band/2)+0.1)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_H_alpha_line_plot.pdf'.format(HJD), format='pdf')
                
        # HARPS
        
        elif Instrument == 'HARPS':
            
            # Opening the FITS file using 'astropy.io.fits'
            # NOTE: The format of this FITS file must be ADP which contains the reduced spectrum with the wav, flux and flux_err in three columns
            
            file = fits.open(file_path[i])
            
            if ccf_file_path:
                ccf_file = fits.open(ccf_file_path[i]) # Opening the CCF FITS file to extract the RV
                RV = ccf_file[0].header['HIERARCH ESO DRS CCF RV']*1000 # Radial velocity converted from km/s to m/s
                
            else:
                RV = radial_velocity
            
            #Extracting useful information from the fits file header
            
            MJD = file[0].header['MJD-OBS'] # Modified Julian Date
            EXPTIME = file[0].header['EXPTIME'] # Exposure time in s
            OBS_DATE = file[0].header['DATE-OBS'] # Observation Date
            PROG_ID = file[0].header['PROG_ID'] # Program ID
            SNR = file[0].header['SNR'] # Signal to Noise ratio
            SIGDET = file[0].header['HIERARCH ESO DRS CCD SIGDET']  #CCD Readout Noise [e-]
            CONAD = file[0].header['HIERARCH ESO DRS CCD CONAD'] #CCD conversion factor [e-/ADU]; from e- to ADU
            RON = SIGDET * CONAD #CCD Readout Noise [ADU]
            
            # Defining each wavelength, flux and flux error arrays from the FITS file!
            
            wvl = file[1].data[0][0]/10 # dividing it by 10 to convert the wavelength from Å to nm!
            flx = file[1].data[0][1] # Flux in ADU
            flx_err = file[1].data[0][2]
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
           
            shift = ((radial_velocity/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 3)) # Using only 3 decimal places for the shift value since that's the precision of the wavelength in the .fits files!
            
            # Since the HARPS spectra have their individual spectral orders stitched together, we do not have to extract them separately as done for NARVAL. Thus for HARPS, the required region is extracted by slicing the spectrum with the index corresponding to the left and right continuum obtained using the 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            # If condition for when certain files have NaN as the flux errors; probably for all since the ESO Phase 3 data currently does not provide the flux errors
            
            flx_err_nan = np.isnan(np.sum(flx_err)) # NOTE: This returns true if there is one NaN or all are NaN!
            
            if flx_err_nan:
                if print_stat:
                    print('File contains NaN in flux errors array. Calculating flux error using CCD readout noise: {}'.format(np.round(RON, 4)))
                    print('----------------------------------------------------------------------------------------------------------------')
                # Flux error calculated as photon noise plus CCD readout noise 
                # NOTE: The error calculation depends on a lot of other CCD parameters such as the pixel binning in each CCD
                # array and so on. But for photometric limited measurements, this noise is generally insignificant.
                
                flx_err_ron = [np.sqrt(flux + np.square(RON)) for flux in flx]
                
                if np.isnan(np.sum(flx_err_ron)):
                    if print_stat:
                        print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                        print('----------------------------------------------------------------------------------------------------------------')
                
                # Slicing the data to contain only the region required for the index calculation as explained above and 
                # creating a spectrum class for it.
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err_ron[left_idx:right_idx], unit=u.Jy))
                
            else:
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx], unit=u.Jy))
            
            if print_stat:
                print('The doppler shift size using RV {} m/s and the H alpha line of 656.2808nm is: {}nm'.format(radial_velocity, shift))
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('----------------------------------------------------------------------------------------------------------------')
            
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('----------------------------------------------------------------------------------------------------------------')
                
                # 'fit_generic_continuum' is a function imported from 'specutils' which fits a given polynomial model to the given spectrum.
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis) # Continuum fit y values are calculated by inputting the spectral axis x values into the polynomial fit equation 
                spec_normalized = spec1d / y_cont_fitted
                
                spec = spec_normalized # Note the continuum normalized spectrum also has new uncertainty values which are simply the errors divided by this polynomial fit.
                
                # Plots the polynomial fits
                if plot_fit:
                    
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Flux (adu)')
                    ax1.set_title("Continuum Fitting")
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('----------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(MJD), format='pdf')
                    
                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, label='Re-Normalized')
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized ")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(MJD), format='pdf')
                
            else:
                spec = spec1d
                
                
            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            if plot_spec:
                f, ax  = plt.subplots(figsize=(10,4)) 
                ax.plot(spec.spectral_axis, spec.flux, '-k')  
                ax.set_xlabel('$\lambda (nm)$')
                ax.set_ylabel("Normalized Flux")
                plt.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Hα {}±{}nm'.format(H_alpha_line, H_alpha_band/2))
                plt.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='blue', label='Blue cont. {}±{}nm'.format(F1_line, F1_band/2))
                plt.vlines(F1_line+(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='blue')
                plt.vlines(F2_line-(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='red', label='Red cont. {}±{}nm'.format(F2_line, F2_band/2))
                plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='red')
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                plt.minorticks_on()
                ax.tick_params(direction='in', which='both')
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_reduced_spec_plot.pdf'.format(MJD), format='pdf')
                
                f, ax1  = plt.subplots(figsize=(10,4)) 
                ax1.plot(spec.spectral_axis, spec.flux)
                ax1.set_xlabel('$\lambda (nm)$')
                ax1.set_ylabel("Normalized Flux")
                plt.vlines(H_alpha_line, ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='green')
                plt.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Hα band width = {}nm'.format(H_alpha_band))
                plt.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax1.set_xlim(H_alpha_line-(H_alpha_band/2)-0.1, H_alpha_line+(H_alpha_band/2)+0.1)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_H_alpha_line_plot.pdf'.format(MJD), format='pdf')
                
        elif Instrument=='HARPS-N':
            
            # Opening the FITS file using 'astropy.io.fits'
            # NOTE: The format of this FITS file must be s1d which only contains flux array. 
            # The wavelength array is constructed using the starting point (CRVAL1), length of spectral axis (NAXIS1) 
            # and wavelength step (CDELT1) from the FITS file header.
            
            file = fits.open(file_path[i])
            
            if ccf_file_path:
                ccf_file = fits.open(ccf_file_path[i])  # Opening the CCF FITS file to extract the RV
                RV = ccf_file[0].header['HIERARCH TNG DRS CCF RV']*1000 # Radial velocity converted from km/s to m/s
                
            else:
                RV = radial_velocity
            
            #Extracting useful information from the fits file header
            
            MJD = file[0].header['MJD-OBS'] # Modified Julian Date
            EXPTIME = file[0].header['EXPTIME'] # Exposure time in seconds
            OBS_DATE = file[0].header['DATE-OBS'] # Observation Date
            PROG_ID = file[0].header['PROGRAM'] # Program ID
            
            
            flx = file[0].data # Flux in ADU
            wvl = file[0].header['CRVAL1'] + file[0].header['CDELT1']*np.arange(0, file[0].header['NAXIS1']) # constructing the spectral axis using start point, delta and axis length from file header
            wvl = wvl/10 # convert wvl from Å to nm!
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
            
            shift = ((RV/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 3)) 
            
            # Same as the HARPS spectra, the HARPS-N spectra have their individual spectral orders stitched together and 
            # we do not have to extract them separately as done for NARVAL. Thus, the required region is extracted by slicing
            # the spectrum with the index corresponding to the left and right continuum obtained using the 
            # 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            flx_err = [np.sqrt(flux) for flux in flx] # Using only photon noise as flx_err approx since no RON info available!
            
            # Slicing the data to contain only the region required for the index calculation as explained above and creating 
            # a spectrum class for it
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx], unit=u.Jy))
            
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('----------------------------------------------------------------------------------------------------------------')
                
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis)
                spec_normalized = spec1d / y_cont_fitted
                
                # Plots the polynomial fits
                if plot_fit:
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalized Flux')
                    ax1.set_title("Continuum Fitting")
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('----------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(MJD), format='pdf')
                    
                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, color='blue', label='Re-Normalized', alpha=0.6)
    #                 ax2.plot(spec1d.spectral_axis, spec1d.flux, color='red', label='Pipeline Normalized', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized ")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(MJD), format='pdf')
                    
                spec = spec_normalized # Note the continuum normalized spectrum also has new uncertainty values!
                
            else:
                
                spec = spec1d
            
            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            if plot_spec:
                f, ax  = plt.subplots(figsize=(10,4))  
                ax.plot(spec.spectral_axis, spec.flux, '-k')  
                ax.set_xlabel('$\lambda (nm)$')
                ax.set_ylabel("Normalized Flux")
                plt.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Hα {}±{}nm'.format(H_alpha_line, H_alpha_band/2))
                plt.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='blue', label='Blue cont. {}±{}nm'.format(F1_line, F1_band/2))
                plt.vlines(F1_line+(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='blue')
                plt.vlines(F2_line-(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='red', label='Red cont. {}±{}nm'.format(F2_line, F2_band/2))
                plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='red')
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                plt.minorticks_on()
                ax.tick_params(direction='in', which='both')
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_reduced_spec_plot.pdf'.format(MJD), format='pdf')
                
                f, ax1  = plt.subplots(figsize=(10,4))
                ax1.plot(spec.spectral_axis, spec.flux)
                ax1.set_xlabel('$\lambda (nm)$')
                ax1.set_ylabel("Normalized Flux")
                plt.vlines(H_alpha_line, ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='green')
                plt.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Hα band width = {}nm'.format(H_alpha_band))
                plt.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax1.set_xlim(H_alpha_line-(H_alpha_band/2)-0.1, H_alpha_line+(H_alpha_band/2)+0.1)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_H_alpha_line_plot.pdf'.format(MJD), format='pdf')
                    
        else:
            raise ValueError('Instrument type not recognised. Available options are "NARVAL", "HARPS" and "HARPS-N"')
            
        # Now we have the spectrum to work with as a variable, 'spec'!
        
        # The three regions required for H alpha index calculation are extracted from 'spec' using the 'extract region' function from 'specutils'. 
        # The function uses another function called 'SpectralRegion' as one of its arguments which defines the region to be extracted done so using the line and line bandwidth values; i.e. left end of region would be 'line - bandwidth/2' and right end would be 'line + bandwidth/2'.
        # Note: These values must have the same units as the spec wavelength axis.
            
        F_H_alpha_region = extract_region(spec, region=SpectralRegion((H_alpha_line-(H_alpha_band/2))*u.nm, (H_alpha_line+(H_alpha_band/2))*u.nm))
        
        # Mean of the flux within this region is calculated using np.mean and rounded off to 5 decimal places
        F_H_alpha_mean = np.round(np.mean(F_H_alpha_region.flux.value), 5)
        
        # The error on the mean flux is calculated as the standard error of the mean
        F_H_alpha_sum_err = [i**2 for i in F_H_alpha_region.uncertainty.array]
        F_H_alpha_mean_err = np.round((np.sqrt(np.sum(F_H_alpha_sum_err))/len(F_H_alpha_sum_err)), 5)
        
        # Same thing repeated for the F1 and F2 regions
        F1_region = extract_region(spec, region=SpectralRegion((F1_line-(F1_band/2))*u.nm, (F1_line+(F1_band/2))*u.nm))
        F1_mean = np.round(np.mean(F1_region.flux.value), 5)
        F1_sum_err = [i**2 for i in F1_region.uncertainty.array]
        F1_mean_err = np.round((np.sqrt(np.sum(F1_sum_err))/len(F1_sum_err)), 5)
        
        F2_region = extract_region(spec, region=SpectralRegion((F2_line-(F2_band/2))*u.nm, (F2_line+(F2_band/2))*u.nm))
        F2_mean = np.round(np.mean(F2_region.flux.value), 5)
        F2_sum_err = [i**2 for i in F2_region.uncertainty.array]
        F2_mean_err = np.round((np.sqrt(np.sum(F2_sum_err))/len(F2_sum_err)), 5)
        
        
                   
        if print_stat:
            print('H alpha region used ranges from {}nm to {}nm:'.format(F_H_alpha_region.spectral_axis[0].value, 
                                                                 F_H_alpha_region.spectral_axis[-1].value))
            print('F1 region used ranges from {}nm to {}nm:'.format(F1_region.spectral_axis[0].value, 
                                                                 F1_region.spectral_axis[-1].value))
            print('F2 region used ranges from {}nm to {}nm:'.format(F2_region.spectral_axis[0].value, 
                                                                 F2_region.spectral_axis[-1].value))
            print('----------------------------------------------------------------------------------------------------------------')
        
        # H alpha index is computed using the calculated mean fluxes.
        
        Hai_from_mean = np.round((F_H_alpha_mean/(F1_mean + F2_mean)), 5)
        
        # Continuum flux error is calculated as explained at the start of the tutorial Jupyter Notebook!
        
        sigma_F12_from_mean = np.sqrt((np.square(F1_mean_err) + np.square(F2_mean_err)))
        
        # Error on this index is calculated as explained at the start of the tutorial Jupyter notebook!
        
        sigma_Hai_from_mean = np.round((Hai_from_mean*np.sqrt(np.square(F_H_alpha_mean_err/F_H_alpha_mean) + np.square(sigma_F12_from_mean/(F1_mean+F2_mean)))), 5)
        
        if print_stat:
    
            print('Mean of {} flux points in H alpha: {}±{}'.format(len(F_H_alpha_region.flux), F_H_alpha_mean, F_H_alpha_mean_err))
            print('Mean of {} flux points in F1: {}±{}'.format(len(F1_region.flux), F1_mean, F1_mean_err))
            print('Mean of {} flux points in F2: {}±{}'.format(len(F2_region.flux), F2_mean, F2_mean_err))
            print('----------------------------------------------------------------------------------------------------------------')
            print('Index from mean of flux points in each band: {}±{}'.format(Hai_from_mean, sigma_Hai_from_mean))
            print('----------------------------------------------------------------------------------------------------------------')
            
        ## CaI_index calculation here
        
        if CaI_index:
            
            if print_stat:
                print('Calculating CaI Index')
                print('----------------------------------------------------------------------------------------------------------------')
            
            # Extracting the CaI region using the given CaI_line and CaI_band arguments 
            F_CaI_region = extract_region(spec, region=SpectralRegion((CaI_line-(CaI_band/2))*u.nm, (CaI_line+(CaI_band/2))*u.nm))
            F_CaI_mean = np.round(np.mean(F_CaI_region.flux.value), 5) # Calculating mean of the flux within this region
            
            # The error on the mean flux is calculated as the standard error of the mean
            F_CaI_sum_err = [i**2 for i in F_CaI_region.uncertainty.array]
            F_CaI_mean_err = np.round((np.sqrt(np.sum(F_CaI_sum_err))/len(F_CaI_sum_err)), 5)
            
            # Calculating the CaI index using the mean fluxes calculated above
            CaI_from_mean = np.round((F_CaI_mean/(F1_mean + F2_mean)), 5)
            
            # Index error calculated in the same way as that for H alpha index above
            sigma_CaI_from_mean = np.round((CaI_from_mean*np.sqrt(np.square(F_CaI_mean_err/F_CaI_mean) + np.square(sigma_F12_from_mean/(F1_mean+F2_mean)))), 5)
            
            if print_stat:
                print('Mean of {} flux points in CaI: {}±{}'.format(len(F_CaI_region.flux), F_CaI_mean, F_CaI_mean_err))
                print('----------------------------------------------------------------------------------------------------------------')
                print('Index from mean of flux points in each band: {}±{}'.format(CaI_from_mean, sigma_CaI_from_mean))
                print('----------------------------------------------------------------------------------------------------------------')
                
        else:
            
            CaI_from_mean = float('nan')
            sigma_CaI_from_mean = float('nan')
            

        if Instrument=='NARVAL':
            res = [HJD, Hai_from_mean, sigma_Hai_from_mean, CaI_from_mean, sigma_CaI_from_mean] # Creating results list 'res' containing the calculated parameters and appending this list to the 'results' empty list created at the start of this function!
            results.append(res)
        
        elif Instrument=='HARPS':
            res = [MJD, OBS_DATE, Hai_from_mean, sigma_Hai_from_mean, CaI_from_mean, sigma_CaI_from_mean, RV, EXPTIME, SNR, RON, PROG_ID]
            results.append(res)
            
        elif Instrument=='HARPS-N':
            res = [MJD, OBS_DATE, Hai_from_mean, sigma_Hai_from_mean, CaI_from_mean, sigma_CaI_from_mean, RV, EXPTIME, PROG_ID]
            results.append(res)
                
    
    # Saving the results in a csv file format  
    if save_results:
        
        if print_stat:
            print('Saving results in the working directory in file: {}.csv'.format(results_file_name))
            print('----------------------------------------------------------------------------------------------------------------')
            
            if Instrument=='NARVAL':
                
                header = ['HJD', 'I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err']
                
            elif Instrument=='HARPS':
                
                header = ['MJD', 'OBS_DATE', 'I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err', 'RV', 'T_exp', 'SNR', 'RON', 'PROG_ID']
                
            elif Instrument=='HARPS-N':
                
                header = ['MJD', 'OBS_DATE', 'I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err', 'RV', 'T_exp', 'PROG_ID']

        with open('{}.csv'.format(results_file_name), 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow(header)
            for row in results:
                writer.writerow(row)  
            
    return results

## Defining a function to calculate the NaI index following Rodrigo F. Díaz et al. 2007 (2007MNRAS.378.1007D)

def NaI_index_Rodrigo(file_path,
                      radial_velocity=9609,
                      degree=4,
                      NaID2=588.995,
                      NaID1=589.592,
                      NaI_band=0.1,
                      F1_line=580.5,
                      F1_band=1.0,
                      F2_line=609.0,
                      F2_band=2.0,
                      hfv=10, 
                      Instrument='NARVAL',
                      Stokes_profile=['V'],
                      norm_spec=False,
                      plot_fit=False,
                      plot_spec=True,
                      print_stat=True,
                      save_results=False,
                      results_file_name=None,
                      save_figs=False,
                      out_file_path=None,
                      ccf_file_path=None):
    
    """
    
    This function calculates the NaI doublet index following the method proposed in Rodrigo F. Díaz et al. 2007.
    
    Parameters:
    -----------
    
    file_path: list, .s format (NARVAL), ADP..._.fits format (HARPS) or s1d_A.fits format (HARPS-N)
    List containing paths of the spectrum files
    
    radial_velocity: int, default: 9609 m/s
    Stellar radial velocity along the line-of-sight taken from GAIA DR2 for GJ 436.
    This value is used for doppler shifting the spectra to its rest frame.
    
    degree: int, default: 4
    The degree of the Chebyshev1D polynomial to fit to the continuum for normalisation.
    Normalisation done using Specutils. 
    For more info, see https://specutils.readthedocs.io/en/stable/api/specutils.fitting.fit_generic_continuum.html#specutils.fitting.fit_generic_continuum
    
    NaID1: int, default: 588.995 nm
    Line centre for the first doublet in nm.
    
    NaID2: int, default: 589.592 nm
    Line centre for the second doublet in nm.
    
    NaI_band: int, default: 0.1 nm
    Band width (nm) in which to calculate the mean doublet flux value.
    
    F1_line: int, default: 580.5 nm
    Centre of the blue continuum for pseudo-cont. estimation
    
    F1_band: int, default: 1.0 nm
    Band width (nm) in which to estimate the continuum flux.
    
    F2_line: int, default: 609.0 nm
    Centre of the red continuum for pseudo-cont. estimation
    
    F2_band: int, default: 2.0 nm
    Band width (nm) in which to estimate the continuum flux.
    
    hfv: int, default: 10
    Number of highest flux values (hfv) to use for estimating the continuum flux in each red/blue band.
    
    Instrument: str, default: 'NARVAL'
    The instrument from which the data has been collected. Input takes arguments 'NARVAL', 'HARPS' or 'HARPS-N'.
    
    Stokes_profile: str, default: ['V']
    The Stokes profile for the input data. 'V' for per night and 'I' for per sub-exposure per night. Used only when Instrument type is 'NARVAL'
    
    norm_spec: bool, default: False
    Normalizes ths spectrum.
    
    plot_fit: bool, default: False
    Plots the continuum fitting normalization processes.
    
    plot_spec: bool, default: True
    Plots the final reduced spectrum.
    
    print_stat: bool, default: True
    Prints the status of each process within the function.
    
    save_results: bool, default: False
    Saves the run results in a .csv file format in the working directory
    
    results_file_name: str, default: None
    Name of the file with the which the results file is saved
    
    save_figs: bool, default: False
    Save the plots in a pdf format in the working directory
    
    out_file_path: list, .out format (NARVAL), default: None
    List containing paths of the .out files used to extract OBS_HJD.
    
    ccf_file_path: list, .fits format (HARPS/HARPS-N), default: None
    List containing paths of the CCF FITS files used to extract the radial velocity. If None, the given radial velocity arg is used for all files
    
    Returns:
    -----------
    NARVAL: HJD of observation, NaI index, error on index, pseudo-continuum estimation and error on the pseudo-continuum.
    HARPS: MJD of observation, Observation date, NaI index, error on index, radial velocity, exposure time (s), SNR, ReadOut noise and Program ID
    HARPS-N: MJD of observation, Observation date, NaI index, error on index, radial velocity, exposure time (s) and Program ID
    
    All values are type float().
    
    """
    
    
    results = [] # Empty list to which the run results will be appended
    
    # Creating a loop to go through each given file_path in the list of file paths
    
    # Using the tqdm function 'log_progress' to provide a neat progress bar in Jupyter Notebook which shows the total number of
    # runs, the run time per iteration and the total run time for all files!
    
    for i in log_progress(range(len(file_path)), desc='Calculating NaI Index'):
        
        # Creating a loop for each instrument type.
        
        ## NARVAL
        
        if Instrument=='NARVAL':
            
            if out_file_path != None:
                
                file = open(out_file_path[i]).readlines() # Opening the .out file and reading each line as a string
                
                string = '   Heliocentric Julian date (UTC) :' # Creating a string variable that matches the string in the .out file
                
                idx = find_string_idx(out_file_path[i], string) # Using the 'find_string_idx' function to find the index of the line that contains the above string. 
                
                HJD = float(file[idx][-14:-1]) # Using the line index found above, the HJD is extracted by indexing just that from the line.
                
            else:
                if print_stat:
                    print('out_file_path not given as an argument. Returning HJD as NaN instead.')
                    print('----------------------------------------------------------------------------------------------------------------')
                HJD = float('nan')
                
            # Defining column names for pandas to read the file easily
    
            col_names_V = ['Wavelength', 'Intensity', 'Polarized', 'N1', 'N2', 'I_err'] # For Stokes V 
            col_names_I = ['Wavelength', 'Intensity', 'I_err'] # For Stokes I 
            
            # Reading data using pandas and skipping the first 2 rows
            
            if Stokes_profile==['V']:
                data_spec = pd.read_fwf(file_path[i], names=col_names_V, skiprows=2) 
            else:
                data_spec = pd.read_fwf(file_path[i], names=col_names_I, skiprows=2)
            
            # Extracting indivdidual spectral orders using 'extract_orders'
            
            if print_stat:
                print('Extracting spectral orders')
                print('----------------------------------------------------------------------------------------------------------------')
            
            spectral_orders = extract_orders(data_spec['Wavelength'],
                                             data_spec['Intensity'],
                                             data_spec['I_err'])
            
            ord_39 = spectral_orders[61-39] # order 39 contains the F1 line
            ord_38 = spectral_orders[61-38] # Both order 39 and 38 contain the D1 and D2 lines but only order 38 is used since it has a higher SNR; (see .out file)
            ord_37 = spectral_orders[61-37] # order 37 contains the F2 line
            
            if print_stat:
                print('Using orders #39, #38 and #37 for Index calculation')
                print('----------------------------------------------------------------------------------------------------------------')
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c
            
            shift = ((radial_velocity/ap.constants.c.value)*NaID1) # Using the rest wavelength of NaID1 line
            shift = (round(shift, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in .s file!
            
            # Creating three spectrum classes for each of the three orders using 'Spectrum1D' from 'specutils'
            # Docs for 'specutils' here; https://specutils.readthedocs.io/en/stable/ 
            
            # The spectral and flux axes are given nm and Jy units using 'astropy.units' as 'u'. The uncertainty has units Jy as well!
            
            
            spec1 = Spectrum1D(spectral_axis=np.round((ord_39[0].values - shift), 4)*u.nm, 
                               flux=ord_39[1].values*u.Jy, 
                               uncertainty=StdDevUncertainty(ord_39[2].values))
            
            spec2 = Spectrum1D(spectral_axis=np.round((ord_38[0].values - shift), 4)*u.nm, 
                               flux=ord_38[1].values*u.Jy, 
                               uncertainty=StdDevUncertainty(ord_38[2].values))
            
            spec3 = Spectrum1D(spectral_axis=np.round((ord_37[0].values - shift), 4)*u.nm, 
                               flux=ord_37[1].values*u.Jy, 
                               uncertainty=StdDevUncertainty(ord_37[2].values))
                            
            if print_stat:
                print('The three spectral orders used range from; {}nm-{}nm, {}nm-{}nm, and {}nm-{}nm'.format(spec1.spectral_axis[0].value, 
                                                                                                              spec1.spectral_axis[-1].value,
                                                                                                              spec2.spectral_axis[0].value, 
                                                                                                              spec2.spectral_axis[-1].value, 
                                                                                                              spec3.spectral_axis[0].value, 
                                                                                                              spec3.spectral_axis[-1].value))
                print('----------------------------------------------------------------------------------------------------------------')
                print('The doppler shift size is: {}nm'.format(shift))
                print('----------------------------------------------------------------------------------------------------------------')
                
            # Fitting the continuum for each order separately using 'specutils'
        
            if norm_spec:
                if print_stat:
                    print('Normalising the spectras by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('----------------------------------------------------------------------------------------------------------------')
                    
                # First order
                      
                g1_fit = fit_generic_continuum(spec1, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted1 = g1_fit(spec1.spectral_axis) # Continuum fit y values are calculated by inputting the spectral axis x values into the polynomial fit equation 
                
                # The spectrum is divided by the continuum fit to get the normalized spectrum
                spec_normalized1 = spec1 / y_cont_fitted1 # Note the continuum normalized spectrum also has new uncertainty values!
                
                if plot_fit:
                    f, ax1 = plt.subplots()  
                    ax1.plot(spec1.spectral_axis, spec1.flux)  
                    ax1.plot(spec1.spectral_axis, y_cont_fitted1)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalized Flux')
                    ax1.set_title("Continuum Fitting First Order") 
                    
                    f, ax2 = plt.subplots()  
                    ax2.plot(spec_normalized1.spectral_axis, spec_normalized1.flux, label='Normalized', alpha=0.6)
                    ax2.plot(spec1.spectral_axis, spec1.flux, color='red', label='Non-Normalized', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(F1_line+(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized First Order")  
                    plt.legend()
                    
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('----------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_F1_plot.pdf'.format(HJD), format='pdf')
                          
                # Second order
                      
                g2_fit = fit_generic_continuum(spec2, model=Chebyshev1D(degree))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted2 = g2_fit(spec2.spectral_axis)
                spec_normalized2 = spec2 / y_cont_fitted2
                
                if plot_fit:
                    f, ax1 = plt.subplots()  
                    ax1.plot(spec2.spectral_axis, spec2.flux)  
                    ax1.plot(spec2.spectral_axis, y_cont_fitted2)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalized Flux')
                    ax1.set_title("Continuum Fitting Second Order") 
                    
                    f, ax2 = plt.subplots()  
                    ax2.plot(spec_normalized2.spectral_axis, spec_normalized2.flux, label='Normalized', alpha=0.6)
                    ax2.plot(spec2.spectral_axis, spec2.flux, color='red', label='Non-Normalized', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(NaID1-1.0, ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(NaID2+1.0, ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized Second Order")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_fit_F2_plot.pdf'.format(HJD), format='pdf')
                          
                # Third order
                      
                g3_fit = fit_generic_continuum(spec3, model=Chebyshev1D(degree))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted3 = g3_fit(spec3.spectral_axis)
                spec_normalized3 = spec3 / y_cont_fitted3
                
                if plot_fit:
                    f, ax1 = plt.subplots()  
                    ax1.plot(spec3.spectral_axis, spec3.flux)  
                    ax1.plot(spec3.spectral_axis, y_cont_fitted3)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalized Flux')
                    ax1.set_title("Continuum Fitting Third Order") 
                    
                    f, ax2 = plt.subplots()  
                    ax2.plot(spec_normalized3.spectral_axis, spec_normalized3.flux, label='Normalized', alpha=0.6)
                    ax2.plot(spec3.spectral_axis, spec3.flux, color='red', label='Non-Normalized', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(F2_line-(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized Third Order")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_fit_F3_plot.pdf'.format(HJD), format='pdf')
                
                spec1 = spec_normalized1
                spec2 = spec_normalized2
                spec3 = spec_normalized3
                
            # Extracting the regions required for index calculation from each spectrum using 'extract_region' and the given bandwidths
            
            NaID1_region = extract_region(spec2, region=SpectralRegion((NaID1-(NaI_band/2))*u.nm, 
                                                                       (NaID1+(NaI_band/2))*u.nm))
            
            NaID2_region = extract_region(spec2, region=SpectralRegion((NaID2-(NaI_band/2))*u.nm, 
                                                                       (NaID2+(NaI_band/2))*u.nm))
            # Using spec1 for blue continuum
            
            F1_region = extract_region(spec1, region=SpectralRegion((F1_line-(F1_band/2))*u.nm, 
                                                                       (F1_line+(F1_band/2))*u.nm))
            # Using spec3 for red continuum
            
            F2_region = extract_region(spec3, region=SpectralRegion((F2_line-(F2_band/2))*u.nm, 
                                                                       (F2_line+(F2_band/2))*u.nm))
            
            # Definig the pseudo-continuum
            
            # Sorting the flux in F1 region from lowest to highest and using only the given number of highest flux values, (hfv), for the mean.
            
            F1_sorted_flux = F1_region.flux[np.argsort(-F1_region.flux)[:hfv]] 
            F1_mean = np.round(np.mean(F1_sorted_flux), 5)
            F1_err = F1_region.uncertainty.array[np.argsort(-F1_region.flux)[:hfv]]
            
            # The error on this mean is calculated using error propagation
            
            F1_sum_err = [i**2 for i in F1_err]
            F1_err = np.round((np.sqrt(np.sum(F1_sum_err))/len(F1_sum_err)), 5)
            
            # Same process for F2 region
            
            F2_sorted_flux = F2_region.flux[np.argsort(-F2_region.flux)[:hfv]]
            F2_mean = np.round(np.mean(F2_sorted_flux), 5)
            F2_err = F2_region.uncertainty.array[np.argsort(-F2_region.flux)[:hfv]]
            
            F2_sum_err = [i**2 for i in F2_err]
            F2_err = np.round((np.sqrt(np.sum(F2_sum_err))/len(F2_sum_err)), 5)
            
            # The pseudo-continuum is taken as the mean of the fluxes calculated abvove in F1 and F2 regions
            
            F_cont = np.round(((F1_mean+F2_mean)/2), 5) # This value is used for the index calculation
            F_cont_err = np.round((np.sqrt(F1_err**2 + F2_err**2)/2), 5) # Error calculated using error propagation
            
            # Plotting the pseudo-continuum as the linear interpolation of the values in each red and blue cont. window!
            
            if plot_spec:
                
                x = [F1_line, F2_line]
                y = [F1_mean.value, F2_mean.value]
                
                f, ax  = plt.subplots(figsize=(10,4)) 
                ax.plot(spec1.spectral_axis, spec1.flux, color='red', alpha=0.5)
                ax.plot(spec2.spectral_axis, spec2.flux, color='blue', alpha=0.5)
                ax.plot(spec3.spectral_axis, spec3.flux, color='green', alpha=0.5)
                ax.plot(x, y, 'or--')
                ax.set_xlabel('$\lambda (nm)$')
                ax.set_ylabel("Normalized Flux")
                ax.set_title('Overplotting 3 orders around NaI D lines')
                plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='--', colors='black')
                plt.axhline(1.0, ls='--', c='gray')
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_reduced_spec_plot.pdf'.format(HJD), format='pdf')
                
            # Calculating the mean flux in the D1 D2 lines
            
            NaID1_mean = np.round(np.mean(NaID1_region.flux.value), 5)
            
            # Error calculated using error propagation
            NaID1_sum_err = [i**2 for i in NaID1_region.uncertainty.array]
            NaID1_err = np.round((np.sqrt(np.sum(NaID1_sum_err))/len(NaID1_sum_err)), 5)
            
            NaID2_mean = np.round(np.mean(NaID2_region.flux.value), 5)
            NaID2_sum_err = [i**2 for i in NaID2_region.uncertainty.array]
            NaID2_err = np.round((np.sqrt(np.sum(NaID2_sum_err))/len(NaID2_sum_err)), 5)
            
            # Error on the sum of mean fluxes in D1 and D2
            sigma_D12 = np.sqrt(np.square(NaID1_err) + np.square(NaID2_err))
            
            # Calculating the index and rounding it up to 5 decimal places
            NaID_index = np.round(((NaID1_mean + NaID2_mean)/F_cont.value), 5)
            
            # Error calculated using error propagation and rounding it up to 5 decimal places
            sigma_NaID_index = np.round((NaID_index*np.sqrt(np.square(sigma_D12/(NaID1_mean + NaID2_mean)) + np.square(F_cont_err/F_cont.value))), 5)
            
            if print_stat:
                print('Using {} higher flux values in each band for the pseudo-cont. calculation'.format(hfv))
                print('----------------------------------------------------------------------------------------------------------------')
                print('Flux in blue cont. is {}±{}'.format(F1_mean, F1_err))
                print('Flux in red cont. is {}±{}'.format(F2_mean, F2_err))
                print('Mean cont. flux is {}±{}'.format(F_cont.value, F_cont_err))
                print('NaID1 mean flux is {}±{}'.format(NaID1_mean, NaID1_err))
                print('NaID2 mean flux is {}±{}'.format(NaID2_mean, NaID2_err))
                print('----------------------------------------------------------------------------------------------------------------')
                print('The NaI doublet index is: {}±{}'.format(NaID_index, sigma_NaID_index))
                print('----------------------------------------------------------------------------------------------------------------')
            
            res = [HJD, NaID_index, sigma_NaID_index, F_cont.value, F_cont_err] # Creating a list containing the results for this file
            results.append(res) # Appending the res list into the empty results list created at the start of this function
                
        ## HARPS
                
        elif Instrument=='HARPS':
            
            # Opening the FITS file using 'astropy.io.fits'
            # NOTE: The format of this FITS file must be ADP which contains the reduced spectrum with the wav, flux and flux_err in three columns
            
            file = fits.open(file_path[i]) 
            
            if ccf_file_path:
                ccf_file = fits.open(ccf_file_path[i]) # Opening the CCF FITS file to extract the RV
                RV = ccf_file[0].header['HIERARCH ESO DRS CCF RV']*1000 # Radial velocity converted to m/s
                
            else:
                RV = radial_velocity
            
            #Extracting useful information from the fits file header
            
            MJD = file[0].header['MJD-OBS'] # Modified Julian Date
            EXPTIME = file[0].header['EXPTIME'] # Exposure time in s
            OBS_DATE = file[0].header['DATE-OBS'] # Observation Date
            PROG_ID = file[0].header['PROG_ID'] # Program ID
            SNR = file[0].header['SNR'] # Signal to Noise ratio
            SIGDET = file[0].header['HIERARCH ESO DRS CCD SIGDET'] #CCD Readout Noise [e-]
            CONAD = file[0].header['HIERARCH ESO DRS CCD CONAD']  #CCD conversion factor [e-/ADU]; from e- to ADU
            RON = SIGDET * CONAD #CCD Readout Noise [ADU]
            
            
            # Defining each wavelength, flux and flux error arrays from the FITS file!
            
            wvl = file[1].data[0][0]/10 # dividing it by 10 to convert the wavelength from Å to nm!
            flx = file[1].data[0][1] # ADU
            flx_err = file[1].data[0][2]
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
            
            shift = ((RV/ap.constants.c.value)*NaID1)  
            shift = (round(shift, 3)) # Using only 3 decimal places for the shift value since that's the precision of the wavelength in the .FITS files!
            
            # Since the HARPS spectra have their individual spectral orders stitched together, we do not have to extract them separately as done for NARVAL. Thus for HARPS, the required region is extracted by slicing the spectrum with the index corresponding to the left and right continuum obtained using the 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2 nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            # If condition for when certain files have NaN as the flux errors; probably for all since the ESO Phase 3 data currently does not the flux errors
            flx_err_nan = np.isnan(np.sum(flx_err)) # NOTE: This returns True if there is one NaN value or all are NaN values!
            
            if flx_err_nan:
                if print_stat:
                    print('File contains NaN in flux errors array. Calculating flux errors using CCD readout noise: {}'.format(np.round(RON, 4)))
                    print('----------------------------------------------------------------------------------------------------------------')
                
                # Flux error calculated as photon noise plus CCD readout noise 
                # NOTE: The error calculation depends on a lot of other CCD parameters such as the pixel binning in each CCD
                # array and so on. But for photometric limited measurements, this noise is generally insignificant.
                
                flx_err_ron = [np.sqrt(flux + np.square(RON)) for flux in flx]
                
                if np.isnan(np.sum(flx_err_ron)):
                    if print_stat:
                        print('The calculated flux array contains a few NaN values due to negative flux encountered in the square root.')
                        print('----------------------------------------------------------------------------------------------------------------')
                
                # Slicing the data to contain only the region required for the index calculation as explained above and 
                # creating a spectrum class for it.
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err_ron[left_idx:right_idx], unit=u.Jy))
                
            else:
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx], unit=u.Jy))
            
            if print_stat:
                print('The doppler shift size using NaID1 is: {}nm'.format(shift))
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('----------------------------------------------------------------------------------------------------------------')
            
            if norm_spec=='scale':
                if print_stat:
                    print('Normalizing the spectra by scaling it down to max. flux equals 1.0')
                    print('----------------------------------------------------------------------------------------------------------------')
                    
                flux_norm = (spec1d.flux - min(spec1d.flux))/(max(spec1d.flux) - min(spec1d.flux)) # Same normalization as the ACTIN code. See here for more info on ACTIN https://github.com/gomesdasilva/ACTIN 
                
                spec_normalized = Spectrum1D(spectral_axis=spec1d.spectral_axis,
                                             flux=flux_norm*u.Jy,
                                             uncertainty=StdDevUncertainty(spec1d.uncertainty.array, unit=u.Jy))
                                 
                spec = spec_normalized 
                
                if plot_fit:
                    
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec.spectral_axis, spec.flux, label='Scaled down spectra')
                    plt.axhline(1.0, ls='--', c='gray')
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalized Flux')
                    ax1.set_title("Continuum Normalized ")  
                    plt.legend()
                    
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('----------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(MJD), format='pdf')
                            
            elif norm_spec=='poly1dfit':
                if print_stat:
                    print('Normalizing the spectra by fitting a 1st degree polynomial to the continuum')
                    print('----------------------------------------------------------------------------------------------------------------')
                
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(1))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis)
                
                # Adding a constant offset to the continuum fit to lift it up so it starts from the centre of the spectrum
                left_init_mean = np.mean(spec1d.flux.value[:100]) + 150.0
                delta_flx = left_init_mean*u.Jy - y_cont_fitted[0]
                y_cont_fitted += delta_flx
                
                spec_normalized = spec1d / y_cont_fitted
                
                spec = spec_normalized
                
                if plot_fit:
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Flux (adu)')
                    ax1.set_title("Continuum Fitting") 
                    
                    if save_figs:
                        plt.savefig('{}_cont_fit_plot.pdf'.format(MJD), format='pdf')
                    
                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec.spectral_axis, spec.flux, label='Normalized')
                    ax2.axhline(1.0, ls='--', c='gray')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(MJD), format='pdf')
                
                    
            else:
                spec = spec1d
                
                
            # Extracting the regions required for the index calculation using 'extract_region'
            
            NaID1_region = extract_region(spec, region=SpectralRegion((NaID1-(NaI_band/2))*u.nm, 
                                                                      (NaID1+(NaI_band/2))*u.nm))
            
            NaID2_region = extract_region(spec, region=SpectralRegion((NaID2-(NaI_band/2))*u.nm, 
                                                                      (NaID2+(NaI_band/2))*u.nm))
            
            F1_region = extract_region(spec, region=SpectralRegion((F1_line-(F1_band/2))*u.nm, 
                                                                   (F1_line+(F1_band/2))*u.nm))
            
            F2_region = extract_region(spec, region=SpectralRegion((F2_line-(F2_band/2))*u.nm, 
                                                                   (F2_line+(F2_band/2))*u.nm))
    
             # Definig the pseudo-continuum; same process as that for NARVAL above
            
            F1_sorted_flux = F1_region.flux[np.argsort(-F1_region.flux)[:hfv]] 
            F1_mean = np.round(np.mean(F1_sorted_flux), 5)
            F1_err = F1_region.uncertainty.array[np.argsort(-F1_region.flux)[:hfv]]
                        
            F1_sum_err = [i**2 for i in F1_err]
            F1_err = np.sqrt(np.sum(F1_sum_err))/len(F1_sum_err)
            
            F2_sorted_flux = F2_region.flux[np.argsort(-F2_region.flux)[:hfv]]
            F2_mean = np.round(np.mean(F2_sorted_flux), 5)
            F2_err = F2_region.uncertainty.array[np.argsort(-F2_region.flux)[:hfv]]
            
            F2_sum_err = [i**2 for i in F2_err]
            F2_err = np.sqrt(np.sum(F2_sum_err))/len(F2_sum_err)
            
            F_cont = np.round(((F1_mean+F2_mean)/2), 5) # This value is used for the index calc.
            F_cont_err = np.round((np.sqrt(F1_err**2 + F2_err**2)/2), 5)
            
            if plot_spec:
                
                x = [F1_line, F2_line]
                y = [F1_mean.value, F2_mean.value]
    
                f, ax  = plt.subplots(figsize=(10,4)) 
                ax.plot(spec.spectral_axis, spec.flux, color='black')
                ax.axhline(1.0, ls='--', c='gray')
                ax.plot(x, y, 'og--')
                plt.vlines(NaID1-(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(NaID1+(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(NaID2-(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(NaID2+(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(F1_line-(F1_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='blue')
                plt.vlines(F1_line+(F1_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='blue')
                plt.vlines(F2_line-(F2_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='red')
                plt.vlines(F2_line+(F2_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='red')
                ax.set_xlabel('$\lambda (nm)$')
                ax.set_ylabel("Normalized Flux")
                
                if save_figs:
                        plt.savefig('{}_reduced_spec_plot.pdf'.format(MJD), format='pdf')
            
            # Calculating mean flux in the D1,2 lines
            
            NaID1_mean = np.round(np.mean(NaID1_region.flux.value), 5)
            
            # Error calculated using error propagation
            
            NaID1_sum_err = [i**2 for i in NaID1_region.uncertainty.array]
            NaID1_err = np.round((np.sqrt(np.sum(NaID1_sum_err))/len(NaID1_sum_err)), 5)
            
            NaID2_mean = np.round(np.mean(NaID2_region.flux.value), 5)
            NaID2_sum_err = [i**2 for i in NaID2_region.uncertainty.array]
            NaID2_err = np.round((np.sqrt(np.sum(NaID2_sum_err))/len(NaID2_sum_err)), 5)
            
            sigma_D12 = np.sqrt(np.square(NaID1_err) + np.square(NaID2_err)) # Error on the sum of the mean fluxes in D1 and D2
            
            # Calculating the NaI index
            
            NaID_index = np.round(((NaID1_mean + NaID2_mean)/F_cont.value), 5)
            
            # Error on NaI index calculated using error propagation
            
            sigma_NaID_index = np.round((NaID_index*np.sqrt(np.square(sigma_D12/(NaID1_mean + NaID2_mean)) + np.square(F_cont_err/F_cont.value))), 5)
            
            if print_stat:
                print('Using {} higher flux values in each band for the pseudo-cont. calculation'.format(hfv))
                print('----------------------------------------------------------------------------------------------------------------')
                print('Flux in blue cont. is {}'.format(F1_mean))
                print('Flux in red cont. is {}'.format(F2_mean))
                print('Mean cont. flux is {}±{}'.format(F_cont.value, F_cont_err))
                print('NaID1 mean flux is {}±{}'.format(NaID1_mean, NaID1_err))
                print('NaID2 mean flux is {}±{}'.format(NaID2_mean, NaID2_err))
                print('----------------------------------------------------------------------------------------------------------------')
                print('The NaI doublet index is: {}±{}'.format(NaID_index, sigma_NaID_index))
                print('----------------------------------------------------------------------------------------------------------------')
            
            res = [MJD, OBS_DATE, NaID_index, sigma_NaID_index, RV, EXPTIME, SNR, RON, PROG_ID] # Creating a list containing the results for this file
            results.append(res) # Appending the res list into the empty results list created at the start of this function
        
        elif Instrument=='HARPS-N':
            
            # Opening the FITS file using 'astropy.io.fits'
            # NOTE: The format of this FITS file must be s1d which only contains flux array. 
            # The wavelength array is constructed using the starting point (CRVAL1), length of spectral axis (NAXIS1) 
            # and wavelength step (CDELT1) from the FITS file header.
            
            file = fits.open(file_path[i])
            
            if ccf_file_path:
                ccf_file = fits.open(ccf_file_path[i]) # Opening the CCF FITS file to extract the RV
                RV = ccf_file[0].header['HIERARCH TNG DRS CCF RV']*1000 # Radial velocity in m/s
                
            else:
                RV = radial_velocity
            
            #Extracting useful information from the FITS file header
            
            MJD = file[0].header['MJD-OBS'] # Modified Julian Date
            EXPTIME = file[0].header['EXPTIME'] # Exposure time in s
            OBS_DATE = file[0].header['DATE-OBS'] # Observation Date
            PROG_ID = file[0].header['PROGRAM'] # Program ID
            
            flx = file[0].data # ADU
            wvl = file[0].header['CRVAL1'] + file[0].header['CDELT1']*np.arange(0, file[0].header['NAXIS1']) 
            wvl = wvl/10 # convert wvl from Å to nm!
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
            
            shift = ((RV/ap.constants.c.value)*NaID1)  
            shift = (round(shift, 3)) 
            
             # Same as the HARPS spectra, the HARPS-N spectra have their individual spectral orders stitched together and 
             # we do not have to extract them separately as done for NARVAL. Thus, the required region is extracted by slicing
             # the spectrum with the index corresponding to the left and right continuum obtained using the 
             # 'find_nearest' function. 
            
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            flx_err = [np.sqrt(flux) for flux in flx] # Using only photon noise as flx_err approx since no RON info available!
            
            # Slicing the data to contain only the region required for the index calculation as explained above and creating 
            # a spectrum class for it
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx], unit=u.Jy))
            
            if print_stat:
                print('The doppler shift size using NaID1 is: {}nm'.format(shift))
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('----------------------------------------------------------------------------------------------------------------')
                
            if norm_spec=='scale':
                if print_stat:
                    print('Normalizing the spectra by scaling it down to max. flux equals 1.0')
                    print('----------------------------------------------------------------------------------------------------------------')
                flux_norm = (spec1d.flux - min(spec1d.flux))/(max(spec1d.flux) - min(spec1d.flux)) # Same normalization as the ACTIN code!
                spec_normalized = Spectrum1D(spectral_axis=spec1d.spectral_axis,
                                             flux=flux_norm*u.Jy,
                                             uncertainty=StdDevUncertainty(spec1d.uncertainty.array, unit=u.Jy))
                                 
                spec = spec_normalized 
                
                if plot_fit:
                    
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec.spectral_axis, spec.flux, label='Scaled down spectra')
                    plt.axhline(1.0, ls='--', c='gray')
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalized Flux')
                    ax1.set_title("Continuum Normalized ")  
                    plt.legend()
                    
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('----------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(MJD), format='pdf')
                    
            elif norm_spec=='poly1dfit':
                if print_stat:
                    print('Normalizing the spectra by fitting a 1st degree polynomial to the continuum')
                    print('----------------------------------------------------------------------------------------------------------------')
                
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(1))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis)
                
                # Adding a constant offset to the continuum fit to lift it up so it starts from the centre of the spectrum
                left_init_mean = np.mean(spec1d.flux.value[:100]) + 150.0
                delta_flx = left_init_mean*u.Jy - y_cont_fitted[0]
                y_cont_fitted += delta_flx
                
                spec_normalized = spec1d / y_cont_fitted
                
                spec = spec_normalized
                
                if plot_fit:
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Flux (adu)')
                    ax1.set_title("Continuum Fitting") 
                    
                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec.spectral_axis, spec.flux, label='Normalized')
                    ax2.axhline(1.0, ls='--', c='gray')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_fit_plot.pdf'.format(MJD), format='pdf')
                
                    
            else:
                spec = spec1d
                
            if plot_spec:
    
                ax  = plt.subplots()[1]  
                ax.plot(spec.spectral_axis, spec.flux, color='black')
                plt.vlines(NaID1-(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='black')
                plt.vlines(NaID1+(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='black')
                plt.vlines(NaID2-(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='black')
                plt.vlines(NaID2+(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='black')
                plt.vlines(F1_line-(F1_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='blue')
                plt.vlines(F1_line+(F1_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='blue')
                plt.vlines(F2_line-(F2_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='red')
                plt.vlines(F2_line+(F2_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='red')
                ax.set_xlabel('$\lambda (nm)$')
                ax.set_ylabel("Normalized Flux")
                
                if save_figs:
                        plt.savefig('{}_reduced_spec_plot.pdf'.format(MJD), format='pdf')
                
                
            # Extracting the regions required for the index calculation using 'extract_region'
            
            NaID1_region = extract_region(spec, region=SpectralRegion((NaID1-(NaI_band/2))*u.nm, 
                                                                      (NaID1+(NaI_band/2))*u.nm))
            
            NaID2_region = extract_region(spec, region=SpectralRegion((NaID2-(NaI_band/2))*u.nm, 
                                                                      (NaID2+(NaI_band/2))*u.nm))
            
            F1_region = extract_region(spec, region=SpectralRegion((F1_line-(F1_band/2))*u.nm, 
                                                                   (F1_line+(F1_band/2))*u.nm))
            
            F2_region = extract_region(spec, region=SpectralRegion((F2_line-(F2_band/2))*u.nm, 
                                                                   (F2_line+(F2_band/2))*u.nm))
            
            # Defining the pseudo-continuum; same process as that for NARVAL above
            
            F1_sorted_flux = F1_region.flux[np.argsort(-F1_region.flux)[:hfv]] 
            F1_mean = np.round(np.mean(F1_sorted_flux), 5)
            F1_err = F1_region.uncertainty.array[np.argsort(-F1_region.flux)[:hfv]]
                        
            F1_sum_err = [i**2 for i in F1_err]
            F1_err = np.sqrt(np.sum(F1_sum_err))/len(F1_sum_err)
            
            F2_sorted_flux = F2_region.flux[np.argsort(-F2_region.flux)[:hfv]]
            F2_mean = np.round(np.mean(F2_sorted_flux), 5)
            F2_err = F2_region.uncertainty.array[np.argsort(-F2_region.flux)[:hfv]]
            
            F2_sum_err = [i**2 for i in F2_err]
            F2_err = np.sqrt(np.sum(F2_sum_err))/len(F2_sum_err)
            
            F_cont = np.round(((F1_mean+F2_mean)/2), 5) # This value is used for the index calc.
            F_cont_err = np.round((np.sqrt(F1_err**2 + F2_err**2)/2), 5)
            
            # Calculating mean flux in the D1,2 lines
            
            NaID1_mean = np.round(np.mean(NaID1_region.flux.value), 5)
            NaID1_sum_err = [i**2 for i in NaID1_region.uncertainty.array]
            NaID1_err = np.sqrt(np.sum(NaID1_sum_err))/len(NaID1_sum_err)
            
            NaID2_mean = np.round(np.mean(NaID2_region.flux.value), 5)
            NaID2_sum_err = [i**2 for i in NaID2_region.uncertainty.array]
            NaID2_err = np.sqrt(np.sum(NaID2_sum_err))/len(NaID2_sum_err)
            
            sigma_D12 = np.sqrt(np.square(NaID1_err) + np.square(NaID2_err))
            
            # Calculating the NaI index 
            
            NaID_index = np.round(((NaID1_mean + NaID2_mean)/F_cont.value), 5)
            
            # Error on NaI index is calculated using error propagation
            
            sigma_NaID_index = np.round((NaID_index*np.sqrt(np.square(sigma_D12/(NaID1_mean + NaID2_mean)) + 
                                                  np.square(F_cont_err/F_cont.value))), 5)
            
            if print_stat:
                print('Using {} higher flux values in each band for the pseudo-cont. calculation'.format(hfv))
                print('----------------------------------------------------------------------------------------------------------------')
                print('Flux in blue cont. is {}'.format(F1_mean))
                print('Flux in red cont. is {}'.format(F2_mean))
                print('Mean cont. flux is {}±{}'.format(F_cont.value, F_cont_err))
                print('NaID1 mean flux is {}±{}'.format(NaID1_mean, NaID1_err))
                print('NaID2 mean flux is {}±{}'.format(NaID2_mean, NaID2_err))
                print('----------------------------------------------------------------------------------------------------------------')
                print('The NaI doublet index is: {}±{}'.format(NaID_index, sigma_NaID_index))
                print('----------------------------------------------------------------------------------------------------------------')
            
            res = [MJD, OBS_DATE, NaID_index, sigma_NaID_index, RV, EXPTIME, PROG_ID] # Creating a list containing the results for this file
            results.append(res) # Appending the res list into the empty results list created at the start of this function
            
        else:
            
            raise ValueError('Instrument type not recognised. Available options are "NARVAL", "HARPS" and "HARPS-N"')
            
    # Saving the results of each Instrument type run in a .txt file with the given file name separated by a space; ' '.
         
    if save_results:
        
        if print_stat:
            print('Saving results in the working directory in file: {}.csv'.format(results_file_name))
            print('----------------------------------------------------------------------------------------------------------------')
            
            if Instrument=='NARVAL':
                
                header = ['HJD', 'I_NaI', 'I_NaI_err']
                
            elif Instrument=='HARPS':
                
                header = ['MJD', 'OBS_DATE', 'I_NaI', 'I_NaI_err', 'RV', 'T_exp', 'SNR', 'RON', 'PROG_ID']
                
            elif Instrument=='HARPS-N':
                
                header = ['MJD', 'OBS_DATE', 'I_NaI', 'I_NaI_err', 'RV', 'T_exp', 'PROG_ID']

        with open('{}.csv'.format(results_file_name), 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow(header)
            for row in results:
                writer.writerow(row)

    return results

## Defining a function to calculate the CaIIH index following Morgenthaler et al. 2012 (2012A&A...540A.138M)

def CaIIH_Index(file_path,
                radial_velocity=9609, 
                degree=4, 
                CaIIH_line=396.847, 
                CaIIH_band=0.04, 
                cont_R_line=400.107,
                cont_R_band=2.0,
                Instrument='NARVAL',
                Stokes_profile=['V'], 
                norm_spec=False,
                plot_fit=False,
                plot_spec=True,
                print_stat=True,
                save_results=False,
                results_file_name=None,
                save_figs=False,
                out_file_path=None,
                ccf_file_path=None):
    
    """
    Calculates the CaIIH index following Morgenthaler A., et al., 2012, A&A, 540, A138. 
    NOTE: The CaIIH line flux is measured within a rectangular bandpass instead of a triangular one following Boisse I., et al., 2009, A&A, 495, 959.
    
    Parameters:
    -----------
    file_path: list, .s format (NARVAL), ADP..._.fits format (HARPS) or s1d_A.fits format (HARPS-N)
    List containng the paths of the spectrum files 
    
    radial_velocity: int, default: 9609 m/s
    Stellar radial velocity along the line-of-sight taken from GAIA DR2 for GJ 436.
    This value is used for doppler shifting the spectra to its rest frame.
    
    degree: int, default: 4
    The degree of the Chebyshev1D polynomial to fit to the continuum for normalisation.
    Normalisation done using Specutils. 
    For more info, 
    see https://specutils.readthedocs.io/en/stable/api/specutils.fitting.fit_generic_continuum.html#specutils.fitting.fit_generic_continuum
    
    CaIIH_line: int, default: 396.847 nm
    CaII H line centre in nm.
    
    CaIIH_band: int, default: 0.04 nm
    Band width (nm) in which to calculate the mean flux.
    
    cont_R_line: int, default: 400.107 nm
    Line centre of the red reference continuum.
    
    cont_R_band: int, default: 2.0 nm
    Band width (nm) in which to calculate the mean continuum flux.
    
    Instrument: str, default: 'NARVAL'
    The instrument from which the data has been collected. Available options are 'NARVAL', 'HARPS' or 'HARPS-N'.
    
    Stokes_profile: str, default: ['V']
    The Stokes profile for the input data. 'V' for per night and 'I' for per sub-exposure per night. Used only when Instrument type is 'NARVAL'
    
    norm_spec: bool, default: False
    Normalizes the spectrum.
    
    plot_fit: bool, default: False
    Plots the continuum fitting normalization processes.
    
    plot_spec: bool, default: True
    Plots the final reduced spectrum.
    
    print_stat: bool, default: True
    Prints the status of each process within the function.
    
    save_results: bool, default: False
    Saves the run results in a .csv format in the working directory
    
    results_file_name: str, default: None
    Name of the file with which to save the results file
    
    save_figs: bool, default: False
    Save the plots in a pdf format in the working directory
    
    out_file_path: list, .out format (NARVAL), default: None
    List containing the paths of the .out files to extract the OBS_HJD. If None, HJD is returned as NaN. Used only when Instrument type is 'NARVAL'
    
    ccf_file_path: list, .fits format (HARPS/HARPS-N), default: None
    List containig the paths of the CCF FITS files to extract the radial velocity. If None, the given radial velocity argument is used for all files for doppler shift corrections
    
    Returns:
    -----------
    NARVAL: HJD of observation, CaIIH index and error on index. 
    HARPS: MJD of observation, Observation date, CaIIH index, error on index, radial velocity, exposure time (s), SNR, ReadOut noise and Program ID
    HARPS-N: MJD of observation, Observation date, CaIIH index, error on index, radial velocity, exposure time (s) and Program ID
    
    All values are type float().
    
    """

    results = [] # Empty list to which the run results will be appended
    
    # Creating a loop to go through each given file_path in the list of file paths
    
    # Using the tqdm function 'log_progress' to provide a neat progress bar in Jupyter Notebook which shows the total number of
    # runs, the run time per iteration and the total run time for all files!
    
    for i in log_progress(range(len(file_path)), desc='Calculating CaIIH Index'):
        
        # Creating a loop for each instrument type.
        
        ## NARVAL

        if Instrument == 'NARVAL':

            if out_file_path != None:

                file = open(out_file_path[i]).readlines() # Opening the .out file and reading each line as a string
                
                string = '   Heliocentric Julian date (UTC) :' # Creating a string variable that matches the string in the .out file
                
                idx = find_string_idx(out_file_path[i], string) # Using the 'find_string_idx' function to find the index of the line that contains the above string. 
                
                HJD = float(file[idx][-14:-1]) # Using the line index found above, the HJD is extracted by indexing just that from the line.

            else:
                if print_stat:
                    print('out_file_path not given as an argument. Returning NaN as HJD instead.')
                    print('----------------------------------------------------------------------------------------------------------------')
                HJD = float('nan')
                
            # Defining column names for pandas to read the file easily
            
            col_names_V = ['Wavelength', 'Intensity', 'Polarized', 'N1', 'N2', 'I_err'] # For Stokes V
            col_names_I = ['Wavelength', 'Intensity', 'I_err'] # For Stokes I
            
            # Reading data using pandas and skipping the first 2 rows

            if Stokes_profile==['V']:
                data_spec = pd.read_fwf(file_path[i], names=col_names_V, skiprows=2) 
            else:
                data_spec = pd.read_fwf(file_path[i], names=col_names_I, skiprows=2)


            # Extracting indivdidual spectral orders using the 'extract_orders' function
            
            if print_stat:
                print('Extracting spectral orders')
                print('----------------------------------------------------------------------------------------------------------------')
                
            orders = extract_orders(data_spec['Wavelength'].values, 
                                    data_spec['Intensity'].values, 
                                    flx_err=data_spec['I_err'].values)
            
            if print_stat:
                print('Total {} spectral orders extracted'.format(len(orders)))
                print('----------------------------------------------------------------------------------------------------------------')
            
            # The CaIIH line is found only within one spectral order, # 57
            
            order_57 = orders[61-57] # The orders begin from # 61 so to get # 57, we index as 61-57.
            
            if print_stat:
                print('The #57 order wavelength read from .s file using pandas is: {}'.format(order_57[0]))
                print('The #57 order intensity read from .s file using pandas is: {}'.format(order_57[1]))
                print('The #57 order intensity error read from .s file using pandas is: {}'.format(order_57[2]))
                print('----------------------------------------------------------------------------------------------------------------')
                
            # The spectra is now doppler shift corrected in the wavelength axis using the stellar radial velocity and the rest wavelength of CaIIH line; delta_lambda = (v/c)*lambda

            shift = ((radial_velocity/ap.constants.c.value)*CaIIH_line)
            shift = (round(shift, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in the .s files!

            wvl = np.round((order_57[0] - shift), 4)
            flx = order_57[1]
            flx_err = order_57[2]
            
            # Creating a spectrum object called 'spec1d' using 'Spectrum1D' from 'specutils'
            # Docs for 'specutils' are here; https://specutils.readthedocs.io/en/stable/ 
            
            # The spectral and flux axes are given units nm and Jy respectively using 'astropy.units'. 
            # The uncertainty has units Jy as well!

            spec1d = Spectrum1D(spectral_axis=wvl*u.nm, 
                                flux=flx*u.Jy, 
                                uncertainty=StdDevUncertainty(flx, unit=u.Jy))
            
            # Printing info
            
            if print_stat:
                print('The doppler shift size using the CaIIH line of 396.847nm is: {}nm'.format(shift))
                print('The spectral order used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 4 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('----------------------------------------------------------------------------------------------------------------')
                
            # Fitting an nth order polynomial to the continuum for normalisation using specutils

            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('----------------------------------------------------------------------------------------------------------------')
                    
                # 'fit_generic_continuum' is a function imported from 'specutils' which fits a given polynomial model to the given spectrum.
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis) # Continuum fit y values are calculated by inputting the spectral axis x values into the polynomial fit equation 
                
                spec_normalized = spec1d / y_cont_fitted # Spectrum is normalised by dividing it with the polynomial fit
                
                # Plots the polynomial fits
                if plot_fit:
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalised Flux')
                    ax1.set_title("Continuum Fitting") 
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('----------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(HJD), format='pdf')

                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, color='blue', label='Re-normalised', alpha=0.6)
                    ax2.plot(spec1d.spectral_axis, spec1d.flux, color='red', label='Pipeline normalised', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalised Flux')
                    ax2.set_title("Continuum Normalized ")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(HJD), format='pdf')

                spec = spec_normalized 

            else:
                spec = spec1d 
            
            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            if plot_spec:
                f, ax = plt.subplots(figsize=(10,4)) 
                ax.plot(spec.spectral_axis, spec.flux)  
                ax.set_xlabel('$\lambda (nm)$')
                ax.set_ylabel("Normalised Flux")
                plt.vlines(CaIIH_line, ymin=0.0, ymax=1.5, linestyles='dotted', colors='green')
                plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=0.0, ymax=3.0, linestyles='--', colors='black', label='CaIIH band width = ({}±{})nm'.format(CaIIH_line, CaIIH_band/2))
                plt.vlines(CaIIH_line+(CaIIH_band/2), ymin=0.0, ymax=3.0, linestyles='--', colors='black')
                plt.vlines(cont_R_line-(cont_R_band/2), ymin=0.0, ymax=3.0, linestyles='--', colors='red', label='Right ref. band width = ({}±{})nm'.format(cont_R_line, cont_R_band/2))
                plt.vlines(cont_R_line+(cont_R_band/2), ymin=0.0, ymax=3.0, linestyles='--', colors='red')
                ax.set_xlim(CaIIH_line-(CaIIH_band/2)-0.05, cont_R_line+(cont_R_band/2)+0.05)
                ax.set_ylim(-0.35, 3.0)
                
                if save_figs:
                        plt.savefig('{}_reduced_spec_plot.pdf'.format(HJD), format='pdf')

                f, ax1 = plt.subplots(figsize=(10,4)) 
                ax1.plot(spec.spectral_axis, spec.flux)
                ax1.set_xlabel('$\lambda (nm)$')
                ax1.set_ylabel("Normalised Flux")
                plt.vlines(CaIIH_line, ymin=0.0, ymax=3.0, linestyles='dotted', colors='green')
                plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=0.0, ymax=3.0, linestyles='--', colors='black', label='CaIIH band width = {}nm'.format(CaIIH_band))
                plt.vlines(CaIIH_line+(CaIIH_band/2), ymin=0.0, ymax=3.0, linestyles='--', colors='black')
                ax1.set_xlim(CaIIH_line-(CaIIH_band/2)-0.1, CaIIH_line+(CaIIH_band/2)+0.1)
                ax1.set_ylim(-0.35, 3.0)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_CaIIH_line_plot.pdf'.format(HJD), format='pdf')

        ## HARPS 

        elif Instrument == 'HARPS':

            # Opening the FITS file using 'astropy.io.fits'
            # NOTE: The format of this FITS file must be ADP which contains the reduced spectrum with the wav, flux and flux_err in three columns
            
            file = fits.open(file_path[i])
            
            if ccf_file_path:
                ccf_file = fits.open(ccf_file_path[i]) # Opening the CCF FITS file to extract the RV
                RV = ccf_file[0].header['HIERARCH ESO DRS CCF RV']*1000 # Radial velocity converted from km/s to m/s
                
            else:
                RV = radial_velocity
            
            #Extracting useful information from the fits file header
            
            MJD = file[0].header['MJD-OBS'] # Modified Julian Date
            EXPTIME = file[0].header['EXPTIME'] # Exposure time in s
            OBS_DATE = file[0].header['DATE-OBS'] # Observation Date
            PROG_ID = file[0].header['PROG_ID'] # Program ID
            SNR = file[0].header['SNR'] # Signal to Noise ratio
            SIGDET = file[0].header['HIERARCH ESO DRS CCD SIGDET']  #CCD Readout Noise [e-]
            CONAD = file[0].header['HIERARCH ESO DRS CCD CONAD'] #CCD conversion factor [e-/ADU]; from e- to ADU
            RON = SIGDET * CONAD #CCD Readout Noise [ADU]
            
            # Defining each wavelength, flux and flux error arrays from the FITS file!
            
            wvl = file[1].data[0][0]/10 # dividing it by 10 to convert the wavelength from Å to nm!
            flx = file[1].data[0][1] # Flux in ADU
            flx_err = file[1].data[0][2]
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
           
            shift = ((radial_velocity/ap.constants.c.value)*CaIIH_line)  
            shift = (round(shift, 3)) # Using only 3 decimal places for the shift value since that's the precision of the wavelength in the .fits files!
            
            # Since the HARPS spectra have their individual spectral orders stitched together, 
            # we do not have to extract them separately as done for NARVAL. Thus for HARPS, the required 
            # region is extracted by slicing the spectrum with the index corresponding to the CaIIH line (left) and cont R (right) obtained using the 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, CaIIH_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, cont_R_line+2)
            
            # If condition for when certain files have NaN as the flux errors; probably for all since the ESO Phase 3 data currently does not provide the flux errors
            
            flx_err_nan = np.isnan(np.sum(flx_err)) # NOTE: This returns true if there is one NaN or all are NaN!
            
            if flx_err_nan:
                if print_stat:
                    print('File contains NaN in flux errors array. Calculating flux error using CCD readout noise: {}'.format(np.round(RON, 4)))
                    print('----------------------------------------------------------------------------------------------------------------')
                # Flux error calculated as photon noise plus CCD readout noise 
                # NOTE: The error calculation depends on a lot of other CCD parameters such as the pixel binning in each CCD
                # array and so on. But for photometric limited measurements, this noise is generally insignificant.
                
                flx_err_ron = [np.sqrt(flux + np.square(RON)) for flux in flx]
                
                if np.isnan(np.sum(flx_err_ron)):
                    if print_stat:
                        print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                        print('----------------------------------------------------------------------------------------------------------------')
                
                # Slicing the data to contain only the region required for the index calculation as explained above and 
                # creating a spectrum class for it.
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err_ron[left_idx:right_idx], unit=u.Jy))
                
            else:
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx], unit=u.Jy))

            if print_stat:
                print('The doppler shift size using CaIIH line of 396.847nm is: {}nm'.format(shift))
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, 
                                                                                                                                                              spec1d.spectral_axis[-1].value))
                print('----------------------------------------------------------------------------------------------------------------')

            # Fitting an nth order polynomial to the continuum for normalisation using specutils
            
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('----------------------------------------------------------------------------------------------------------------')
                
                # 'fit_generic_continuum' is a function imported from 'specutils' which fits a given polynomial model to the given spectrum.
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis) # Continuum fit y values are calculated by inputting the spectral axis x values into the polynomial fit equation 
                spec_normalized = spec1d / y_cont_fitted
                
                spec = spec_normalized # Note the continuum normalized spectrum also has new uncertainty values which are simply the errors divided by this polynomial fit.

                # Plots the polynomial fits
                if plot_fit:
                    f, ax1 = plt.subplots()  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalised Flux')
                    ax1.set_title("Continuum Fitting") 
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('----------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(MJD), format='pdf')

                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, label='Re-Normalized')
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(cont_R_line+(cont_R_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized ")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(MJD), format='pdf')

                spec = spec_normalized # Note the continuum normalised spectrum also has new uncertainty values!

            else:
                spec = spec1d


            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            if plot_spec:
                ax  = plt.subplots()[1]  
                ax.plot(spec.spectral_axis, spec.flux)  
                ax.set_xlabel('$\lambda (nm)$')
                ax.set_ylabel("Normalised Flux")
                plt.vlines(CaIIH_line, ymin=0.0, ymax=2.5, linestyles='dotted', colors='green')
                plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black', label='CaIIH band width = ({}±{})nm'.format(CaIIH_line, CaIIH_band/2))
                plt.vlines(CaIIH_line+(CaIIH_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(cont_R_line-(cont_R_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='red', label='Right ref. band width = ({}±{})nm'.format(cont_R_line, cont_R_band/2))
                plt.vlines(cont_R_line+(cont_R_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='red')
                plt.xlim(CaIIH_line-(CaIIH_band/2)-0.05, cont_R_line+(cont_R_band/2)+0.05)
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_reduced_spec_plot.pdf'.format(MJD), format='pdf')

                ax1  = plt.subplots()[1]  
                ax1.plot(spec.spectral_axis, spec.flux)
                ax1.set_xlabel('$\lambda (nm)$')
                ax1.set_ylabel("Normalised Flux")
                plt.vlines(CaIIH_line, ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='dotted', colors='green')
                plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black', label='CaIIH band width = {}nm'.format(CaIIH_band))
                plt.vlines(CaIIH_line+(CaIIH_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax1.set_xlim(CaIIH_line-(CaIIH_band/2)-0.1, CaIIH_line+(CaIIH_band/2)+0.1)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_CaIIH_line_plot.pdf'.format(MJD), format='pdf')
                
        ## HARPS-N
                
        elif Instrument=='HARPS-N':
            
            # Opening the FITS file using 'astropy.io.fits'
            # NOTE: The format of this FITS file must be s1d which only contains flux array. 
            # The wavelength array is constructed using the starting point (CRVAL1), length of spectral axis (NAXIS1) 
            # and wavelength step (CDELT1) from the FITS file header.
            
            file = fits.open(file_path[i])
            
            if ccf_file_path:
                ccf_file = fits.open(ccf_file_path[i])  # Opening the CCF FITS file to extract the RV
                RV = ccf_file[0].header['HIERARCH TNG DRS CCF RV']*1000 # Radial velocity converted from km/s to m/s
                
            else:
                RV = radial_velocity
            
            #Extracting useful information from the fits file header
            
            MJD = file[0].header['MJD-OBS'] # Modified Julian Date
            EXPTIME = file[0].header['EXPTIME'] # Exposure time in seconds
            OBS_DATE = file[0].header['DATE-OBS'] # Observation Date
            PROG_ID = file[0].header['PROGRAM'] # Program ID
            
            
            flx = file[0].data # Flux in ADU
            wvl = file[0].header['CRVAL1'] + file[0].header['CDELT1']*np.arange(0, file[0].header['NAXIS1']) # constructing the spectral axis using start point, delta and axis length from file header
            wvl = wvl/10 # convert wvl from Å to nm!
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
            
            shift = ((RV/ap.constants.c.value)*CaIIH_line)  
            shift = (round(shift, 3)) 
            
            # Same as the HARPS spectra, the HARPS-N spectra have their individual spectral orders stitched together and 
            # we do not have to extract them separately as done for NARVAL. Thus, the required region is extracted by slicing
            # the spectrum with the index corresponding to the left and right continuum obtained using the 
            # 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, CaIIH_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, cont_R_line+2)
            
            flx_err = [np.sqrt(flux) for flux in flx] # Using only photon noise as flx_err approx since no RON info available!
            
            # Slicing the data to contain only the region required for the index calculation as explained above and creating 
            # a spectrum class for it
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx], unit=u.Jy))
            
            # Fitting an nth order polynomial to the continuum for normalisation using specutils
            
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('----------------------------------------------------------------------------------------------------------------')
                
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('----------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis)
                spec_normalized = spec1d / y_cont_fitted
                
                # Plots the polynomial fits
                if plot_fit:
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalized Flux')
                    ax1.set_title("Continuum Fitting")
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('----------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(MJD), format='pdf')
                    
                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, color='blue', label='Re-Normalized', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized ")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(MJD), format='pdf')
                    
                spec = spec_normalized # Note the continuum normalized spectrum also has new uncertainty values!
                
            else:
                
                spec = spec1d
                
            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            if plot_spec:
                ax  = plt.subplots()[1]  
                ax.plot(spec.spectral_axis, spec.flux)  
                ax.set_xlabel('$\lambda (nm)$')
                ax.set_ylabel("Normalised Flux")
                plt.vlines(CaIIH_line, ymin=0.0, ymax=2.5, linestyles='dotted', colors='green')
                plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=-1.0, ymax=4, linestyles='--', colors='black', label='CaIIH band width = ({}±{})nm'.format(CaIIH_line, CaIIH_band/2))
                plt.vlines(CaIIH_line+(CaIIH_band/2), ymin=-1.0, ymax=4, linestyles='--', colors='black')
                plt.vlines(cont_R_line-(cont_R_band/2), ymin=-1.0, ymax=4, linestyles='--', colors='red', label='Right ref. band width = ({}±{})nm'.format(cont_R_line, cont_R_band/2))
                plt.vlines(cont_R_line+(cont_R_band/2), ymin=-1.0, ymax=4, linestyles='--', colors='red')
                plt.xlim(CaIIH_line-(CaIIH_band/2)-0.05, cont_R_line+(cont_R_band/2)+0.05)
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_reduced_spec_plot.pdf'.format(MJD), format='pdf')

                ax1  = plt.subplots()[1]  
                ax1.plot(spec.spectral_axis, spec.flux)
                ax1.set_xlabel('$\lambda (nm)$')
                ax1.set_ylabel("Normalised Flux")
                plt.vlines(CaIIH_line, ymin=0.0, ymax=2.5, linestyles='dotted', colors='green')
                plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=-1, ymax=4, linestyles='--', colors='black', label='CaIIH band width = {}nm'.format(CaIIH_band))
                plt.vlines(CaIIH_line+(CaIIH_band/2), ymin=-1, ymax=4, linestyles='--', colors='black')
                ax1.set_xlim(CaIIH_line-(CaIIH_band/2)-0.1, CaIIH_line+(CaIIH_band/2)+0.1)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                        plt.savefig('{}_CaIIH_line_plot.pdf'.format(MJD), format='pdf')

        else:
            raise ValueError('Instrument type not recognisable. Available options are "NARVAL", "HARPS" and "HARPS-N"')
            
        # Now we have the final spectrum to work with as a variable, 'spec'!
        
        # The two regions required for CaIIH index calculation are extracted from 'spec' using the 'extract region' function from 'specutils'. 
        # The function uses another function called 'SpectralRegion' as one of its arguments which defines the region to be extracted done so using the line and line bandwidth values; i.e. left end of region would be 'line - bandwidth/2' and right end would be 'line + bandwidth/2'.
        # Note: These values must have the same units as the spec wavelength axis.


        # Extracting the CaIIH line region using the given bandwidth 'CaIIH_band'
        F_CaIIH_region = extract_region(spec, region=SpectralRegion((CaIIH_line-(CaIIH_band/2))*u.nm, (CaIIH_line+(CaIIH_band/2))*u.nm))
        F_CaIIH_mean = np.round(np.mean(F_CaIIH_region.flux.value), 5) # Calculating mean of the flux within this bandwidth
        
        # Calculating the standard error on the mean flux calculated above.
        F_CaIIH_sum_err = [i**2 for i in F_CaIIH_region.uncertainty.array]
        F_CaIIH_mean_err = np.round((np.sqrt(np.sum(F_CaIIH_sum_err))/len(F_CaIIH_sum_err)), 5)

        
        # Doing the same for the cont R region!
        cont_R_region = extract_region(spec, region=SpectralRegion((cont_R_line-(cont_R_band/2))*u.nm, (cont_R_line+(cont_R_band/2))*u.nm))
        cont_R_mean = np.round(np.mean(cont_R_region.flux.value), 5)
        cont_R_sum_err = [i**2 for i in cont_R_region.uncertainty.array]
        cont_R_mean_err = np.round((np.sqrt(np.sum(cont_R_sum_err))/len(cont_R_sum_err)), 5)


        # Calculating the index from the mean fluxes calculated above
        CaIIH_from_mean = np.round((F_CaIIH_mean/cont_R_mean), 5)
        
        # Error on this index is calculated using error propagation!
        sigma_CaIIH_from_mean = np.round((CaIIH_from_mean*np.sqrt(np.square(F_CaIIH_mean_err/F_CaIIH_mean) + np.square(cont_R_mean_err/cont_R_mean))), 5)
        
        if print_stat:
            print('CaIIH region used ranges from {}nm to {}nm:'.format(F_CaIIH_region.spectral_axis[0].value, 
                                                                 F_CaIIH_region.spectral_axis[-1].value))
            print('cont R region used ranges from {}nm to {}nm:'.format(cont_R_region.spectral_axis[0].value, 
                                                                 cont_R_region.spectral_axis[-1].value))
            print('----------------------------------------------------------------------------------------------------------------')
            print('Mean of flux points in CaIIH: {}±{}'.format(F_CaIIH_mean, F_CaIIH_mean_err))
            print('Mean of flux points in cont R: {}±{}'.format(cont_R_mean, cont_R_mean_err))
            print('----------------------------------------------------------------------------------------------------------------')
            print('Index from mean of flux points in each band: {}±{}'.format(CaIIH_from_mean, sigma_CaIIH_from_mean))
            print('----------------------------------------------------------------------------------------------------------------')
            
        if Instrument=='NARVAL':
            res = [HJD, CaIIH_from_mean, sigma_CaIIH_from_mean] # Creating results list 'res' containing the calculated parameters and appending this list to the 'results' empty list created at the start of this function!
            results.append(res)
        
        elif Instrument=='HARPS':
            res = [MJD, OBS_DATE, CaIIH_from_mean, sigma_CaIIH_from_mean, RV, EXPTIME, SNR, RON, PROG_ID]
            results.append(res)
            
        elif Instrument=='HARPS-N':
            res = [MJD, OBS_DATE, CaIIH_from_mean, sigma_CaIIH_from_mean, RV, EXPTIME, PROG_ID]
            results.append(res)
            
    # Saving the results in a csv file format  
    if save_results:
        
        if print_stat:
            print('Saving results in the working directory in file: {}.csv'.format(results_file_name))
            print('----------------------------------------------------------------------------------------------------------------')
            
            if Instrument=='NARVAL':
                
                header = ['HJD', 'I_CaIIH', 'I_CaIIH_err']
                
            elif Instrument=='HARPS':
                
                header = ['MJD', 'OBS_DATE', 'I_CaIIH', 'I_CaIIH_err', 'RV', 'T_exp', 'SNR', 'RON', 'PROG_ID']
                
            elif Instrument=='HARPS-N':
                
                header = ['MJD', 'OBS_DATE', 'I_CaIIH', 'I_CaIIH_err', 'RV', 'T_exp', 'PROG_ID']

        with open('{}.csv'.format(results_file_name), 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow(header)
            for row in results:
                writer.writerow(row)  
            
    return results
