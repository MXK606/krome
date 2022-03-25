#!/usr/bin/env python
# coding: utf-8

"""
index_calc.py: This python module contains the CaIIH, NaI, and Hα (CaI within it) activity index calculation functions.

"""

__author__ = "Mukul Kumar"
__email__ = "Mukul.k@uaeu.ac.ae, MXK606@alumni.bham.ac.uk"
__date__ = "10-03-2022"
__version__ = "1.8.1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import warnings
from tqdm.notebook import tqdm as log_progress
import astropy.units as u
import astropy as ap
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from specutils.manipulation import extract_region
from astropy.modeling.polynomial import Chebyshev1D
from astropy.nddata import StdDevUncertainty
from astropy.io import fits
from krome.spec_analysis import find_nearest, read_data, calc_ind
    
## Defining a function for calculating the H alpha index following Boisse et al. 2009 (2009A&A...495..959B)

def H_alpha_index(file_path,
                  radial_velocity,
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
                  norm_spec=False,
                  plot_fit=False, 
                  plot_spec=True,
                  print_stat=True,
                  save_results=False, 
                  results_file_name=None,
                  save_figs=False,
                  save_figs_name=None,
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
    
    radial_velocity: int
    Stellar radial velocity along the line-of-sight. This value is used for doppler shifting the spectra to its rest frame.
    
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
    
    save_figs_name: str, default=None
    Name with which to save the figures. NOTE: This should ideally be the observation date of the given spectrum.
    
    out_file_path: list, .out format (NARVAL), default: None
    List containing the paths of the .out files to extract the OBS_HJD. If None, HJD is returned as NaN. Used only when Instrument type is 'NARVAL'
    
    ccf_file_path: list, .fits format (HARPS/HARPS-N), default: None
    List containig the paths of the CCF FITS files to extract the radial velocity. If None, the given radial velocity argument is used for all files for doppler shift corrections
    
    CaI_index: bool, default=True
    Calculates the activity insensitive CaI index as well. If False, NaN values are returned instead.
    
    Returns:
    -----------
    NARVAL: HJD, RA, DEC, AIRMASS, Exposure time[s], No. of exposures, GAIN [e-/ADU], ReadOut Noise [e-], V_mag, T_eff[K], RV[m/s], H alpha index, error on H alpha index, CaI index and error on CaI index.
    HARPS: BJD, RA, DEC, AIRMASS, Exposure time[s], Barycentric RV[km/s], OBS_DATE, Program ID, SNR, CCD Readout Noise[e-], CCD conv factor[e-/ADU], ReadOut Noise[ADU], RV[m/s], H alpha index, error on H alpha index, CaI index, error on CaI index
    HARPS-N: BJD, RA, DEC, AIRMASS, Exposure time[s], OBS_DATE, Program ID', RV[m/s], H alpha index, error on H alpha index, CaI index and error on CaI index
    
    All values are type float() given inside a list.
    
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
                
                # Using read_data from krome.spec_analysis to extract useful object parameters and all individual spectral orders
                
                obj_params, orders = read_data(file_path=file_path[i],
                                               out_file_path=out_file_path[i],
                                               Instrument=Instrument,
                                               print_stat=print_stat,
                                               show_plots=False)
                
                obj_params['RV'] = radial_velocity # setting radial_velocity as part of the obj_params dictionary for continuity 
                
            else:
                
                orders = read_data(file_path=file_path[i],
                                   Instrument=Instrument,
                                   print_stat=print_stat,
                                   out_file_path=None,
                                   show_plots=False)
                
                if print_stat:
                    print('"out_file_path" not given as an argument. Run will only return the indices and their errros instead.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                

            if print_stat:
                print('Total {} spectral orders extracted'.format(len(orders)))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
            
            order_34 = orders[61-34] # The orders begin from # 61 so to get # 34, we index as 61-34.
            
            if print_stat:
                print('The #34 order wavelength read from .s file using pandas is: {}'.format(order_34[0].values))
                print('The #34 order intensity read from .s file using pandas is: {}'.format(order_34[1].values))
                print('The #34 order intensity error read from .s file using pandas is: {}'.format(order_34[2].values))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
            
            # The spectra is now doppler shift corrected in the wavelength axis using the stellar radial velocity and the rest wavelength of H alpha line; delta_lambda = (v/c)*lambda
            
            shift = ((radial_velocity/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in the .s files!
            
            wvl = np.round((order_34[0].values - shift), 4) # Subtracting the calculated doppler shift value from the wavelength axis since the stellar radial velocity is positive. If the stellar RV is negative, the shift value will be added instead.
            flx = order_34[1].values # Indexing flux array from order_34
            flx_err = order_34[2].values # Indexing flux_err array from order_34
            
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
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral order used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 4 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            # Fitting an nth order polynomial to the continuum for normalisation using specutils
            
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                # 'fit_generic_continuum' is a function imported from 'specutils' which fits a given polynomial model to the given spectrum.
                
                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis) # Continuum fit y values are calculated by inputting the spectral axis x values into the polynomial fit equation 
                
                spec_normalized = spec1d / y_cont_fitted # Spectrum is normalised by diving it with the polynomial fit
                
                # Plots the polynomial fits
                if plot_fit:
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Normalized Flux')
                    ax1.set_title("Continuum Fitting")
                    plt.tight_layout()
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')
                    
                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, color='blue', label='Re-Normalized', alpha=0.6)
                    ax2.plot(spec1d.spectral_axis, spec1d.flux, color='red', label='Pipeline Normalized', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec1d.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec1d.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized ")
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(save_figs_name), format='pdf')
                        
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
                
                if CaI_index:
                    plt.vlines(CaI_line-(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='black', label='CaI {}±{}nm'.format(CaI_line, CaI_band/2))
                    plt.vlines(CaI_line+(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='black')
                
                ax.set_xlim(F1_line-1.1, F2_line+1.1)
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                plt.minorticks_on()
                ax.tick_params(direction='in', which='both')
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_reduced_spec_plot.pdf'.format(save_figs_name), format='pdf')
                
                # Plots the zoomed in regions around the H alpha line.
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
                    plt.savefig('{}_H_alpha_line_plot.pdf'.format(save_figs_name), format='pdf')
                        
                        
                if CaI_index:
                    # Plots the zoomed in regions around the CaI line.
                    f, ax2  = plt.subplots(figsize=(10,4))
                    ax2.plot(spec.spectral_axis, spec.flux)
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel("Normalized Flux")
                    plt.vlines(CaI_line, ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='green')
                    plt.vlines(CaI_line-(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='CaI band width = {}nm'.format(CaI_band))
                    plt.vlines(CaI_line+(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlim(CaI_line-(CaI_band/2)-0.1, CaI_line+(CaI_band/2)+0.1)
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_CaI_line_plot.pdf'.format(save_figs_name), format='pdf')
                
        # HARPS
        
        elif Instrument == 'HARPS':
            
            # Opening the FITS file using 'astropy.io.fits' and extracting useful object parameters and spectrum using read_data from krome.spec_analysis
            # NOTE: The format of this FITS file must be ADP which contains the reduced spectrum with the wav, flux and flux_err in three columns
            
            if ccf_file_path != None:
                obj_params, spec = read_data(file_path=file_path[i],
                                             ccf_file_path=ccf_file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
            else:
                obj_params, spec = read_data(file_path=file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
                
                obj_params['RV'] = radial_velocity # setting obj_params['RV'] to the given radial_velocity argument!
                
            # Assigning appropriate variables from spec individually!
            wvl = spec[0] # nm
            flx = spec[1] # ADU
            flx_err = spec[2]
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
           
            shift = ((obj_params['RV']/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 3)) # Using only 3 decimal places for the shift value since that's the precision of the wavelength in the .fits files!
            
            # Since the HARPS spectra have their individual spectral orders stitched together, we do not have to extract them separately as done for NARVAL. Thus for HARPS, the required region is extracted by slicing the spectrum with the index corresponding to the left and right continuum obtained using the 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            # If condition for when certain files have NaN as the flux errors; probably for all since the ESO Phase 3 data currently does not provide the flux errors
            
            flx_err_nan = np.isnan(np.sum(flx_err)) # NOTE: This returns true if there is one NaN or all are NaN!
            
            if flx_err_nan:
                if print_stat:
                    print('File contains NaN in flux errors array. Calculating flux error using CCD readout noise: {}'.format(np.round(obj_params['RON'], 4)))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                # Flux error calculated as photon noise plus CCD readout noise 
                # NOTE: The error calculation depends on a lot of other CCD parameters such as the pixel binning in each CCD
                # array and so on. But for photometric limited measurements, this noise is generally insignificant.
                
                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err_ron = [np.sqrt(flux + np.square(obj_params['RON'])) for flux in flx]
                
                if np.isnan(np.sum(flx_err_ron)):
                    if print_stat:
                        print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
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
                print('The doppler shift size using RV {} m/s and the H alpha line of 656.2808nm is: {}nm'.format(obj_params['RV'], shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                # 'fit_generic_continuum' is a function imported from 'specutils' which fits a given polynomial model to the given spectrum.
                
                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
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
                    plt.tight_layout()
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')
                    
                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, label='Re-Normalized')
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized ")
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(save_figs_name), format='pdf')
                
            else:
                spec = spec1d
                
                
            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            if plot_spec:
                f, ax  = plt.subplots(figsize=(10,4)) 
                ax.plot(spec.spectral_axis, spec.flux, '-k')  
                ax.set_xlabel('$\lambda (nm)$')
                if norm_spec:
                    ax.set_ylabel("Normalized Flux")
                else:
                    ax.set_ylabel("Flux (adu)")
                plt.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Hα {}±{}nm'.format(H_alpha_line, H_alpha_band/2))
                plt.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='blue', label='Blue cont. {}±{}nm'.format(F1_line, F1_band/2))
                plt.vlines(F1_line+(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='blue')
                plt.vlines(F2_line-(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='red', label='Red cont. {}±{}nm'.format(F2_line, F2_band/2))
                plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='red')
                
                if CaI_index:
                    plt.vlines(CaI_line-(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='black', label='CaI {}±{}nm'.format(CaI_line, CaI_band/2))
                    plt.vlines(CaI_line+(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='black')
                
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                plt.minorticks_on()
                ax.tick_params(direction='in', which='both')
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_reduced_spec_plot.pdf'.format(save_figs_name), format='pdf')
                
                f, ax1  = plt.subplots(figsize=(10,4)) 
                ax1.plot(spec.spectral_axis, spec.flux)
                ax1.set_xlabel('$\lambda (nm)$')
                if norm_spec:
                    ax1.set_ylabel("Normalized Flux")
                else:
                    ax1.set_ylabel("Flux (adu)")
                plt.vlines(H_alpha_line, ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='green')
                plt.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Hα band width = {}nm'.format(H_alpha_band))
                plt.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax1.set_xlim(H_alpha_line-(H_alpha_band/2)-0.1, H_alpha_line+(H_alpha_band/2)+0.1)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_H_alpha_line_plot.pdf'.format(save_figs_name), format='pdf')
                        
                if CaI_index:
                    # Plots the zoomed in regions around the CaI line.
                    f, ax2  = plt.subplots(figsize=(10,4))
                    ax2.plot(spec.spectral_axis, spec.flux)
                    ax2.set_xlabel('$\lambda (nm)$')
                    if norm_spec:
                        ax2.set_ylabel("Normalized Flux")
                    else:
                        ax2.set_ylabel("Flux (adu)")
                    plt.vlines(CaI_line, ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='green')
                    plt.vlines(CaI_line-(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='CaI band width = {}nm'.format(CaI_band))
                    plt.vlines(CaI_line+(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlim(CaI_line-(CaI_band/2)-0.1, CaI_line+(CaI_band/2)+0.1)
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_CaI_line_plot.pdf'.format(save_figs_name), format='pdf')
                
        elif Instrument=='HARPS-N':
            
            # Opening the FITS file using 'astropy.io.fits' and extracting useful object parameters and spectrum using read_data from krome.spec_analysis
            # NOTE: The format of this FITS file must be s1d which only contains flux array. 
            # The wavelength array is constructed using the starting point (CRVAL1), length of spectral axis (NAXIS1) 
            # and wavelength step (CDELT1) from the FITS file header.
            
            if ccf_file_path != None:
                obj_params, spec = read_data(file_path=file_path[i],
                                             ccf_file_path=ccf_file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
            else:
                obj_params, spec = read_data(file_path=file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
                
                obj_params['RV'] = radial_velocity # setting obj_params['RV'] to the given radial_velocity argument!
                
            # Assigning appropriate variables from spec individually!
            wvl = spec[0] # nm
            flx = spec[1] # ADU
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
            
            shift = ((obj_params['RV']/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 3)) 
            
            # Same as the HARPS spectra, the HARPS-N spectra have their individual spectral orders stitched together and 
            # we do not have to extract them separately as done for NARVAL. Thus, the required region is extracted by slicing
            # the spectrum with the index corresponding to the left and right continuum obtained using the 
            # 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err = [np.sqrt(flux) for flux in flx] # Using only photon noise as flx_err approx since no RON info available!
            
            # Slicing the data to contain only the region required for the index calculation as explained above and creating 
            # a spectrum class for it
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx], unit=u.Jy))
            
            if print_stat:
                print('The doppler shift size using RV {} m/s and the H alpha line of 656.2808nm is: {}nm'.format(obj_params['RV'], shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                # 'fit_generic_continuum' is a function imported from 'specutils' which fits a given polynomial model to the given spectrum.
                
                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis)
                spec_normalized = spec1d / y_cont_fitted
                
                # Plots the polynomial fits
                if plot_fit:
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Flux (adu)')
                    ax1.set_title("Continuum Fitting")
                    plt.tight_layout()
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')
                    
                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, color='blue', label='Re-Normalized', alpha=0.6)
    #                 ax2.plot(spec1d.spectral_axis, spec1d.flux, color='red', label='Pipeline Normalized', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized ")
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(save_figs_name), format='pdf')
                    
                spec = spec_normalized # Note the continuum normalized spectrum also has new uncertainty values!
                
            else:
                
                spec = spec1d
            
            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            if plot_spec:
                f, ax  = plt.subplots(figsize=(10,4))  
                ax.plot(spec.spectral_axis, spec.flux, '-k')  
                ax.set_xlabel('$\lambda (nm)$')
                if norm_spec:
                    ax.set_ylabel("Normalized Flux")
                else:
                    ax.set_ylabel("Flux (adu)")
                plt.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Hα {}±{}nm'.format(H_alpha_line, H_alpha_band/2))
                plt.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='blue', label='Blue cont. {}±{}nm'.format(F1_line, F1_band/2))
                plt.vlines(F1_line+(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='blue')
                plt.vlines(F2_line-(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='red', label='Red cont. {}±{}nm'.format(F2_line, F2_band/2))
                plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='red')
                
                if CaI_index:
                    plt.vlines(CaI_line-(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='black', label='CaI {}±{}nm'.format(CaI_line, CaI_band/2))
                    plt.vlines(CaI_line+(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='dashdot', colors='black')
                
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                plt.minorticks_on()
                ax.tick_params(direction='in', which='both')
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_reduced_spec_plot.pdf'.format(save_figs_name), format='pdf')
                
                f, ax1  = plt.subplots(figsize=(10,4))
                ax1.plot(spec.spectral_axis, spec.flux)
                ax1.set_xlabel('$\lambda (nm)$')
                if norm_spec:
                    ax1.set_ylabel("Normalized Flux")
                else:
                    ax1.set_ylabel("Flux (adu)")
                plt.vlines(H_alpha_line, ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='green')
                plt.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Hα band width = {}nm'.format(H_alpha_band))
                plt.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax1.set_xlim(H_alpha_line-(H_alpha_band/2)-0.1, H_alpha_line+(H_alpha_band/2)+0.1)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_H_alpha_line_plot.pdf'.format(save_figs_name), format='pdf')
                        
                if CaI_index:
                    # Plots the zoomed in regions around the CaI line.
                    f, ax2  = plt.subplots()
                    ax2.plot(spec.spectral_axis, spec.flux)
                    ax2.set_xlabel('$\lambda (nm)$')
                    if norm_spec:
                        ax2.set_ylabel("Normalized Flux")
                    else:
                        ax2.set_ylabel("Flux (adu)")
                    plt.vlines(CaI_line, ymin=0, ymax=max(spec.flux.value), linestyles='dotted', colors='green')
                    plt.vlines(CaI_line-(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='CaI band width = {}nm'.format(CaI_band))
                    plt.vlines(CaI_line+(CaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlim(CaI_line-(CaI_band/2)-0.1, CaI_line+(CaI_band/2)+0.1)
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_CaI_line_plot.pdf'.format(save_figs_name), format='pdf')
                    
        else:
            raise ValueError('Instrument type not recognised. Available options are "NARVAL", "HARPS" and "HARPS-N"')
            
        # Now we have the spectrum to work with as a variable, 'spec'!
        
        # The three regions required for H alpha index calculation are extracted from 'spec' using the 'extract region' function from 'specutils'. 
        # The function uses another function called 'SpectralRegion' as one of its arguments which defines the region to be extracted done so using the line and line bandwidth values; i.e. left end of region would be 'line - bandwidth/2' and right end would be 'line + bandwidth/2'.
        # Note: These values must have the same units as the spec wavelength axis.
        
        F_H_alpha_region = extract_region(spec, region=SpectralRegion((H_alpha_line-(H_alpha_band/2))*u.nm, (H_alpha_line+(H_alpha_band/2))*u.nm))
        F1_region = extract_region(spec, region=SpectralRegion((F1_line-(F1_band/2))*u.nm, (F1_line+(F1_band/2))*u.nm))
        F2_region = extract_region(spec, region=SpectralRegion((F2_line-(F2_band/2))*u.nm, (F2_line+(F2_band/2))*u.nm))
        
        if CaI_index:
            F_CaI_region = extract_region(spec, region=SpectralRegion((CaI_line-(CaI_band/2))*u.nm, (CaI_line+(CaI_band/2))*u.nm))
            regions = [F_H_alpha_region, F1_region, F2_region, F_CaI_region]
        else:
            regions = [F_H_alpha_region, F1_region, F2_region]
            
        # The indices are calculated using the 'calc_ind' function from krome.spec_analysis by inputting the extracted regions as shown
        
        I_Ha, I_Ha_err, I_CaI, I_CaI_err = calc_ind(regions=regions,
                                                    index_name='HaI',
                                                    print_stat=print_stat,
                                                    CaI_index=CaI_index)
            

        if Instrument=='NARVAL':
            if out_file_path != None:
                header = ['HJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'NUM_EXP', 'GAIN', 'RON', 'V_mag', 'T_eff', 'RV', 'I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err']
                res = list(obj_params.values()) + [I_Ha, I_Ha_err, I_CaI, I_CaI_err] # Creating results list 'res' containing the calculated parameters and appending this list to the 'results' empty list created at the start of this function!
                results.append(res)
            else:
                header = ['I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err']
                res = [I_Ha, I_Ha_err, I_CaI, I_CaI_err]
                results.append(res)
        
        elif Instrument=='HARPS':
            header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'BERV', 'OBS_DATE', 'PROG_ID', 'SNR', 'SIGDET', 'CONAD', 'RON', 'RV', 'I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err']
            res = list(obj_params.values()) + [I_Ha, I_Ha_err, I_CaI, I_CaI_err]
            results.append(res)
            
        elif Instrument=='HARPS-N':
            header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'OBS_DATE', 'PROG_ID', 'RV', 'I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err']
            res = list(obj_params.values()) + [I_Ha, I_Ha_err, I_CaI, I_CaI_err]
            results.append(res)
                
    
    # Saving the results in a csv file format  
    if save_results:
        
        if print_stat:
            print('Saving results in the working directory in file: {}.csv'.format(results_file_name))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')

        with open('{}.csv'.format(results_file_name), 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow(header)
            for row in results:
                writer.writerow(row)  
            
    return results

## Defining a function to calculate the NaI index following Rodrigo F. Díaz et al. 2007 (2007MNRAS.378.1007D)

def NaI_index(file_path,
              radial_velocity,
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
              norm_spec=False,
              plot_fit=False,
              plot_spec=True,
              print_stat=True,
              save_results=False,
              results_file_name=None,
              save_figs=False,
              save_figs_name=None,
              out_file_path=None,
              ccf_file_path=None):
    
    """
    
    This function calculates the NaI doublet index following the method proposed in Rodrigo F. Díaz et al. 2007.
    
    Parameters:
    -----------
    
    file_path: list, .s format (NARVAL), ADP..._.fits format (HARPS) or s1d_A.fits format (HARPS-N)
    List containing paths of the spectrum files
    
    radial_velocity: int
    Stellar radial velocity along the line-of-sight. This value is used for doppler shifting the spectra to its rest frame.
    
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
    NOTE: If you'd like to use all of the flux points within the bandwidth, set this parameter to None.
    
    Instrument: str, default: 'NARVAL'
    The instrument from which the data has been collected. Input takes arguments 'NARVAL', 'HARPS' or 'HARPS-N'.
    
    norm_spec: bool, default: False
    Normalizes ths spectrum. NOTE: This argument also accepts str type of 'scale' and 'poly1dfit' to normalize the spectrum by either scaling it down
    to maximum flux of 1.0,  or, by fitting the continuum with a line. But these are ONLY used for Instrument types 'HARPS' & 'HARPS-N'
    
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
    
    save_figs_name: str, default=None
    Name with which to save the figures. NOTE: This should ideally be the observation date of the given spectrum.
    
    out_file_path: list, .out format (NARVAL), default: None
    List containing paths of the .out files used to extract OBS_HJD.
    
    ccf_file_path: list, .fits format (HARPS/HARPS-N), default: None
    List containing paths of the CCF FITS files used to extract the radial velocity. If None, the given radial velocity arg is used for all files
    
    Returns:
    -----------
    NARVAL: HJD, RA, DEC, AIRMASS, Exposure time[s], No. of exposures, GAIN [e-/ADU], ReadOut Noise [e-], V_mag, T_eff[K], RV[m/s], NaI index and error on NaI index
    HARPS: BJD, RA, DEC, AIRMASS, Exposure time[s], Barycentric RV[km/s], OBS_DATE, Program ID, SNR, CCD Readout Noise[e-], CCD conv factor[e-/ADU], ReadOut Noise[ADU], RV[m/s], NaI index and error on NaI index
    HARPS-N: BJD, RA, DEC, AIRMASS, Exposure time[s], OBS_DATE, Program ID', RV[m/s], NaI index and error on NaI index
    
    All values are type float() given inside a list.
    
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
                
                # Using read_data from krome.spec_analysis to extract useful object parameters and all individual spectral orders
                
                obj_params, orders = read_data(file_path=file_path[i],
                                               out_file_path=out_file_path[i],
                                               Instrument=Instrument,
                                               print_stat=print_stat,
                                               show_plots=False)
                
                obj_params['RV'] = radial_velocity
                
            else:
                
                orders = read_data(file_path=file_path[i],
                                   Instrument=Instrument,
                                   print_stat=print_stat,
                                   out_file_path=None,
                                   show_plots=False)
                
                if print_stat:
                    print('"out_file_path" not given as an argument. Run will only return the indices and their errros instead.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            if print_stat:
                print('Total {} spectral orders extracted'.format(len(orders)))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            ord_39 = orders[61-39] # order 39 contains the F1 line
            ord_38 = orders[61-38] # Both order 39 and 38 contain the D1 and D2 lines but only order 38 is used since it has a higher SNR; (see .out file)
            ord_37 = orders[61-37] # order 37 contains the F2 line
            
            if print_stat:
                print('Using orders #39, #38 and #37 for Index calculation')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
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
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The doppler shift size using RV {} m/s and the NaID1 line of 588.995nm is: {}nm'.format(radial_velocity, shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            # Fitting the continuum for each order separately using 'specutils'
        
            if norm_spec:
                if print_stat:
                    print('Normalising the spectras by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
                # First order
                     
                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    g1_fit = fit_generic_continuum(spec1, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g1_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
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
                    plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='--', colors='black', label='Blue cont. region')
                    plt.vlines(F1_line+(F1_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized First Order")  
                    plt.legend()
                    
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_F1_plot.pdf'.format(save_figs_name), format='pdf')
                          
                # Second order
                
                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    g2_fit = fit_generic_continuum(spec2, model=Chebyshev1D(degree))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g2_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
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
                    plt.vlines(NaID1-1.0, ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black', label='NaID lines region')
                    plt.vlines(NaID2+1.0, ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized Second Order")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_fit_F2_plot.pdf'.format(save_figs_name), format='pdf')
                          
                # Third order
                
                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    g3_fit = fit_generic_continuum(spec3, model=Chebyshev1D(degree))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g3_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
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
                    plt.vlines(F2_line-(F2_band/2), ymin=0, ymax=max(spec3.flux.value), linestyles='--', colors='black', label='F2 region')
                    plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec3.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized Third Order")  
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_fit_F3_plot.pdf'.format(save_figs_name), format='pdf')
                
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
            
            regions = [NaID1_region, NaID2_region, F1_region, F2_region]
            
            # Calculating the index using 'calc_ind' from krome.spec_analysis
            
            I_NaI, I_NaI_err, F1_mean, F2_mean = calc_ind(regions=regions,
                                                          index_name='NaI',
                                                          print_stat=print_stat,
                                                          hfv=hfv)
            # Plotting the pseudo-continuum as the linear interpolation of the values in each red and blue cont. window!
            
            if plot_spec:
                
                x = [F1_line, F2_line]
                y = [F1_mean.value, F2_mean.value]
                
                f, ax  = plt.subplots(figsize=(10,4)) 
                ax.plot(spec1.spectral_axis, spec1.flux, color='red', label='#39', alpha=0.5)
                ax.plot(spec2.spectral_axis, spec2.flux, color='blue', label='#38', alpha=0.5)
                ax.plot(spec3.spectral_axis, spec3.flux, color='green', label='#37', alpha=0.5)
                ax.plot(x, y, 'ok--', label='pseudo-continuum')
                ax.set_xlabel('$\lambda (nm)$')
                ax.set_ylabel("Normalized Flux")
                ax.set_title('Overplotting 3 orders around NaI D lines')
                plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='dotted', colors='blue', label='Blue cont. {}±{}'.format(F1_line, F1_band/2))
                plt.vlines(F1_line+(F1_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='dotted', colors='blue')
                plt.vlines(F2_line-(F2_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='dashdot', colors='red', label='Red cont. {}±{}'.format(F2_line, F2_band/2))
                plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='dashdot', colors='red')
                plt.axhline(1.0, ls='--', c='gray')
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    if print_stat:
                        print('Saving plots as PDFs in the working directory')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    plt.savefig('{}_reduced_spec_plot.pdf'.format(save_figs_name), format='pdf')
                        
                f, ax1  = plt.subplots(figsize=(10,4))
                ax1.plot(spec2.spectral_axis, spec2.flux, color='blue', label='#38')
                ax1.set_xlabel('$\lambda (nm)$')
                ax1.set_ylabel("Normalized Flux")
                plt.vlines(NaID1, ymin=0, ymax=max(spec2.flux.value), linestyles='dotted', colors='red', label='D1')
                plt.vlines(NaID2, ymin=0, ymax=max(spec2.flux.value), linestyles='dotted', colors='blue', label='D2')
                plt.vlines(NaID1-(NaI_band/2), ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black', label='D1,D2 band width = {}nm'.format(NaI_band))
                plt.vlines(NaID1+(NaI_band/2), ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black')
                plt.vlines(NaID2-(NaI_band/2), ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black')
                plt.vlines(NaID2+(NaI_band/2), ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black')
                ax1.set_xlim(NaID2-(NaI_band/2)-0.2, NaID1+(NaI_band/2)+0.2)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_NaID1D2_lines_plot.pdf'.format(save_figs_name), format='pdf')
                
            
            if out_file_path != None:
                header = ['HJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'NUM_EXP', 'GAIN', 'RON', 'V_mag', 'T_eff', 'RV', 'I_NaI', 'I_NaI_err']
                res = list(obj_params.values()) + [I_NaI, I_NaI_err] # Creating results list 'res' containing the calculated parameters and appending this list to the 'results' empty list created at the start of this function!
                results.append(res)
            else:
                header = ['I_NaI', 'I_NaI_err']
                res = [I_NaI, I_NaI_err]
                results.append(res)
                
        ## HARPS
                
        elif Instrument=='HARPS':
            
            # Opening the FITS file using 'astropy.io.fits' and extracting useful object parameters and spectrum using read_data from krome.spec_analysis
            # NOTE: The format of this FITS file must be ADP which contains the reduced spectrum with the wav, flux and flux_err in three columns
            
            if ccf_file_path != None:
                obj_params, spec = read_data(file_path=file_path[i],
                                             ccf_file_path=ccf_file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
            else:
                obj_params, spec = read_data(file_path=file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
                
                obj_params['RV'] = radial_velocity # setting obj_params['RV'] to the given radial_velocity argument!
                
            # Assigning appropriate variables from spec individually!
            wvl = spec[0] # nm
            flx = spec[1] # ADU
            flx_err = spec[2]
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
            
            shift = ((obj_params['RV']/ap.constants.c.value)*NaID1)  
            shift = (round(shift, 3)) # Using only 3 decimal places for the shift value since that's the precision of the wavelength in the .FITS files!
            
            # Since the HARPS spectra have their individual spectral orders stitched together, we do not have to extract them separately as done for NARVAL. Thus for HARPS, the required region is extracted by slicing the spectrum with the index corresponding to the left and right continuum obtained using the 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2 nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            # If condition for when certain files have NaN as the flux errors; probably for all since the ESO Phase 3 data currently does not the flux errors
            flx_err_nan = np.isnan(np.sum(flx_err)) # NOTE: This returns True if there is one NaN value or all are NaN values!
            
            if flx_err_nan:
                if print_stat:
                    print('File contains NaN in flux errors array. Calculating flux errors using CCD readout noise: {}'.format(np.round(RON, 4)))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                # Flux error calculated as photon noise plus CCD readout noise 
                # NOTE: The error calculation depends on a lot of other CCD parameters such as the pixel binning in each CCD
                # array and so on. But for photometric limited measurements, this noise is generally insignificant.
                
                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err_ron = [np.sqrt(flux + np.square(RON)) for flux in flx]
                
                if np.isnan(np.sum(flx_err_ron)):
                    if print_stat:
                        print('The calculated flux array contains a few NaN values due to negative flux encountered in the square root.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
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
                print('The doppler shift size using RV {} m/s and the NaID1 line of 588.995nm is: {}nm'.format(obj_params['RV'], shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            if norm_spec=='scale':
                if print_stat:
                    print('Normalizing the spectra by scaling it down to max. flux equals 1.0')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
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
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')
                            
            elif norm_spec=='poly1dfit':
                if print_stat:
                    print('Normalizing the spectra by fitting a 1st degree polynomial to the continuum')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(1))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
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
                    plt.tight_layout()
                    
                    if save_figs:
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')
                    
                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec.spectral_axis, spec.flux, label='Normalized')
                    ax2.axhline(1.0, ls='--', c='gray')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized")
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(save_figs_name), format='pdf')
                
                    
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
    
            regions = [NaID1_region, NaID2_region, F1_region, F2_region]
            
            # Calculating the index using 'calc_ind' from krome.spec_analysis
            
            I_NaI, I_NaI_err, F1_mean, F2_mean = calc_ind(regions=regions,
                                                          index_name='NaI',
                                                          print_stat=print_stat,
                                                          hfv=hfv)
            # Plotting the pseudo-continuum as the linear interpolation of the values in each red and blue cont. window!
            
            if plot_spec:
                
                x = [F1_line, F2_line]
                y = [F1_mean.value, F2_mean.value]
    
                f, ax  = plt.subplots(figsize=(10,4)) 
                ax.plot(spec.spectral_axis, spec.flux, color='black')
                ax.plot(x, y, 'og--', label='pseudo-continuum')
                plt.vlines(NaID1-(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(NaID1+(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(NaID2-(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(NaID2+(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(F1_line-(F1_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles=':', colors='blue', label='Blue cont. {}±{}'.format(F1_line, F1_band/2))
                plt.vlines(F1_line+(F1_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles=':', colors='blue')
                plt.vlines(F2_line-(F2_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='-.', colors='red', label='Red cont. {}±{}'.format(F2_line, F2_band/2))
                plt.vlines(F2_line+(F2_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='-.', colors='red')
                ax.set_xlabel('$\lambda (nm)$')
                if norm_spec:
                    ax.set_ylabel("Normalized Flux")
                else:
                    ax.set_ylabel("Flux (adu)")
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    if print_stat:
                        print('Saving plots as PDFs in the working directory')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    plt.savefig('{}_reduced_spec_plot.pdf'.format(save_figs_name), format='pdf')
                        
                f, ax1  = plt.subplots(figsize=(10,4))
                ax1.plot(spec.spectral_axis, spec.flux, color='black')
                ax1.set_xlabel('$\lambda (nm)$')
                if norm_spec:
                    ax1.set_ylabel("Normalized Flux")
                else:
                    ax1.set_ylabel("Flux (adu)")
                plt.vlines(NaID1, ymin=0, ymax=max(spec.flux.value), linestyles=':', colors='red', label='D1')
                plt.vlines(NaID2, ymin=0, ymax=max(spec.flux.value), linestyles=':', colors='blue', label='D2')
                plt.vlines(NaID1-(NaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='D1,D2 band width = {}nm'.format(NaI_band))
                plt.vlines(NaID1+(NaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(NaID2-(NaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(NaID2+(NaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax1.set_xlim(NaID2-(NaI_band/2)-0.2, NaID1+(NaI_band/2)+0.2)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_NaID1D2_lines_plot.pdf'.format(save_figs_name), format='pdf')
            
            header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'BERV', 'OBS_DATE', 'PROG_ID', 'SNR', 'SIGDET', 'CONAD', 'RON', 'RV', 'I_NaI', 'I_NaI_err']
            res = list(obj_params.values()) + [I_NaI, I_NaI_err]
            results.append(res)
        
        elif Instrument=='HARPS-N':
            
            # Opening the FITS file using 'astropy.io.fits' and extracting useful object parameters and spectrum using read_data from krome.spec_analysis
            # NOTE: The format of this FITS file must be s1d which only contains flux array. 
            # The wavelength array is constructed using the starting point (CRVAL1), length of spectral axis (NAXIS1) 
            # and wavelength step (CDELT1) from the FITS file header.
            
            if ccf_file_path != None:
                obj_params, spec = read_data(file_path=file_path[i],
                                             ccf_file_path=ccf_file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
            else:
                obj_params, spec = read_data(file_path=file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
                
                obj_params['RV'] = radial_velocity # setting obj_params['RV'] to the given radial_velocity argument!
                
            # Assigning appropriate variables from spec individually!
            wvl = spec[0] # nm
            flx = spec[1] # ADU
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
            
            shift = ((obj_params['RV']/ap.constants.c.value)*NaID1)  
            shift = (round(shift, 3)) 
            
             # Same as the HARPS spectra, the HARPS-N spectra have their individual spectral orders stitched together and 
             # we do not have to extract them separately as done for NARVAL. Thus, the required region is extracted by slicing
             # the spectrum with the index corresponding to the left and right continuum obtained using the 
             # 'find_nearest' function. 
            
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            with warnings.catch_warnings(): # Ignore warnings
                warnings.simplefilter('ignore')
                flx_err = [np.sqrt(flux) for flux in flx] # Using only photon noise as flx_err approx since no RON info available!
            
            # Slicing the data to contain only the region required for the index calculation as explained above and creating 
            # a spectrum class for it
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx], unit=u.Jy))
            
            if print_stat:
                print('The doppler shift size using RV {} m/s and the NaID1 line of 588.995nm is: {}nm'.format(obj_params['RV'], shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            if norm_spec=='scale':
                if print_stat:
                    print('Normalizing the spectra by scaling it down to max. flux equals 1.0')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
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
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')
                    
            elif norm_spec=='poly1dfit':
                if print_stat:
                    print('Normalizing the spectra by fitting a 1st degree polynomial to the continuum')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(1))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
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
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')
                
                    
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
                if norm_spec:
                    ax.set_ylabel("Normalized Flux")
                else:
                    ax.set_ylabel("Flux (adu)")
                
                if save_figs:
                    plt.savefig('{}_reduced_spec_plot.pdf'.format(save_figs_name), format='pdf')
                
                
            # Extracting the regions required for the index calculation using 'extract_region'
            
            NaID1_region = extract_region(spec, region=SpectralRegion((NaID1-(NaI_band/2))*u.nm, 
                                                                      (NaID1+(NaI_band/2))*u.nm))
            
            NaID2_region = extract_region(spec, region=SpectralRegion((NaID2-(NaI_band/2))*u.nm, 
                                                                      (NaID2+(NaI_band/2))*u.nm))
            
            F1_region = extract_region(spec, region=SpectralRegion((F1_line-(F1_band/2))*u.nm, 
                                                                   (F1_line+(F1_band/2))*u.nm))
            
            F2_region = extract_region(spec, region=SpectralRegion((F2_line-(F2_band/2))*u.nm, 
                                                                   (F2_line+(F2_band/2))*u.nm))
            
            regions = [NaID1_region, NaID2_region, F1_region, F2_region]
            
            # Calculating the index using 'calc_ind' from krome.spec_analysis
            
            I_NaI, I_NaI_err, F1_mean, F2_mean = calc_ind(regions=regions,
                                                          index_name='NaI',
                                                          print_stat=print_stat,
                                                          hfv=hfv)
            
            # Plotting the pseudo-continuum as the linear interpolation of the values in each red and blue cont. window!
            
            header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'OBS_DATE', 'PROG_ID', 'RV', 'I_NaI', 'I_NaI_err']
            res = list(obj_params.values()) + [I_NaI, I_NaI_err]
            results.append(res)
            
        else:
            
            raise ValueError('Instrument type not recognised. Available options are "NARVAL", "HARPS" and "HARPS-N"')
            
    # Saving the results of each Instrument type run in a .txt file with the given file name separated by a space; ' '.
         
    if save_results:
        
        if print_stat:
            print('Saving results in the working directory in file: {}.csv'.format(results_file_name))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')

        with open('{}.csv'.format(results_file_name), 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow(header)
            for row in results:
                writer.writerow(row)

    return results

## Defining a function to calculate the CaIIH index following Morgenthaler et al. 2012 (2012A&A...540A.138M)

def CaIIH_Index(file_path,
                radial_velocity, 
                degree=4, 
                CaIIH_line=396.847, 
                CaIIH_band=0.04, 
                cont_R_line=400.107,
                cont_R_band=2.0,
                Instrument='NARVAL',
                norm_spec=False,
                plot_fit=False,
                plot_spec=True,
                print_stat=True,
                save_results=False,
                results_file_name=None,
                save_figs=False,
                save_figs_name=None,
                out_file_path=None,
                ccf_file_path=None):
    
    """
    Calculates the CaIIH index following Morgenthaler A., et al., 2012, A&A, 540, A138. 
    NOTE: The CaIIH line flux is measured within a rectangular bandpass instead of a triangular one following Boisse I., et al., 2009, A&A, 495, 959.
    
    Parameters:
    -----------
    file_path: list, .s format (NARVAL), ADP..._.fits format (HARPS) or s1d_A.fits format (HARPS-N)
    List containng the paths of the spectrum files 
    
    radial_velocity: int
    Stellar radial velocity along the line-of-sight. This value is used for doppler shifting the spectra to its rest frame.
    
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
    
    save_figs_name: str, default=None
    Name with which to save the figures. NOTE: This should ideally be the observation date of the given spectrum.
    
    out_file_path: list, .out format (NARVAL), default: None
    List containing the paths of the .out files to extract the OBS_HJD. If None, HJD is returned as NaN. Used only when Instrument type is 'NARVAL'
    
    ccf_file_path: list, .fits format (HARPS/HARPS-N), default: None
    List containig the paths of the CCF FITS files to extract the radial velocity. If None, the given radial velocity argument is used for all files for doppler shift corrections
    
    Returns:
    -----------
    NARVAL: HJD, RA, DEC, AIRMASS, Exposure time[s], No. of exposures, GAIN [e-/ADU], ReadOut Noise [e-], V_mag, T_eff[K], RV[m/s], CaIIH index and error on CaIIH index
    HARPS: BJD, RA, DEC, AIRMASS, Exposure time[s], Barycentric RV[km/s], OBS_DATE, Program ID, SNR, CCD Readout Noise[e-], CCD conv factor[e-/ADU], ReadOut Noise[ADU], RV[m/s], CaIIH index and error on CaIIH index
    HARPS-N: BJD, RA, DEC, AIRMASS, Exposure time[s], OBS_DATE, Program ID', RV[m/s], CaIIH index and error on CaIIH index
    
    All values are type float() given inside a list.
    
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
                
                # Using read_data from krome.spec_analysis to extract useful object parameters and all individual spectral orders
                
                obj_params, orders = read_data(file_path=file_path[i],
                                               out_file_path=out_file_path[i],
                                               Instrument=Instrument,
                                               print_stat=print_stat,
                                               show_plots=False)
                
                obj_params['RV'] = radial_velocity # setting radial_velocity as part of the obj_params dictionary for continuity 
                
            else:
                
                orders = read_data(file_path=file_path[i],
                                   Instrument=Instrument,
                                   print_stat=print_stat,
                                   out_file_path=None,
                                   show_plots=False)
                
                if print_stat:
                    print('"out_file_path" not given as an argument. Run will only return the indices and their errros instead.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                

            if print_stat:
                print('Total {} spectral orders extracted'.format(len(orders)))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            # The CaIIH line is found only within one spectral order, # 57
            
            order_57 = orders[61-57] # The orders begin from # 61 so to get # 57, we index as 61-57.
            
            if print_stat:
                print('The #57 order wavelength read from .s file using pandas is: {}'.format(order_57[0].values))
                print('The #57 order intensity read from .s file using pandas is: {}'.format(order_57[1].values))
                print('The #57 order intensity error read from .s file using pandas is: {}'.format(order_57[2].values))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            # The spectra is now doppler shift corrected in the wavelength axis using the stellar radial velocity and the rest wavelength of CaIIH line; delta_lambda = (v/c)*lambda

            shift = ((radial_velocity/ap.constants.c.value)*CaIIH_line)
            shift = (round(shift, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in the .s files!

            wvl = np.round((order_57[0].values - shift), 4)
            flx = order_57[1].values
            flx_err = order_57[2].values
            
            # Creating a spectrum object called 'spec1d' using 'Spectrum1D' from 'specutils'
            # Docs for 'specutils' are here; https://specutils.readthedocs.io/en/stable/ 
            
            # The spectral and flux axes are given units nm and Jy respectively using 'astropy.units'. 
            # The uncertainty has units Jy as well!

            spec1d = Spectrum1D(spectral_axis=wvl*u.nm, 
                                flux=flx*u.Jy, 
                                uncertainty=StdDevUncertainty(flx, unit=u.Jy))
            
            # Printing info
            
            if print_stat:
                print('The doppler shift size using RV {} m/s and the CaIIH line of 396.847nm is: {}nm'.format(radial_velocity, shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral order used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 4 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            # Fitting an nth order polynomial to the continuum for normalisation using specutils

            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
                # 'fit_generic_continuum' is a function imported from 'specutils' which fits a given polynomial model to the given spectrum.
                
                with warnings.catch_warnings(): # Ignore warnings
                    warnings.simplefilter('ignore')
                    g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
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
                    plt.tight_layout()
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')

                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, color='blue', label='Re-normalised', alpha=0.6)
                    ax2.plot(spec1d.spectral_axis, spec1d.flux, color='red', label='Pipeline normalised', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalised Flux')
                    ax2.set_title("Continuum Normalized ")
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(save_figs_name), format='pdf')

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
                plt.tight_layout()
                
                if save_figs:
                    plt.savefig('{}_reduced_spec_plot.pdf'.format(save_figs_name), format='pdf')

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
                    plt.savefig('{}_CaIIH_line_plot.pdf'.format(save_figs_name), format='pdf')

        ## HARPS 

        elif Instrument == 'HARPS':

            # Opening the FITS file using 'astropy.io.fits' and extracting useful object parameters and spectrum using read_data from krome.spec_analysis
            # NOTE: The format of this FITS file must be ADP which contains the reduced spectrum with the wav, flux and flux_err in three columns
            
            if ccf_file_path != None:
                obj_params, spec = read_data(file_path=file_path[i],
                                             ccf_file_path=ccf_file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
            else:
                obj_params, spec = read_data(file_path=file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
                
                obj_params['RV'] = radial_velocity # setting obj_params['RV'] to the given radial_velocity argument!
                
            # Assigning appropriate variables from spec individually!
            wvl = spec[0] # nm
            flx = spec[1] # ADU
            flx_err = spec[2]
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
           
            shift = ((obj_params['RV']/ap.constants.c.value)*CaIIH_line)  
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
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                # Flux error calculated as photon noise plus CCD readout noise 
                # NOTE: The error calculation depends on a lot of other CCD parameters such as the pixel binning in each CCD
                # array and so on. But for photometric limited measurements, this noise is generally insignificant.
                
                with warnings.catch_warnings(): # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err_ron = [np.sqrt(flux + np.square(RON)) for flux in flx]
                
                if np.isnan(np.sum(flx_err_ron)):
                    if print_stat:
                        print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
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
                print('The doppler shift size using RV {} m/s and the CaIIH line of 396.847nm is: {}nm'.format(obj_params['RV'], shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, 
                                                                                                                                                              spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')

            # Fitting an nth order polynomial to the continuum for normalisation using specutils
            
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                # 'fit_generic_continuum' is a function imported from 'specutils' which fits a given polynomial model to the given spectrum.
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree)) # Using 'Chebyshev1D' to define an nth order polynomial model
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis) # Continuum fit y values are calculated by inputting the spectral axis x values into the polynomial fit equation 
                spec_normalized = spec1d / y_cont_fitted
                
                spec = spec_normalized # Note the continuum normalized spectrum also has new uncertainty values which are simply the errors divided by this polynomial fit.

                # Plots the polynomial fits
                if plot_fit:
                    f, ax1 = plt.subplots()  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Flux (adu)')
                    ax1.set_title("Continuum Fitting")
                    plt.tight_layout()
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')

                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, label='Re-Normalized')
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(cont_R_line+(cont_R_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized ")
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(save_figs_name), format='pdf')

                spec = spec_normalized # Note the continuum normalised spectrum also has new uncertainty values!

            else:
                spec = spec1d


            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            if plot_spec:
                ax  = plt.subplots()[1]  
                ax.plot(spec.spectral_axis, spec.flux)  
                ax.set_xlabel('$\lambda (nm)$')
                if norm_spec:
                    ax.set_ylabel("Normalized Flux")
                else:
                    ax.set_ylabel("Flux (adu)")
                plt.vlines(CaIIH_line, ymin=0.0, ymax=2.5, linestyles='dotted', colors='green')
                plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black', label='CaIIH band width = ({}±{})nm'.format(CaIIH_line, CaIIH_band/2))
                plt.vlines(CaIIH_line+(CaIIH_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black')
                plt.vlines(cont_R_line-(cont_R_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='red', label='Right ref. band width = ({}±{})nm'.format(cont_R_line, cont_R_band/2))
                plt.vlines(cont_R_line+(cont_R_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='red')
                plt.xlim(CaIIH_line-(CaIIH_band/2)-0.05, cont_R_line+(cont_R_band/2)+0.05)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_reduced_spec_plot.pdf'.format(save_figs_name), format='pdf')

                ax1  = plt.subplots()[1]  
                ax1.plot(spec.spectral_axis, spec.flux)
                ax1.set_xlabel('$\lambda (nm)$')
                if norm_spec:
                    ax1.set_ylabel("Normalized Flux")
                else:
                    ax1.set_ylabel("Flux (adu)")
                plt.vlines(CaIIH_line, ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='dotted', colors='green')
                plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black', label='CaIIH band width = {}nm'.format(CaIIH_band))
                plt.vlines(CaIIH_line+(CaIIH_band/2), ymin=min(spec.flux.value), ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax1.set_xlim(CaIIH_line-(CaIIH_band/2)-0.1, CaIIH_line+(CaIIH_band/2)+0.1)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_CaIIH_line_plot.pdf'.format(save_figs_name), format='pdf')
                
        ## HARPS-N
                
        elif Instrument=='HARPS-N':
            
            # Opening the FITS file using 'astropy.io.fits' and extracting useful object parameters and spectrum using read_data from krome.spec_analysis
            # NOTE: The format of this FITS file must be s1d which only contains flux array. 
            # The wavelength array is constructed using the starting point (CRVAL1), length of spectral axis (NAXIS1) 
            # and wavelength step (CDELT1) from the FITS file header.
            
            if ccf_file_path != None:
                obj_params, spec = read_data(file_path=file_path[i],
                                             ccf_file_path=ccf_file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
            else:
                obj_params, spec = read_data(file_path=file_path[i],
                                             Instrument=Instrument,
                                             print_stat=print_stat,
                                             show_plots=False)
                
                obj_params['RV'] = radial_velocity # setting obj_params['RV'] to the given radial_velocity argument!
                
            # Assigning appropriate variables from spec individually!
            wvl = spec[0] # nm
            flx = spec[1] # ADU
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from the CCF FITS file
            
            shift = ((obj_params['RV']/ap.constants.c.value)*CaIIH_line)  
            shift = (round(shift, 3)) 
            
            # Same as the HARPS spectra, the HARPS-N spectra have their individual spectral orders stitched together and 
            # we do not have to extract them separately as done for NARVAL. Thus, the required region is extracted by slicing
            # the spectrum with the index corresponding to the left and right continuum obtained using the 
            # 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, CaIIH_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, cont_R_line+2)
            
            with warnings.catch_warnings(): # Ignore warnings
                warnings.simplefilter('ignore')
                flx_err = [np.sqrt(flux) for flux in flx] # Using only photon noise as flx_err approx since no RON info available!
            
            # Slicing the data to contain only the region required for the index calculation as explained above and creating 
            # a spectrum class for it
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx], unit=u.Jy))
            
            if print_stat:
                print('The doppler shift size using RV {} m/s and the CaIIH line of 396.847nm is: {}nm'.format(obj_params['RV'], shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, 
                                                                                                                                                              spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            # Fitting an nth order polynomial to the continuum for normalisation using specutils
            
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                g_fit = fit_generic_continuum(spec1d, model=Chebyshev1D(degree))
                
                if print_stat:
                    print('Polynomial fit coefficients:')
                    print(g_fit)
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                y_cont_fitted = g_fit(spec1d.spectral_axis)
                spec_normalized = spec1d / y_cont_fitted
                
                # Plots the polynomial fits
                if plot_fit:
                    f, ax1 = plt.subplots(figsize=(10,4))  
                    ax1.plot(spec1d.spectral_axis, spec1d.flux)  
                    ax1.plot(spec1d.spectral_axis, y_cont_fitted)
                    ax1.set_xlabel('$\lambda (nm)$')
                    ax1.set_ylabel('Flux (adu)')
                    ax1.set_title("Continuum Fitting")
                    plt.tight_layout()
                    
                    # Saves the plot in a pdf format in the working directory
                    if save_figs:
                        if print_stat:
                            print('Saving plots as PDFs in the working directory')
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')
                    
                    f, ax2 = plt.subplots(figsize=(10,4))  
                    ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux, color='blue', label='Re-Normalized', alpha=0.6)
                    plt.axhline(1.0, ls='--', c='gray')
                    plt.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='Region used for index calc.')
                    plt.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized ")
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_norm_plot.pdf'.format(save_figs_name), format='pdf')
                    
                spec = spec_normalized # Note the continuum normalized spectrum also has new uncertainty values!
                
            else:
                
                spec = spec1d
                
            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            if plot_spec:
                ax  = plt.subplots()[1]  
                ax.plot(spec.spectral_axis, spec.flux)  
                ax.set_xlabel('$\lambda (nm)$')
                if norm_spec:
                    ax.set_ylabel("Normalized Flux")
                else:
                    ax.set_ylabel("Flux (adu)")
                plt.vlines(CaIIH_line, ymin=0.0, ymax=2.5, linestyles='dotted', colors='green')
                plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=-1.0, ymax=4, linestyles='--', colors='black', label='CaIIH band width = ({}±{})nm'.format(CaIIH_line, CaIIH_band/2))
                plt.vlines(CaIIH_line+(CaIIH_band/2), ymin=-1.0, ymax=4, linestyles='--', colors='black')
                plt.vlines(cont_R_line-(cont_R_band/2), ymin=-1.0, ymax=4, linestyles='--', colors='red', label='Right ref. band width = ({}±{})nm'.format(cont_R_line, cont_R_band/2))
                plt.vlines(cont_R_line+(cont_R_band/2), ymin=-1.0, ymax=4, linestyles='--', colors='red')
                plt.xlim(CaIIH_line-(CaIIH_band/2)-0.05, cont_R_line+(cont_R_band/2)+0.05)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_reduced_spec_plot.pdf'.format(save_figs_name), format='pdf')

                ax1  = plt.subplots()[1]  
                ax1.plot(spec.spectral_axis, spec.flux)
                ax1.set_xlabel('$\lambda (nm)$')
                if norm_spec:
                    ax1.set_ylabel("Normalized Flux")
                else:
                    ax1.set_ylabel("Flux (adu)")
                plt.vlines(CaIIH_line, ymin=0.0, ymax=2.5, linestyles='dotted', colors='green')
                plt.vlines(CaIIH_line-(CaIIH_band/2), ymin=-1, ymax=4, linestyles='--', colors='black', label='CaIIH band width = {}nm'.format(CaIIH_band))
                plt.vlines(CaIIH_line+(CaIIH_band/2), ymin=-1, ymax=4, linestyles='--', colors='black')
                ax1.set_xlim(CaIIH_line-(CaIIH_band/2)-0.1, CaIIH_line+(CaIIH_band/2)+0.1)
                plt.tight_layout()
                plt.legend()
                
                if save_figs:
                    plt.savefig('{}_CaIIH_line_plot.pdf'.format(save_figs_name), format='pdf')

        else:
            raise ValueError('Instrument type not recognisable. Available options are "NARVAL", "HARPS" and "HARPS-N"')
            
        # Now we have the final spectrum to work with as a variable, 'spec'!
        
        # The two regions required for CaIIH index calculation are extracted from 'spec' using the 'extract region' function from 'specutils'. 
        # The function uses another function called 'SpectralRegion' as one of its arguments which defines the region to be extracted done so using the line and line bandwidth values; i.e. left end of region would be 'line - bandwidth/2' and right end would be 'line + bandwidth/2'.
        # Note: These values must have the same units as the spec wavelength axis.

        # Extracting the CaIIH line region using the given bandwidth 'CaIIH_band'
        F_CaIIH_region = extract_region(spec, region=SpectralRegion((CaIIH_line-(CaIIH_band/2))*u.nm, (CaIIH_line+(CaIIH_band/2))*u.nm))
        
        # Doing the same for the cont R region!
        cont_R_region = extract_region(spec, region=SpectralRegion((cont_R_line-(cont_R_band/2))*u.nm, (cont_R_line+(cont_R_band/2))*u.nm))
        
        regions = [F_CaIIH_region, cont_R_region]
        
        # Calculating the index using 'calc_inc' from krome.spec_analysis
        
        I_CaIIH, I_CaIIH_err = calc_ind(regions=regions,
                                        index_name='CaIIH',
                                        print_stat=print_stat)
            
        if Instrument=='NARVAL':
            if out_file_path != None:
                header = ['HJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'NUM_EXP', 'GAIN', 'RON', 'V_mag', 'T_eff', 'RV', 'I_CaIIH', 'I_CaIIH_err']
                res = list(obj_params.values()) + [I_CaIIH, I_CaIIH_err] # Creating results list 'res' containing the calculated parameters and appending this list to the 'results' empty list created at the start of this function!
                results.append(res)
            else:
                header = ['I_CaIIH', 'I_CaIIH_err']
                res = [I_CaIIH, I_CaIIH_err]
                results.append(res)
        
        elif Instrument=='HARPS':
            header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'BERV', 'OBS_DATE', 'PROG_ID', 'SNR', 'SIGDET', 'CONAD', 'RON', 'RV', 'I_CaIIH', 'I_CaIIH_err']
            res = list(obj_params.values()) + [I_CaIIH, I_CaIIH_err]
            results.append(res)
            
        elif Instrument=='HARPS-N':
            header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'OBS_DATE', 'PROG_ID', 'RV', 'I_CaIIH', 'I_CaIIH_err']
            res = list(obj_params.values()) + [I_CaIIH, I_CaIIH_err]
            results.append(res)
            
    # Saving the results in a csv file format  
    if save_results:
        
        if print_stat:
            print('Saving results in the working directory in file: {}.csv'.format(results_file_name))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        with open('{}.csv'.format(results_file_name), 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow(header)
            for row in results:
                writer.writerow(row)  
            
    return results
