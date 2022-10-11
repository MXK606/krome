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
from krome.spec_analysis import find_nearest, read_data, calc_ind, normalise_spec
from krome.plotting import plot_spectrum
    
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
                  meta_file_path=None,
                  ccf_file_path=None,
                  CaI_index=True,
                  plot_only_spec=False):
    
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
    List containing the paths of the .out files to extract the OBS_HJD along with other useful object parameters. 
    
    meta_file_path: list, .meta format (ESPADONS), default: None
    List containing the paths of the .meta files to extract the OBS_HJD along with other useful object parameters. 
    
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
                print('The #34 order wavelength read from .s file using pandas is: {}'.format(order_34[0]))
                print('The #34 order intensity read from .s file using pandas is: {}'.format(order_34[1]))
                print('The #34 order intensity error read from .s file using pandas is: {}'.format(order_34[2]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
            
            # The spectra is now doppler shift corrected in the wavelength axis using the stellar radial velocity and the rest wavelength of H alpha line; delta_lambda = (v/c)*lambda
            
            shift = ((radial_velocity/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in the .s files!
            
            # Subtracting the calculated doppler shift value from the wavelength axis since the stellar radial velocity is positive. 
            # If the stellar RV is negative, the shift value will be added instead.
            
            wvl = np.round((order_34[0] - shift), 4) 
                                                           
            flx = order_34[1] # Indexing flux array from order_34
            flx_err = order_34[2] # Indexing flux_err array from order_34
            
            # Creating a spectrum object called 'spec1d' using 'Spectrum1D' from 'specutils'
            # Docs for 'specutils' are here; https://specutils.readthedocs.io/en/stable/ 
            
            # The spectral and flux axes are given units nm and Jy respectively using 'astropy.units'. 
            # The uncertainty has units Jy as well!
        
            spec1d = Spectrum1D(spectral_axis=wvl*u.nm, 
                                flux=flx*u.Jy, 
                                uncertainty=StdDevUncertainty(flx_err, unit=u.Jy)) 
                        
        # ESPADONS
        
        elif Instrument == 'ESPADONS':
            
            if meta_file_path != None:
                
                # Using read_data from krome.spec_analysis to extract useful object parameters and all individual spectral orders
                
                obj_params, orders = read_data(file_path=file_path[i],
                                               meta_file_path=meta_file_path[i],
                                               Instrument=Instrument,
                                               print_stat=print_stat,
                                               show_plots=False)
                
                obj_params['RV'] = radial_velocity # setting radial_velocity as part of the obj_params dictionary for continuity 
                
            else:
                
                orders = read_data(file_path=file_path[i],
                                   Instrument=Instrument,
                                   print_stat=print_stat,
                                   meta_file_path=None,
                                   show_plots=False)
                
                if print_stat:
                    print('"meta_file_path" not given as an argument. Run will only return the indices and their errros instead.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                

            if print_stat:
                print('Total {} spectral orders extracted'.format(len(orders)))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
            
            order_34 = orders[61-34] # The orders begin from # 61 so to get # 34, we index as 61-34.
            
            if print_stat:
                print('The #34 order wavelength read from .s file using pandas is: {}'.format(order_34[0]))
                print('The #34 order intensity read from .s file using pandas is: {}'.format(order_34[1]))
                print('The #34 order intensity error read from .s file using pandas is: {}'.format(order_34[2]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
            shift = ((radial_velocity/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 4))
            
            wvl = np.round((order_34[0] - shift), 4) 
            flx = order_34[1] 
            flx_err = order_34[2] 
        
            spec1d = Spectrum1D(spectral_axis=wvl*u.nm, 
                                flux=flx*u.Jy, 
                                uncertainty=StdDevUncertainty(flx_err, unit=u.Jy))    
                
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
                    flx_err_ron = np.asarray([np.sqrt(flux + np.square(obj_params['RON'])) for flux in flx])
                
                if np.isnan(np.sum(flx_err_ron)):
                    if print_stat:
                        print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        
                if print_stat:
                    print('The wavelength array read from the .fits file is: {}'.format(wvl))
                    print('The flux array read from the .fits file is: {}'.format(flx))
                    print('The calculated flux error array is: {}'.format(flx_err_ron))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                # Slicing the data to contain only the region required for the index calculation as explained above and 
                # creating a spectrum class for it.
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx+1]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err_ron[left_idx:right_idx+1], unit=u.Jy))
                
            else:
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx+1]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
              
        # HARPS-N
                
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
            
            if print_stat:
                print('Calculating the flux error array as the photon noise')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err = np.asarray([np.sqrt(flux) for flux in flx]) # Using only photon noise as flx_err approx since no RON info available!
                    
            if np.isnan(np.sum(flx_err)):
                if print_stat:
                    print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
            if print_stat:
                print('The wavelength array read from the .fits file is: {}'.format(wvl))
                print('The flux array read from the .fits file is: {}'.format(flx))
                print('The calculated flux error array is: {}'.format(flx_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            # Slicing the data to contain only the region required for the index calculation as explained above and creating 
            # a spectrum class for it
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx+1]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
            
        # SOPHIE
        
        elif Instrument == 'SOPHIE':
            
            obj_params, spec = read_data(file_path=file_path[i],
                                         Instrument=Instrument,
                                         print_stat=print_stat,
                                         show_plots=False)
            
            obj_params['RV'] = radial_velocity 
            
            # Checking if the FITS file is e2ds since it has 39 spectral orders using an arbitray order number of 50. If greater than 50, assume its s1d.
            
            if len(spec[0]) < 50:
                
                if print_stat:
                    print('Total {} spectral orders extracted'.format(len(spec[0])))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        
                wvl = spec[0][36] ## For SOPHIE spectra, the H alpha line along with its reference bands are within the 37th order.
                flx = spec[1][36]
                    
                # Flux error array is calculated as photon noise plus CCD readout noise 

                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err = np.asarray([np.sqrt(flux + np.square(obj_params['RON'])) for flux in flx])
                
            else:
                
                left_idx = find_nearest(spec[0], F1_line-2) # ± 2nm extra included for both!
                right_idx = find_nearest(spec[0], F2_line+2)
                
                # Slicing the data to contain only the region required for the index calculation
                
                wvl = spec[0][left_idx:right_idx+1]
                flx = spec[1][left_idx:right_idx+1]
                
                # Flux error array is calculated as photon noise alone since RON isn't available

                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err = np.asarray([np.sqrt(flux) for flux in flx])
                    
                
            if np.isnan(np.sum(flx_err)):
                if print_stat:
                    print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
            if print_stat:
                print('The wavelength array read from .fits file is: {}'.format(wvl))
                print('The flux array read from .fits file is: {}'.format(flx))
                print('The calculated flux error array is: {}'.format(flx_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')    
                
            
            shift = ((obj_params['RV']/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 4)) 
            
            wvl_shifted = np.round((wvl - shift), 4) 
            
            spec1d = Spectrum1D(spectral_axis=wvl_shifted*u.nm, 
                                flux=flx*u.Jy, 
                                uncertainty=StdDevUncertainty(flx_err, unit=u.Jy)) 
            
        # ELODIE
        
        elif Instrument=='ELODIE':
            
            obj_params, spec = read_data(file_path=file_path[i],
                                         Instrument=Instrument,
                                         print_stat=print_stat,
                                         show_plots=False)
            
            obj_params['RV'] = radial_velocity
            
            wvl = spec[0] # nm
            flx = spec[1] # ADU
            
            shift = ((obj_params['RV']/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 3)) 
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            if print_stat:
                print('Calculating the flux error array as the photon noise')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err = np.asarray([np.sqrt(flux) for flux in flx]) 
                    
            if np.isnan(np.sum(flx_err)):
                if print_stat:
                    print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
            if print_stat:
                print('The wavelength array read from the .fits file is: {}'.format(wvl))
                print('The flux array read from the .fits file is: {}'.format(flx))
                print('The calculated flux error array is: {}'.format(flx_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx+1]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
            
                    
        else:
            raise ValueError('Instrument type not recognised. Available options are "NARVAL", "ESPADONS", "HARPS", "HARPS-N", "SOPHIE", "ELODIE"')
            
        # Printing spec info
            
        if print_stat:
            print('The doppler shift size using RV {} m/s and the Hα line of 656.2808nm is: {:.4f}nm'.format(radial_velocity, shift))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('The spectral axis ranges from {:.4f}nm to {:.4f}nm.'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('These values are doppler shift corrected and rounded off to 4 decimal places')
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        
        # Fitting an nth order polynomial to the continuum for normalisation using specutils
            
        if norm_spec:
            if print_stat:
                print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            # Note the continuum normalized spectrum also has new uncertainty values!
            
            spec = normalise_spec(spec1d, degree, F1_line, F1_band, F2_line, F2_band,
                                  print_stat, plot_fit, save_figs, save_figs_name) 
            
        else:
            spec = spec1d
            
            
        # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
        
        if plot_spec:
            
            lines = [H_alpha_line, H_alpha_band, F1_line, F1_band, F2_line, F2_band, CaI_line, CaI_band]
            
            plot_spectrum(spec, lines, 'HaI', Instrument, norm_spec, save_figs, save_figs_name, CaI_index)
            
        if plot_only_spec:
            
            pass
        
        else:
            
            # Now we have the spectrum to work with as a variable, 'spec'!
            
            # The three regions required for H alpha index calculation are extracted from 'spec' using the 'extract region' function from 'specutils'. 
            # The function uses another function called 'SpectralRegion' as one of its arguments which defines the region to be extracted done so 
            # using the line and line bandwidth values; i.e. left end of region would be 'line - bandwidth/2' and right end would be 'line + bandwidth/2'.
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
                    # Creating results list 'res' containing the calculated parameters and appending this list to the 'results' empty list created at the start of this function!
                    res = list(obj_params.values()) + [I_Ha, I_Ha_err, I_CaI, I_CaI_err] 
                    results.append(res)
                else:
                    header = ['I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err']
                    res = [I_Ha, I_Ha_err, I_CaI, I_CaI_err]
                    results.append(res)
                    
            elif Instrument=='ESPADONS':
                if meta_file_path != None:
                    header = ['OBS_DATE', 'RA', 'DEC', 'V_mag', 'T_eff', 'Distance', 'JD', 'AIRMASS', 'T_EXP', 'RUN_ID', 'SNR', 'RV', 'I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err']
                    res = list(obj_params.values()) + [I_Ha, I_Ha_err, I_CaI, I_CaI_err] 
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
                
            elif Instrument=='SOPHIE':
                header = ['JD', 'RA', 'DEC', 'T_EXP', 'OBS_DATE', 'PROG_ID', 'SIGDET', 'CONAD', 'RON', 'RV', 'I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err']
                res = list(obj_params.values()) + [I_Ha, I_Ha_err, I_CaI, I_CaI_err]
                results.append(res)
                
            elif Instrument=='ELODIE':
                header = ['JD', 'RA', 'DEC', 'T_EXP', 'OBS_DATE', 'AIRMASS', 'SNR', 'GAIN', 'RV', 'I_Ha', 'I_Ha_err', 'I_CaI', 'I_CaI_err']
                res = list(obj_params.values()) + [I_Ha, I_Ha_err, I_CaI, I_CaI_err]
                results.append(res)
                
    if plot_only_spec:
        
        return
    
    else:
        
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
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
                
                spec1 = normalise_spec(spec1,
                                      degree,
                                      F1_line,
                                      F1_band,
                                      F2_line,
                                      F2_band,
                                      print_stat,
                                      plot_fit,
                                      save_figs,
                                      save_figs_name) 
                    
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
                    ax2.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='--', colors='black', label='Blue cont. region')
                    ax2.vlines(F1_line+(F1_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized First Order")
                    plt.tight_layout()
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
                    ax2.vlines(NaID1-1.0, ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black', label='NaID lines region')
                    ax2.vlines(NaID2+1.0, ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized Second Order")
                    plt.tight_layout()
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
                    ax2.vlines(F2_line-(F2_band/2), ymin=0, ymax=max(spec3.flux.value), linestyles='--', colors='black', label='F2 region')
                    ax2.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec3.flux.value), linestyles='--', colors='black')
                    ax2.set_xlabel('$\lambda (nm)$')
                    ax2.set_ylabel('Normalized Flux')
                    ax2.set_title("Continuum Normalized Third Order")
                    plt.tight_layout()
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
                ax.vlines(F1_line-(F1_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='dotted', colors='blue', label='Blue cont. {}±{}'.format(F1_line, F1_band/2))
                ax.vlines(F1_line+(F1_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='dotted', colors='blue')
                ax.vlines(F2_line-(F2_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='dashdot', colors='red', label='Red cont. {}±{}'.format(F2_line, F2_band/2))
                ax.vlines(F2_line+(F2_band/2), ymin=0, ymax=max(spec1.flux.value), linestyles='dashdot', colors='red')
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
                ax1.vlines(NaID1, ymin=0, ymax=max(spec2.flux.value), linestyles='dotted', colors='red', label='D1')
                ax1.vlines(NaID2, ymin=0, ymax=max(spec2.flux.value), linestyles='dotted', colors='blue', label='D2')
                ax1.vlines(NaID1-(NaI_band/2), ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black', label='D1,D2 band width = {}nm'.format(NaI_band))
                ax1.vlines(NaID1+(NaI_band/2), ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black')
                ax1.vlines(NaID2-(NaI_band/2), ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black')
                ax1.vlines(NaID2+(NaI_band/2), ymin=0, ymax=max(spec2.flux.value), linestyles='--', colors='black')
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
                if np.isnan(obj_params['RON']):
                    if print_stat:
                        print('File contains NaN in flux errors array. Calculating flux errors using CCD readout noise: {}'.format(np.round(obj_params['RON'], 4)))
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

                    spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                        flux=flx[left_idx:right_idx+1]*u.Jy,
                                        uncertainty=StdDevUncertainty(flx_err_ron[left_idx:right_idx+1], unit=u.Jy))
                else:
                    if print_stat:
                        print('File contains NaN in flux errors array and could not extract the ReadOut Noise (RON) from FITS file header.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        print('Approximating flux errors as the photon noise instead.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
                    with warnings.catch_warnings():  # Ignore warnings
                        warnings.simplefilter('ignore')
                        flx_err_pn = [np.sqrt(flux) for flux in flx]

                    if np.isnan(np.sum(flx_err_pn)):
                        if print_stat:
                            print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')

                    # Slicing the data to contain only the region required for the index calculation as explained above and 
                    # creating a spectrum class for it.

                    spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                        flux=flx[left_idx:right_idx+1]*u.Jy,
                                        uncertainty=StdDevUncertainty(flx_err_pn[left_idx:right_idx+1], unit=u.Jy))
                
            else:
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx+1]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
            
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
                    plt.tight_layout()
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
                ax.vlines(NaID1-(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax.vlines(NaID1+(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax.vlines(NaID2-(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax.vlines(NaID2+(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax.vlines(F1_line-(F1_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles=':', colors='blue', label='Blue cont. {}±{}'.format(F1_line, F1_band/2))
                ax.vlines(F1_line+(F1_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles=':', colors='blue')
                ax.vlines(F2_line-(F2_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='-.', colors='red', label='Red cont. {}±{}'.format(F2_line, F2_band/2))
                ax.vlines(F2_line+(F2_band/2), ymin=-0.1, ymax=max(spec.flux.value), linestyles='-.', colors='red')
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
                ax1.vlines(NaID1, ymin=0, ymax=max(spec.flux.value), linestyles=':', colors='red', label='D1')
                ax1.vlines(NaID2, ymin=0, ymax=max(spec.flux.value), linestyles=':', colors='blue', label='D2')
                ax1.vlines(NaID1-(NaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black', label='D1,D2 band width = {}nm'.format(NaI_band))
                ax1.vlines(NaID1+(NaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax1.vlines(NaID2-(NaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
                ax1.vlines(NaID2+(NaI_band/2), ymin=0, ymax=max(spec.flux.value), linestyles='--', colors='black')
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
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx+1]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
            
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
                    plt.tight_layout()
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
                    plt.tight_layout()
                    plt.legend()
                    
                    if save_figs:
                        plt.savefig('{}_cont_fit_plot.pdf'.format(save_figs_name), format='pdf')
                
                    
            else:
                spec = spec1d
                
            if plot_spec:
    
                ax  = plt.subplots()[1]  
                ax.plot(spec.spectral_axis, spec.flux, color='black')
                if norm_spec:
                    ax.set_ylabel("Normalized Flux")
                else:
                    ax.set_ylabel("Flux (adu)")
                ax.vlines(NaID1-(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='black')
                ax.vlines(NaID1+(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='black')
                ax.vlines(NaID2-(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='black')
                ax.vlines(NaID2+(NaI_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='black')
                ax.vlines(F1_line-(F1_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='blue', label='Blue cont. {}±{}'.format(F1_line, F1_band/2))
                ax.vlines(F1_line+(F1_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='blue')
                ax.vlines(F2_line-(F2_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='red', label='Red cont. {}±{}'.format(F2_line, F2_band/2))
                ax.vlines(F2_line+(F2_band/2), ymin=-0.1, ymax=max(spec.flux.value)+10, linestyles='--', colors='red')
                ax.set_xlabel('$\lambda (nm)$')
                plt.tight_layout()
                plt.legend()
                
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

## Updating this function to calculate the CaII H&K index!

def CaIIHK_Index(file_path,
                 radial_velocity, 
                 degree=4, 
                 CaIIH_line=396.847, 
                 CaIIH_band=0.04,
                 CaIIK_line=393.3664, 
                 CaIIK_band=0.04,
                 F1_line=390.107,
                 F1_band=2.0,
                 F2_line=400.107,
                 F2_band=2.0,
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
                 meta_file_path=None,
                 plot_only_spec=False):
    
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
    
    for i in log_progress(range(len(file_path)), desc='Calculating CaII HK Index'):
        
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
                
            # The CaIIK line is found within two spectral orders, # 57 and # 58. We will use # 58 since it contains the left reference continuum as well!
            
            CaIIK_order = orders[61-58] # The orders begin from # 61 so to get # 58, we index as 61-58.
            
            if print_stat:
                print('The CaII K order wavelength read from .s file using pandas is: {}'.format(CaIIK_order[0]))
                print('The CaII K order intensity read from .s file using pandas is: {}'.format(CaIIK_order[1]))
                print('The CaII K order intensity error read from .s file using pandas is: {}'.format(CaIIK_order[2]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            # The CaIIH line is found only within one spectral order, # 57
            
            CaIIH_order = orders[61-57] 
            
            if print_stat:
                print('The CaII H order wavelength read from .s file using pandas is: {}'.format(CaIIH_order[0]))
                print('The CaII H order intensity read from .s file using pandas is: {}'.format(CaIIH_order[1]))
                print('The CaII H order intensity error read from .s file using pandas is: {}'.format(CaIIH_order[2]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                
            # The spectra is now doppler shift corrected in the wavelength axis using the stellar radial velocity and 
            # the rest wavelength of CaIIH line; delta_lambda = (v/c)*lambda

            shift = ((radial_velocity/ap.constants.c.value)*CaIIH_line)
            shift = (round(shift, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in the .s files!
            
            wvl_K = np.round((CaIIK_order[0] - shift), 4)
            flx_K = CaIIK_order[1]
            flx_err_K = CaIIK_order[2]

            wvl_H = np.round((CaIIH_order[0] - shift), 4)
            flx_H = CaIIH_order[1]
            flx_err_H = CaIIH_order[2]
            
            # Creating a spectrum object called 'spec1d' using 'Spectrum1D' from 'specutils' for each line
            # Docs for 'specutils' are here; https://specutils.readthedocs.io/en/stable/ 
            
            # The spectral and flux axes are given units nm and Jy respectively using 'astropy.units'. 
            # The uncertainty has units Jy as well!
            
            spec1d_K = Spectrum1D(spectral_axis=wvl_K*u.nm, 
                                  flux=flx_K*u.Jy, 
                                  uncertainty=StdDevUncertainty(flx_err_K, unit=u.Jy))

            spec1d_H = Spectrum1D(spectral_axis=wvl_H*u.nm, 
                                  flux=flx_H*u.Jy, 
                                  uncertainty=StdDevUncertainty(flx_err_H, unit=u.Jy))
            
            spec1d = [spec1d_K, spec1d_H] # Creating a list containing both Spectrum1D objects
            
            
        # ESPADONS
        
        elif Instrument == 'ESPADONS':
            
            if meta_file_path != None:
                
                obj_params, orders = read_data(file_path=file_path[i],
                                               meta_file_path=meta_file_path[i],
                                               Instrument=Instrument,
                                               print_stat=print_stat,
                                               show_plots=False)
                
                obj_params['RV'] = radial_velocity 
                
            else:
                
                orders = read_data(file_path=file_path[i],
                                   Instrument=Instrument,
                                   print_stat=print_stat,
                                   meta_file_path=None,
                                   show_plots=False)
                
                if print_stat:
                    print('"meta_file_path" not given as an argument. Run will only return the indices and their errros instead.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                

            if print_stat:
                print('Total {} spectral orders extracted'.format(len(orders)))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
            
            CaIIK_order = orders[61-58] 
            
            if print_stat:
                print('The CaII K order wavelength read from .s file using pandas is: {}'.format(CaIIK_order[0]))
                print('The CaII K order intensity read from .s file using pandas is: {}'.format(CaIIK_order[1]))
                print('The CaII K order intensity error read from .s file using pandas is: {}'.format(CaIIK_order[2]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            CaIIH_order = orders[61-57] 
            
            if print_stat:
                print('The CaII H order wavelength read from .s file using pandas is: {}'.format(CaIIH_order[0]))
                print('The CaII H order intensity read from .s file using pandas is: {}'.format(CaIIH_order[1]))
                print('The CaII H order intensity error read from .s file using pandas is: {}'.format(CaIIH_order[2]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')

            shift = ((radial_velocity/ap.constants.c.value)*CaIIH_line)
            shift = (round(shift, 4)) 
            
            wvl_K = np.round((CaIIK_order[0] - shift), 4)
            flx_K = CaIIK_order[1]
            flx_err_K = CaIIK_order[2]

            wvl_H = np.round((CaIIH_order[0] - shift), 4)
            flx_H = CaIIH_order[1]
            flx_err_H = CaIIH_order[2]
            
            # Creating a spectrum object called 'spec1d' using 'Spectrum1D' from 'specutils' for each line
            # Docs for 'specutils' are here; https://specutils.readthedocs.io/en/stable/ 
            
            # The spectral and flux axes are given units nm and Jy respectively using 'astropy.units'. 
            # The uncertainty has units Jy as well!
            
            spec1d_K = Spectrum1D(spectral_axis=wvl_K*u.nm, 
                                  flux=flx_K*u.Jy, 
                                  uncertainty=StdDevUncertainty(flx_err_K, unit=u.Jy))

            spec1d_H = Spectrum1D(spectral_axis=wvl_H*u.nm, 
                                  flux=flx_H*u.Jy, 
                                  uncertainty=StdDevUncertainty(flx_err_H, unit=u.Jy))
            
            spec1d = [spec1d_K, spec1d_H] # Creating a list containing both Spectrum1D objects 

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
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            # If condition for when certain files have NaN as the flux errors; probably for all since the ESO Phase 3 data currently does not provide the flux errors
            
            flx_err_nan = np.isnan(np.sum(flx_err)) # NOTE: This returns true if there is one NaN or all are NaN!
            
            if flx_err_nan:
                if np.isnan(obj_params['RON']):
                    if print_stat:
                        print('File contains NaN in flux errors array. Calculating flux error using CCD readout noise: {}'.format(np.round(obj_params['RON'], 4)))
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    # Flux error calculated as photon noise plus CCD readout noise 
                    # NOTE: The error calculation depends on a lot of other CCD parameters such as the pixel binning in each CCD
                    # array and so on. But for photometric limited measurements, this noise is generally insignificant.

                    with warnings.catch_warnings(): # Ignore warnings
                        warnings.simplefilter('ignore')
                        flx_err_ron = [np.sqrt(flux + np.square(obj_params['RON'])) for flux in flx]

                    if np.isnan(np.sum(flx_err_ron)):
                        if print_stat:
                            print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')

                    # Slicing the data to contain only the region required for the index calculation as explained above and 
                    # creating a spectrum class for it.

                    spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                        flux=flx[left_idx:right_idx+1]*u.Jy,
                                        uncertainty=StdDevUncertainty(flx_err_ron[left_idx:right_idx+1], unit=u.Jy))
                
                else:
                    if print_stat:
                        print('File contains NaN in flux errors array and could not extract the ReadOut Noise (RON) from FITS file header.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        print('Approximating flux errors as the photon noise instead.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
                    with warnings.catch_warnings():  # Ignore warnings
                        warnings.simplefilter('ignore')
                        flx_err_pn = [np.sqrt(flux) for flux in flx]

                    if np.isnan(np.sum(flx_err_pn)):
                        if print_stat:
                            print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')

                    # Slicing the data to contain only the region required for the index calculation as explained above and 
                    # creating a spectrum class for it.

                    spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                        flux=flx[left_idx:right_idx+1]*u.Jy,
                                        uncertainty=StdDevUncertainty(flx_err_pn[left_idx:right_idx+1], unit=u.Jy))
                
            else:
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx+1]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
                
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
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            with warnings.catch_warnings(): # Ignore warnings
                warnings.simplefilter('ignore')
                flx_err = [np.sqrt(flux) for flux in flx] # Using only photon noise as flx_err approx since no RON info available!
            
            # Slicing the data to contain only the region required for the index calculation as explained above and creating 
            # a spectrum class for it
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx+1]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
            
        # SOPHIE
        
        elif Instrument == 'SOPHIE':
            
            obj_params, spec = read_data(file_path=file_path[i],
                                         Instrument=Instrument,
                                         print_stat=print_stat,
                                         show_plots=False)
            
            obj_params['RV'] = radial_velocity 
            
            # Checking if the FITS file is e2ds since it has 39 spectral orders using an arbitray order number of 50. If greater than 50, assume its s1d.
            
            if len(spec[0]) < 50:
                
                if print_stat:
                    print('Total {} spectral orders extracted'.format(len(spec[0])))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        
                wvl_K = spec[0][0] ## For SOPHIE spectra, the CaII K line is within the first spectral order.
                flx_K = spec[1][0]
                
                wvl_H = spec[0][1] ## The CaII H line is within the second spectral order.
                flx_H = spec[1][1]
                
                wvl_F2 = spec[0][2] ## The complete F2 line is within the third spectral order.
                flx_F2 = spec[1][2]
                    
                # Flux error array is calculated as photon noise plus CCD readout noise 

                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err_K = np.asarray([np.sqrt(flux + np.square(obj_params['RON'])) for flux in flx_K])
                    flx_err_H = np.asarray([np.sqrt(flux + np.square(obj_params['RON'])) for flux in flx_H])
                    flx_err_F2 = np.asarray([np.sqrt(flux + np.square(obj_params['RON'])) for flux in flx_F2])
                    
                if np.isnan(np.sum(flx_err_K)):
                    if print_stat:
                        print('The calculated flux error array for the CaII K order contains a few NaN values due to negative flux encountered in the square root.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                if np.isnan(np.sum(flx_err_H)):
                    if print_stat:
                        print('The calculated flux error array for the CaII H order contains a few NaN values due to negative flux encountered in the square root.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        
                if np.isnan(np.sum(flx_err_F2)):
                    if print_stat:
                        print('The calculated flux error array for the F2 line order contains a few NaN values due to negative flux encountered in the square root.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        
                if print_stat:
                    print('The CaII K order wavelength read from .fits file is: {}'.format(wvl_K))
                    print('The CaII K order flux read from .fits file is: {}'.format(flx_K))
                    print('The CaII K order calculated flux error is: {}'.format(flx_err_K))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    print('The CaII H order wavelength read from .fits file is: {}'.format(wvl_H))
                    print('The CaII H order flux read from .fits file is: {}'.format(flx_H))
                    print('The CaII H order calculated flux error is: {}'.format(flx_err_H))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    print('The F2 line order wavelength read from .fits file is: {}'.format(wvl_F2))
                    print('The F2 line order flux read from .fits file is: {}'.format(flx_F2))
                    print('The F2 line order calculated flux error is: {}'.format(flx_err_F2))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')

                # The spectra is now doppler shift corrected in the wavelength axis using the stellar radial velocity and 
                # the rest wavelength of CaIIH line; delta_lambda = (v/c)*lambda
    
                shift = ((radial_velocity/ap.constants.c.value)*CaIIH_line)
                shift = (round(shift, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in the .s files!
    
                wvl_K = np.round((wvl_K - shift), 4)
                wvl_H = np.round((wvl_H - shift), 4)
                wvl_F2 = np.round((wvl_F2 - shift), 4)
                
                # Creating a spectrum object called 'spec1d' using 'Spectrum1D' from 'specutils' for each line
                # Docs for 'specutils' are here; https://specutils.readthedocs.io/en/stable/ 
                
                # The spectral and flux axes are given units nm and Jy respectively using 'astropy.units'. 
                # The uncertainty has units Jy as well!
                
                spec1d_K = Spectrum1D(spectral_axis=wvl_K*u.nm, 
                                      flux=flx_K*u.Jy, 
                                      uncertainty=StdDevUncertainty(flx_err_K, unit=u.Jy))
    
                spec1d_H = Spectrum1D(spectral_axis=wvl_H*u.nm, 
                                      flux=flx_H*u.Jy, 
                                      uncertainty=StdDevUncertainty(flx_err_H, unit=u.Jy))
                
                spec1d_F2 = Spectrum1D(spectral_axis=wvl_F2*u.nm, 
                                      flux=flx_F2*u.Jy, 
                                      uncertainty=StdDevUncertainty(flx_err_F2, unit=u.Jy))
                
                spec1d = [spec1d_K, spec1d_H, spec1d_F2] # Creating a list containing all Spectrum1D objects
                
            else:
                
                left_idx = find_nearest(spec[0], F1_line-2) # ± 2nm extra included for both!
                right_idx = find_nearest(spec[0], F2_line+2)
                
                # Slicing the data to contain only the region required for the index calculation
                
                wvl = spec[0][left_idx:right_idx+1]
                flx = spec[1][left_idx:right_idx+1]
                
                # Flux error array is calculated as photon noise alone since RON isn't available

                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err = np.asarray([np.sqrt(flux) for flux in flx])
                    
                
                if np.isnan(np.sum(flx_err)):
                    if print_stat:
                        print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        
                if print_stat:
                    print('The wavelength array read from .fits file is: {}'.format(wvl))
                    print('The flux array read from .fits file is: {}'.format(flx))
                    print('The calculated flux error array is: {}'.format(flx_err))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')    
                    
                
                shift = ((obj_params['RV']/ap.constants.c.value)*CaIIH_line)  
                shift = (round(shift, 4)) 
                
                wvl_shifted = np.round((wvl - shift), 4) 
                
                spec1d = Spectrum1D(spectral_axis=wvl_shifted*u.nm, 
                                    flux=flx*u.Jy, 
                                    uncertainty=StdDevUncertainty(flx_err, unit=u.Jy))

        else:
            raise ValueError('Instrument type not recognisable. Available options are "NARVAL", "ESPADONS", "HARPS", "HARPS-N" and "SOPHIE"')
            
        # Creating two events for further analysis. One for cases where spectrum is 1 dimensional, and one for where the spectrum has individual orders!
            
        if type(spec1d) == list:
            
            # Printing spec info
            
            if print_stat:
                print('The doppler shift size using RV {} m/s and the CaIIH line of 396.847nm is: {}nm'.format(radial_velocity, shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The CaII K order spectral axis ranges from {:.4f}nm to {:.4f}nm.'.format(spec1d[0].spectral_axis[0].value, spec1d[0].spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The CaII H order spectral axis ranges from {:.4f}nm to {:.4f}nm.'.format(spec1d[1].spectral_axis[0].value, spec1d[1].spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                if len(spec1d) == 3: ## This is ONLY in the case of the SOPHIE Instrument
                    print('The F2 line order spectral axis ranges from {:.4f}nm to {:.4f}nm.'.format(spec1d[2].spectral_axis[0].value, spec1d[2].spectral_axis[-1].value))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                print('These values are doppler shift corrected and rounded off to 4 decimal places')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            
            # Fitting an nth order polynomial to the continuum for normalisation using specutils
                
            if norm_spec:
                if print_stat:
                    print('Normalising spectral orders by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                # Note the continuum normalized spectrum also has new uncertainty values!
                
                spec_K = normalise_spec(spec1d[0], degree, F1_line, F1_band, CaIIK_line, CaIIK_band,
                                        print_stat, plot_fit, save_figs, save_figs_name) 
                
                if len(spec1d) == 3:
                    
                    spec_H = normalise_spec(spec1d[1], degree, CaIIH_line, CaIIH_band, CaIIH_line, CaIIH_band,
                                            print_stat, plot_fit, save_figs, save_figs_name)
                    
                    spec_F2 = normalise_spec(spec1d[2], degree, F2_line, F2_band, F2_line, F2_band,
                                             print_stat, plot_fit, save_figs, save_figs_name)
                    
                else:
                    
                    spec_H = normalise_spec(spec1d[1], degree, CaIIH_line, CaIIH_band, F2_line, F2_band,
                                            print_stat, plot_fit, save_figs, save_figs_name)
                
                
            else:
                spec_K = spec1d[0]
                spec_H = spec1d[1]
                
                if len(spec1d) == 3:
                    spec_F2 = spec1d[2]
                
                
            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            
            if plot_spec:
                
                if len(spec1d) == 3:
                    
                    wvl_HK = np.asarray(list(spec_K.spectral_axis.value) + list(spec_H.spectral_axis.value) + list(spec_F2.spectral_axis.value))
                    flx_HK = np.asarray(list(spec_K.flux.value) + list(spec_H.flux.value) + list(spec_F2.flux.value))
                    flx_err_HK = np.asarray(list(spec_K.uncertainty.array) + list(spec_H.uncertainty.array) + list(spec_F2.uncertainty.array))
                    
                else:
                    
                    wvl_HK = np.asarray(list(spec_K.spectral_axis.value) + list(spec_H.spectral_axis.value))
                    flx_HK = np.asarray(list(spec_K.flux.value) + list(spec_H.flux.value))
                    flx_err_HK = np.asarray(list(spec_K.uncertainty.array) + list(spec_H.uncertainty.array))
                
                # This Spectrum1D object is used for plotting ONLY!
                
                spec1d_HK = Spectrum1D(spectral_axis=wvl_HK*u.nm, 
                                       flux=flx_HK*u.Jy, 
                                       uncertainty=StdDevUncertainty(flx_err_HK, unit=u.Jy))
                
                lines = [CaIIH_line, CaIIH_band, CaIIK_line, CaIIK_band, F1_line, F1_band, F2_line, F2_band]
                
                plot_spectrum(spec1d_HK, lines, 'CaIIHK', Instrument, norm_spec, save_figs, save_figs_name)
                
            if plot_only_spec:
                
                pass
            
            else:
            
                # Extracting the CaIIK line region using the given bandwidth 'CaIIK_band'
                F_CaIIK_region = extract_region(spec_K, region=SpectralRegion((CaIIK_line-(CaIIK_band/2))*u.nm, (CaIIK_line+(CaIIK_band/2))*u.nm))
                
                # Extracting the CaIIH line region using the given bandwidth 'CaIIH_band'
                F_CaIIH_region = extract_region(spec_H, region=SpectralRegion((CaIIH_line-(CaIIH_band/2))*u.nm, (CaIIH_line+(CaIIH_band/2))*u.nm))
                
                # Doing the same for the reference continuum regions!
                F1_region = extract_region(spec_K, region=SpectralRegion((F1_line-(F1_band/2))*u.nm, (F1_line+(F1_band/2))*u.nm))
                
                if len(spec1d) == 3:
                    F2_region = extract_region(spec_F2, region=SpectralRegion((F2_line-(F2_band/2))*u.nm, (F2_line+(F2_band/2))*u.nm))
                else:
                    F2_region = extract_region(spec_H, region=SpectralRegion((F2_line-(F2_band/2))*u.nm, (F2_line+(F2_band/2))*u.nm))
            
            
        else:
            
            # Printing spec info
                
            if print_stat:
                print('The doppler shift size using RV {} m/s and the CaIIH line of 396.847nm is: {}nm'.format(radial_velocity, shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral axis ranges from {:.4f}nm to {:.4f}nm.'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('These values are doppler shift corrected and rounded off to 4 decimal places')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            
            # Fitting an nth order polynomial to the continuum for normalisation using specutils
                
            if norm_spec:
                if print_stat:
                    print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                # Note the continuum normalized spectrum also has new uncertainty values!
                
                spec = normalise_spec(spec1d, degree, F1_line, F1_band, F2_line, F2_band,
                                      print_stat, plot_fit, save_figs, save_figs_name) 
                
            else:
                spec = spec1d
                
            # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
            
            if plot_spec:
                
                lines = [CaIIH_line, CaIIH_band, CaIIK_line, CaIIK_band, F1_line, F1_band, F2_line, F2_band]
                
                plot_spectrum(spec, lines, 'CaIIHK', Instrument, norm_spec, save_figs, save_figs_name)
                
            if plot_only_spec:
                
                pass
            
            else:
            
                # Extracting the CaIIH line region using the given bandwidth 'CaIIH_band'
                F_CaIIH_region = extract_region(spec, region=SpectralRegion((CaIIH_line-(CaIIH_band/2))*u.nm, (CaIIH_line+(CaIIH_band/2))*u.nm))
                
                # Extracting the CaIIK line region using the given bandwidth 'CaIIK_band'
                F_CaIIK_region = extract_region(spec, region=SpectralRegion((CaIIK_line-(CaIIK_band/2))*u.nm, (CaIIK_line+(CaIIK_band/2))*u.nm))
                
                # Doing the same for the reference continuum regions!
                F1_region = extract_region(spec, region=SpectralRegion((F1_line-(F1_band/2))*u.nm, (F1_line+(F1_band/2))*u.nm))
                F2_region = extract_region(spec, region=SpectralRegion((F2_line-(F2_band/2))*u.nm, (F2_line+(F2_band/2))*u.nm))
            
        if plot_only_spec:
            
            pass
        
        else:
        
            # Creating a list containing the extracted regions 
            
            regions = [F_CaIIK_region, F_CaIIH_region, F1_region, F2_region]
            
            # Calculating the index using 'calc_inc' from krome.spec_analysis
            
            I_CaIIHK, I_CaIIHK_err = calc_ind(regions=regions,
                                              index_name='CaIIHK',
                                              print_stat=print_stat)
                
            if Instrument=='NARVAL':
                if out_file_path != None:
                    header = ['HJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'NUM_EXP', 'GAIN', 'RON', 'V_mag', 'T_eff', 'RV', 'I_CaIIHK', 'I_CaIIHK_err']
                    res = list(obj_params.values()) + [I_CaIIHK, I_CaIIHK_err] # Creating results list 'res' containing the calculated parameters and appending this list to the 
                                                                               # 'results' empty list created at the start of this function!
                    results.append(res)
                else:
                    header = ['I_CaIIHK', 'I_CaIIHK_err']
                    res = [I_CaIIHK, I_CaIIHK_err]
                    results.append(res)
                    
            elif Instrument=='ESPADONS':
                if meta_file_path != None:
                    header = ['OBS_DATE', 'RA', 'DEC', 'V_mag', 'T_eff', 'Distance', 'JD', 'AIRMASS', 'T_EXP', 'RUN_ID', 'SNR', 'RV', 'I_CaIIHK', 'I_CaIIHK_err']
                    res = list(obj_params.values()) + [I_CaIIHK, I_CaIIHK_err] 
                    results.append(res)
                else:
                    header = ['I_CaIIHK', 'I_CaIIHK_err']
                    res = [I_CaIIHK, I_CaIIHK_err]
                    results.append(res)
            
            elif Instrument=='HARPS':
                header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'BERV', 'OBS_DATE', 'PROG_ID', 'SNR', 'SIGDET', 'CONAD', 'RON', 'RV', 'I_CaIIHK', 'I_CaIIHK_err']
                res = list(obj_params.values()) + [I_CaIIHK, I_CaIIHK_err]
                results.append(res)
                
            elif Instrument=='HARPS-N':
                header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'OBS_DATE', 'PROG_ID', 'RV', 'I_CaIIHK', 'I_CaIIHK_err']
                res = list(obj_params.values()) + [I_CaIIHK, I_CaIIHK_err]
                results.append(res)
                
            elif Instrument=='SOPHIE':
                header = ['JD', 'RA', 'DEC', 'T_EXP', 'OBS_DATE', 'PROG_ID', 'SIGDET', 'CONAD', 'RON', 'RV', 'I_CaIIHK', 'I_CaIIHK_err']
                res = list(obj_params.values()) + [I_CaIIHK, I_CaIIHK_err]
                results.append(res)
            
    if plot_only_spec:
        
        return
    
    else:
    
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

## Defining a function for calculating the HeI D3 index following Gomes da Silva et al. 2011 (2011A&A...534A..30G)

def HeI_index(file_path,
              radial_velocity,
              degree=4,
              HeI_line=587.562,
              HeI_band=0.04,
              F1_line=586.9,
              F1_band=0.5,
              F2_line=588.1, 
              F2_band=0.5,
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
              plot_only_spec=False):
    
    """
    Calculates the HeI index following Gomes da Silva et al. 2011 (2011A&A...534A..30G).  
    NOTE: This paper uses the HeI index method described by Boisse et al. 2009 for a K dwarf 
    but, with different reference continuum lines and bands.
    
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
    
    HeI_line: int, default: 587.562 nm
    HeI line centre in nm.
    
    HeI_band: int, default: 0.04 nm
    Band width (nm) in which to calculate the mean flux.
    
    F1_line: int, default: 586.9 nm
    Line centre of the blue reference continuum.
    
    F1_band: int, default: 0.5 nm
    Band width (nm) in which to calculate the mean continuum flux.
    
    F2_line: int, default: 588.1 nm
    Line centre of the red reference continuum.
    
    F2_band: int, default: 0.5 nm
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
    
    
    Returns:
    -----------
    NARVAL: HJD, RA, DEC, AIRMASS, Exposure time[s], No. of exposures, GAIN [e-/ADU], ReadOut Noise [e-], V_mag, T_eff[K], RV[m/s], HeI index, error on HeI index
    HARPS: BJD, RA, DEC, AIRMASS, Exposure time[s], Barycentric RV[km/s], OBS_DATE, Program ID, SNR, CCD Readout Noise[e-], CCD conv factor[e-/ADU], ReadOut Noise[ADU], RV[m/s], HeI index, error on HeI index
    HARPS-N: BJD, RA, DEC, AIRMASS, Exposure time[s], OBS_DATE, Program ID', RV[m/s], HeI index, error on HeI index
    
    All values are type float() given inside a list.
    
    """
    
    results = [] # Empty list to which the run results will be appended
    
    # Creating a loop to go through each given file_path in the list of file paths
    
    # Using the tqdm function 'log_progress' to provide a neat progress bar in Jupyter Notebook which shows the total number of
    # runs, the run time per iteration and the total run time for all files!
    
    for i in log_progress(range(len(file_path)), desc='Calculating HeID3 Index'):
        
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
                    
            
            order_38 = orders[61-38] # The orders begin from # 61 so to get # 38, we index as 61-38.
            
            if print_stat:
                print('The #38 order wavelength read from .s file using pandas is: {}'.format(order_38[0].values))
                print('The #38 order intensity read from .s file using pandas is: {}'.format(order_38[1].values))
                print('The #38 order intensity error read from .s file using pandas is: {}'.format(order_38[2].values))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
            
            # The spectra is now doppler shift corrected in the wavelength axis using the stellar radial velocity and the rest wavelength of H alpha line; delta_lambda = (v/c)*lambda
            
            shift = ((radial_velocity/ap.constants.c.value)*HeI_line)  
            shift = (round(shift, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in the .s files!
            
            wvl = np.round((order_38[0].values - shift), 4) # Subtracting the calculated doppler shift value from the wavelength axis since the stellar radial velocity is positive. If the stellar RV is negative, the shift value will be added instead.
            flx = order_38[1].values # Indexing flux array from order_34
            flx_err = order_38[2].values # Indexing flux_err array from order_34
            
            # Creating a spectrum object called 'spec1d' using 'Spectrum1D' from 'specutils'
            # Docs for 'specutils' are here; https://specutils.readthedocs.io/en/stable/ 
            
            # The spectral and flux axes are given units nm and Jy respectively using 'astropy.units'. 
            # The uncertainty has units Jy as well!
        
            spec1d = Spectrum1D(spectral_axis=wvl*u.nm, 
                                flux=flx*u.Jy, 
                                uncertainty=StdDevUncertainty(flx_err, unit=u.Jy)) 
                
                
        # ESPADONS
        
        elif Instrument == 'ESPADONS':
            
            if meta_file_path != None:
                
                # Using read_data from krome.spec_analysis to extract useful object parameters and all individual spectral orders
                
                obj_params, orders = read_data(file_path=file_path[i],
                                               meta_file_path=meta_file_path[i],
                                               Instrument=Instrument,
                                               print_stat=print_stat,
                                               show_plots=False)
                
                obj_params['RV'] = radial_velocity # setting radial_velocity as part of the obj_params dictionary for continuity 
                
            else:
                
                orders = read_data(file_path=file_path[i],
                                   Instrument=Instrument,
                                   print_stat=print_stat,
                                   meta_file_path=None,
                                   show_plots=False)
                
                if print_stat:
                    print('"meta_file_path" not given as an argument. Run will only return the indices and their errros instead.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                

            if print_stat:
                print('Total {} spectral orders extracted'.format(len(orders)))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
            
            order_38 = orders[61-38] 
            
            if print_stat:
                print('The #38 order wavelength read from .s file using pandas is: {}'.format(order_38[0]))
                print('The #38 order intensity read from .s file using pandas is: {}'.format(order_38[1]))
                print('The #38 order intensity error read from .s file using pandas is: {}'.format(order_38[2]))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
            shift = ((radial_velocity/ap.constants.c.value)*HeI_line)  
            shift = (round(shift, 4))
            
            wvl = np.round((order_34[0] - shift), 4) 
            flx = order_34[1] 
            flx_err = order_34[2] 
        
            spec1d = Spectrum1D(spectral_axis=wvl*u.nm, 
                                flux=flx*u.Jy, 
                                uncertainty=StdDevUncertainty(flx_err, unit=u.Jy)) 
                
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
           
            shift = ((obj_params['RV']/ap.constants.c.value)*HeI_line)  
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
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx+1]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err_ron[left_idx:right_idx+1], unit=u.Jy))
                
            else:
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx+1]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
            
             
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
            
            shift = ((obj_params['RV']/ap.constants.c.value)*HeI_line)  
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
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx+1]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
            
                
        # SOPHIE
        
        elif Instrument == 'SOPHIE':
            
            obj_params, spec = read_data(file_path=file_path[i],
                                         Instrument=Instrument,
                                         print_stat=print_stat,
                                         show_plots=False)
            
            obj_params['RV'] = radial_velocity 
            
            # Checking if the FITS file is e2ds since it has 39 spectral orders using an arbitray order number of 50. If greater than 50, assume its s1d.
            
            if len(spec[0]) < 50:
                
                if print_stat:
                    print('Total {} spectral orders extracted'.format(len(spec[0])))
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                        
                wvl = spec[0][29] ## For SOPHIE spectra, the HeI line along with its reference bands are within the 30th order.
                flx = spec[1][29]
                    
                # Flux error array is calculated as photon noise plus CCD readout noise 

                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err = np.asarray([np.sqrt(flux + np.square(obj_params['RON'])) for flux in flx])
                
            else:
                
                left_idx = find_nearest(spec[0], F1_line-2) # ± 2nm extra included for both!
                right_idx = find_nearest(spec[0], F2_line+2)
                
                # Slicing the data to contain only the region required for the index calculation
                
                wvl = spec[0][left_idx:right_idx+1]
                flx = spec[1][left_idx:right_idx+1]
                
                # Flux error array is calculated as photon noise alone since RON isn't available

                with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err = np.asarray([np.sqrt(flux) for flux in flx])
                    
                
            if np.isnan(np.sum(flx_err)):
                if print_stat:
                    print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
            if print_stat:
                print('The wavelength array read from .fits file is: {}'.format(wvl))
                print('The flux array read from .fits file is: {}'.format(flx))
                print('The calculated flux error array is: {}'.format(flx_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')    
                
            
            shift = ((obj_params['RV']/ap.constants.c.value)*HeI_line)  
            shift = (round(shift, 4)) 
            
            wvl_shifted = np.round((wvl - shift), 4) 
            
            spec1d = Spectrum1D(spectral_axis=wvl_shifted*u.nm, 
                                flux=flx*u.Jy, 
                                uncertainty=StdDevUncertainty(flx_err, unit=u.Jy))
            
        # ELODIE
        
        elif Instrument=='ELODIE':
            
            obj_params, spec = read_data(file_path=file_path[i],
                                         Instrument=Instrument,
                                         print_stat=print_stat,
                                         show_plots=False)
            
            obj_params['RV'] = radial_velocity
            
            wvl = spec[0] # nm
            flx = spec[1] # ADU
            
            shift = ((obj_params['RV']/ap.constants.c.value)*HeI_line)  
            shift = (round(shift, 3)) 
            
            left_idx = find_nearest(wvl, F1_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, F2_line+2)
            
            if print_stat:
                print('Calculating the flux error array as the photon noise')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            with warnings.catch_warnings():  # Ignore warnings
                    warnings.simplefilter('ignore')
                    flx_err = np.asarray([np.sqrt(flux) for flux in flx]) 
                    
            if np.isnan(np.sum(flx_err)):
                if print_stat:
                    print('The calculated flux error array contains a few NaN values due to negative flux encountered in the square root.')
                    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    
            if print_stat:
                print('The wavelength array read from the .fits file is: {}'.format(wvl))
                print('The flux array read from the .fits file is: {}'.format(flx))
                print('The calculated flux error array is: {}'.format(flx_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx+1]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
                
        else:
            raise ValueError('Instrument type not recognised. Available options are "NARVAL", "ESPADONS", "HARPS", "HARPS-N", "SOPHIE" and "ELODIE"')
            
        # Printing spec info
            
        if print_stat:
            print('The doppler shift size using RV {} m/s and the HeID3 line of 587.562nm is: {:.4f}nm'.format(radial_velocity, shift))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('The spectral axis ranges from {:.4f}nm to {:.4f}nm.'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('These values are doppler shift corrected and rounded off to 4 decimal places')
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        # Fitting an nth order polynomial to the continuum for normalisation using specutils
            
        if norm_spec:
            if print_stat:
                print('Normalising the spectra by fitting a {}th order polynomial to the enitre spectral order'.format(degree))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            # Note the continuum normalized spectrum also has new uncertainty values!
            
            spec = normalise_spec(spec1d, degree, F1_line, F1_band, F2_line, F2_band,
                                  print_stat, plot_fit, save_figs, save_figs_name) 
            
        else:
            spec = spec1d
            
            
        # Plots the final reduced spectra along with the relevant bandwidths and line/continuum positions
        
        if plot_spec:
            
            lines = [HeI_line, HeI_band, F1_line, F1_band, F2_line, F2_band]
            
            plot_spectrum(spec, lines, 'HeI', Instrument, norm_spec, save_figs, save_figs_name)
            
        if plot_only_spec:
            
            pass
        
        else:
            
            # Now we have the spectrum to work with as a variable, 'spec'!
            
            # The three regions required for HeI index calculation are extracted from 'spec' using the 'extract region' function from 'specutils'. 
            # The function uses another function called 'SpectralRegion' as one of its arguments which defines the region to be extracted done so using the line and line 
            # bandwidth values; i.e. left end of region would be 'line - bandwidth/2' and right end would be 'line + bandwidth/2'.
            # Note: These values must have the same units as the spec wavelength axis.
            
            F_HeI_region = extract_region(spec, region=SpectralRegion((HeI_line-(HeI_band/2))*u.nm, (HeI_line+(HeI_band/2))*u.nm))
            F1_region = extract_region(spec, region=SpectralRegion((F1_line-(F1_band/2))*u.nm, (F1_line+(F1_band/2))*u.nm))
            F2_region = extract_region(spec, region=SpectralRegion((F2_line-(F2_band/2))*u.nm, (F2_line+(F2_band/2))*u.nm))
            
            regions = [F_HeI_region, F1_region, F2_region]
                
            # The indices are calculated using the 'calc_ind' function from krome.spec_analysis by inputting the extracted regions as shown below;
            
            I_HeI, I_HeI_err = calc_ind(regions=regions,
                                        index_name='HeI',
                                        print_stat=print_stat)
            
            if Instrument=='NARVAL':
                if out_file_path != None:
                    header = ['HJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'NUM_EXP', 'GAIN', 'RON', 'V_mag', 'T_eff', 'RV', 'I_HeI', 'I_HeI_err']
                    res = list(obj_params.values()) + [I_HeI, I_HeI_err] # Creating results list 'res' containing the calculated parameters and appending this list to the 
                                                                         # 'results' empty list created at the start of this function!
                    results.append(res)
                else:
                    header = ['I_HeI', 'I_HeI_err']
                    res = [I_HeI, I_HeI_err]
                    results.append(res)
                    
            elif Instrument=='ESPADONS':
                if meta_file_path != None:
                    header = ['OBS_DATE', 'RA', 'DEC', 'V_mag', 'T_eff', 'Distance', 'JD', 'AIRMASS', 'T_EXP', 'RUN_ID', 'SNR', 'RV', 'I_HeI', 'I_HeI_err']
                    res = list(obj_params.values()) + [I_HeI, I_HeI_err]
                    results.append(res)
                else:
                    header = ['I_HeI', 'I_HeI_err']
                    res = [I_HeI, I_HeI_err]
                    results.append(res)
            
            elif Instrument=='HARPS':
                header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'BERV', 'OBS_DATE', 'PROG_ID', 'SNR', 'SIGDET', 'CONAD', 'RON', 'RV', 'I_HeI', 'I_HeI_err']
                res = list(obj_params.values()) + [I_HeI, I_HeI_err]
                results.append(res)
                
            elif Instrument=='HARPS-N':
                header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'OBS_DATE', 'PROG_ID', 'RV', 'I_HeI', 'I_HeI_err']
                res = list(obj_params.values()) + [I_HeI, I_HeI_err]
                results.append(res)
                
            elif Instrument=='SOPHIE':
                header = ['JD', 'RA', 'DEC', 'T_EXP', 'OBS_DATE', 'PROG_ID', 'SIGDET', 'CONAD', 'RON', 'RV', 'I_HeI', 'I_HeI_err']
                res = list(obj_params.values()) + [I_HeI, I_HeI_err]
                results.append(res)
                
            elif Instrument=='ELODIE':
                header = ['JD', 'RA', 'DEC', 'T_EXP', 'OBS_DATE', 'AIRMASS', 'SNR', 'GAIN', 'RV', 'I_HeI', 'I_HeI_err']
                res = list(obj_params.values()) + [I_HeI, I_HeI_err]
                results.append(res)
                
    if plot_only_spec:
        
        return
    
    else:
    
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

## Defining a function to calculate the Balmer decrement Frasca, A. et al. 2015 (2015A&A...575A...4F)

def balmer_decrement(file_path,
                     radial_velocity,
                     H_alpha_line=656.2808,
                     H_alpha_band=0.1,
                     H_beta_line=486.135,
                     H_beta_band=0.1, 
                     Instrument='NARVAL',
                     plot_spec=True,
                     print_stat=True,
                     save_results=False, 
                     results_file_name=None,
                     save_figs=False,
                     save_figs_name=None,
                     out_file_path=None,
                     ccf_file_path=None):
    
    """
    
    Calculates the H_alpha/H_beta flux ratio, the so-called Balmer Decrement, following Frasca, A. et al. 2015 (2015A&A...575A...4F).
    NOTE: For first papers about the Balmer Decrement, see Landamn & Mongillo 1979 (Landman, D. A & Mongillo, M. 1979, ApJ, 230, 581) 
    or Chester 1991 (Chester, M. M. 1991, Ph. D. Thesis, Pennsylvania State Univ.)
    
    
    Parameters:
    -----------
    
    file_path: list, .s format (NARVAL), ADP..._.fits format (HARPS) or s1d_A.fits format (HARPS-N)
    List containng the paths of the spectrum files 
    
    radial_velocity: int
    Stellar radial velocity along the line-of-sight. This value is used for doppler shifting the spectra to its rest frame.
    
    H_alpha_line: int, default: 656.2808 nm
    H alpha line centre in nm.
    
    H_alpha_band: int, default: 0.1 nm
    Band width (nm) in which to calculate the mean flux.
    
    H_beta_line: int, default: 486.135 nm
    H beta line centre in nm. See https://physics.nist.gov/PhysRefData/ASD/lines_form.html for Balmer series lines.
    
    H_beta_band: int, default: 0.1 nm
    Band width (nm) in which to calculate the mean flux.
    
    Instrument: str, default: 'NARVAL'
    The instrument from which the data has been collected. Available options are 'NARVAL', 'HARPS' or 'HARPS-N'.
    
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
    
    ccf_file_path: list, .fits format (HARPS), default: None
    List containig the paths of the CCF FITS files to extract the radial velocity. If None, the given radial velocity argument is used for all files for doppler shift corrections
    
    
    Returns:
    -----------
    NARVAL: HJD, RA, DEC, AIRMASS, Exposure time[s], No. of exposures, GAIN [e-/ADU], ReadOut Noise [e-], V_mag, T_eff[K], RV[m/s], H_alpha mean flux, H_alpha mean flux error, H_beta mean flux, H_beta mean flux error, balmer decrement, error on balmer decrement
    HARPS: BJD, RA, DEC, AIRMASS, Exposure time[s], Barycentric RV[km/s], OBS_DATE, Program ID, SNR, CCD Readout Noise[e-], CCD conv factor[e-/ADU], ReadOut Noise[ADU], RV[m/s], H_alpha mean flux, H_alpha mean flux error, H_beta mean flux, H_beta mean flux error, balmer decrement, error on balmer decrement
    
    All values are type float() given inside a list.
    
    """
    
    results = [] # Empty list to which the run results will be appended
    
    # Creating a loop to go through each given file_path in the list of file paths
    
    # Using the tqdm function 'log_progress' to provide a neat progress bar in Jupyter Notebook which shows the total number of
    # runs, the run time per iteration and the total run time for all files!
    
    for i in log_progress(range(len(file_path)), desc='Calculating Balmer Decrement'):
        
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
                    
            
            order_34 = orders[61-34] # The orders begin from # 61 so to get # 34, we index as 61-34. This will be tha H_alpha order
            order_46 = orders[61-46] # This will be the H_beta order.
            
            if print_stat:
                print('The #34 Hα order wavelength read from .s file using pandas is: {}'.format(order_34[0].values))
                print('The #34 Hα order intensity read from .s file using pandas is: {}'.format(order_34[1].values))
                print('The #34 Hα order intensity error read from .s file using pandas is: {}'.format(order_34[2].values))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The #46 Hβ order wavelength read from .s file using pandas is: {}'.format(order_46[0].values))
                print('The #46 Hβ order intensity read from .s file using pandas is: {}'.format(order_46[1].values))
                print('The #46 Hβ order intensity error read from .s file using pandas is: {}'.format(order_46[2].values))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
            # The spectra is now doppler shift corrected in the wavelength axis using the stellar radial velocity and the rest wavelength of H_alpha/H_beta line; delta_lambda = (v/c)*lambda. NOTE: The line shifts are used for the H_alpha/H_beta spectral orders respectively.
            
            shift_alpha = ((radial_velocity/ap.constants.c.value)*H_alpha_line)  
            shift_alpha = (round(shift_alpha, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in the .s files!
            
            shift_beta = ((radial_velocity/ap.constants.c.value)*H_beta_line)  
            shift_beta = (round(shift_beta, 4))
            
            wvl_alpha = np.round((order_34[0].values - shift_alpha), 4) # Subtracting the calculated doppler shift value from the wavelength axis since the stellar radial velocity is positive. If the stellar RV is negative, the shift value will be added instead.
            flx_alpha = order_34[1].values # Indexing flux array from order_34
            flx_err_alpha = order_34[2].values # Indexing flux_err array from order_34
            
            wvl_beta = np.round((order_46[0].values - shift_beta), 4) 
            flx_beta = order_46[1].values 
            flx_err_beta = order_46[2].values
            
            # Creating a spectrum object called 'spec1d' using 'Spectrum1D' from 'specutils'
            # Docs for 'specutils' are here; https://specutils.readthedocs.io/en/stable/ 
            
            # The spectral and flux axes are given units nm and Jy respectively using 'astropy.units'. 
            # The uncertainty has units Jy as well!
        
            spec1d_alpha = Spectrum1D(spectral_axis=wvl_alpha*u.nm, 
                                      flux=flx_alpha*u.Jy, 
                                      uncertainty=StdDevUncertainty(flx_err_alpha, unit=u.Jy)) 
            
            spec1d_beta = Spectrum1D(spectral_axis=wvl_beta*u.nm, 
                                      flux=flx_beta*u.Jy, 
                                      uncertainty=StdDevUncertainty(flx_err_beta, unit=u.Jy))
            
            # Printing info
            
            if print_stat:
                print('The doppler shift size using RV {} m/s and the Hα/Hβ line of 656.2808/486.135 nm is: {}/{}nm'.format(radial_velocity, shift_alpha, shift_beta))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The Hα/Hβ orders used range from {}/{}nm to {}/{}nm. These values are doppler shift corrected and rounded off to 4 decimal places'.format(spec1d_alpha.spectral_axis[0].value, spec1d_beta.spectral_axis[0].value, spec1d_alpha.spectral_axis[-1].value, spec1d_beta.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            # Calculating the fluxes
            
            ## H_alpha
            
            # Using the 'find_nearest' function from 'spec_analysis.py' to find the left and right indices 
            # which are used to slice the spectrum in the H_alpha region alone.
            
            left_idx_alpha = find_nearest(spec1d_alpha.spectral_axis.value, H_alpha_line-(H_alpha_band/2)) 
            right_idx_alpha = find_nearest(spec1d_alpha.spectral_axis.value, H_alpha_line+(H_alpha_band/2))
                                                   
            H_alpha_line_region = [spec1d_alpha.spectral_axis.value[left_idx_alpha:right_idx_alpha+1], # Respective arrays are sliced using the indices above. 
                                   spec1d_alpha.flux.value[left_idx_alpha:right_idx_alpha+1],          # Using right_idx_alpha+1 to include the final element in the array.
                                   spec1d_alpha.uncertainty.array[left_idx_alpha:right_idx_alpha+1]]
            
            H_alpha_line_mean = np.round(np.mean(H_alpha_line_region[1]), 4) # Calculating mean of the flux within this region.
            
            # Calculating the error on the mean flux calculated above using error propagation.
            H_alpha_line_sum_err = [i**2 for i in H_alpha_line_region[2]]
            H_alpha_line_mean_err = np.round((np.sqrt(np.sum(H_alpha_line_sum_err))/len(H_alpha_line_sum_err)), 4)
            
            ## H_beta
            
            left_idx_beta = find_nearest(spec1d_beta.spectral_axis.value, H_beta_line-(H_beta_band/2)) 
            right_idx_beta = find_nearest(spec1d_beta.spectral_axis.value, H_beta_line+(H_beta_band/2))
                                                   
            H_beta_line_region = [spec1d_beta.spectral_axis.value[left_idx_beta:right_idx_beta+1], 
                                  spec1d_beta.flux.value[left_idx_beta:right_idx_beta+1], 
                                  spec1d_beta.uncertainty.array[left_idx_beta:right_idx_beta+1]]
            
            H_beta_line_mean = np.round(np.mean(H_beta_line_region[1]), 4)
            
            H_beta_line_sum_err = [i**2 for i in H_beta_line_region[2]]
            H_beta_line_mean_err = np.round((np.sqrt(np.sum(H_beta_line_sum_err))/len(H_beta_line_sum_err)), 4)
            
            ## Balmer Decrement
            
            balmer_dec = np.round((H_alpha_line_mean/H_beta_line_mean), 4)
            
            balmer_dec_err = np.round((balmer_dec*np.sqrt((H_alpha_line_mean_err/H_alpha_line_mean)**2 + (H_beta_line_mean_err/H_beta_line_mean)**2)), 4)
            
            if print_stat:
                print('Mean of {} flux points in Hα: {}±{}'.format(len(H_alpha_line_region[1]), H_alpha_line_mean, H_alpha_line_mean_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('Mean of {} flux points in Hβ: {}±{}'.format(len(H_beta_line_region[1]), H_beta_line_mean, H_beta_line_mean_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('Calculated Balmer Decrement is: {}±{}'.format(balmer_dec, balmer_dec_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
                
            # Plots spectrum
            
            if plot_spec:
                f, (ax1, ax2)  = plt.subplots(2, 1, figsize=(10,8))
                
                ax1.plot(spec1d_alpha.spectral_axis, spec1d_alpha.flux, '-k')  
                ax1.vlines(H_alpha_line, ymin=0, ymax=max(H_alpha_line_region[1]), linestyles='dotted', colors='red')
                ax1.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(H_alpha_line_region[1]), linestyles='--', colors='black', label='Hα band width = {}nm'.format(H_alpha_band))
                ax1.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(H_alpha_line_region[1]), linestyles='--', colors='black')
                ax1.set_xlabel('$\lambda (nm)$')
                ax1.set_ylabel("Normalized Flux")
                ax1.set_xlim((H_alpha_line-(H_alpha_band)/2)-0.1, (H_alpha_line+(H_alpha_band)/2)+0.1)
                ax1.yaxis.set_ticks_position('both')
                ax1.xaxis.set_ticks_position('both')
                ax1.tick_params(direction='in', which='both')
                ax1.legend()
               
                ax2.plot(spec1d_beta.spectral_axis, spec1d_beta.flux, '-k')  
                ax2.vlines(H_beta_line, ymin=0, ymax=max(H_beta_line_region[1]), linestyles='dotted', colors='blue')
                ax2.vlines(H_beta_line-(H_beta_band/2), ymin=0, ymax=max(H_beta_line_region[1]), linestyles='--', colors='black', label='Hβ band width = {}nm'.format(H_beta_band))
                ax2.vlines(H_beta_line+(H_beta_band/2), ymin=0, ymax=max(H_beta_line_region[1]), linestyles='--', colors='black')
                ax2.set_xlabel('$\lambda (nm)$')
                ax2.set_ylabel("Normalized Flux")
                ax2.set_xlim((H_beta_line-(H_beta_band)/2)-0.1, (H_beta_line+(H_beta_band)/2)+0.1)
                ax2.yaxis.set_ticks_position('both')
                ax2.xaxis.set_ticks_position('both')
                ax2.tick_params(direction='in', which='both')
                ax2.legend()
                
                f.tight_layout()
                
                
                if save_figs:
                    plt.savefig('{}.png'.format(save_figs_name), format='png', dpi=300)
                
            if out_file_path != None:
                header = ['HJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'NUM_EXP', 'GAIN', 'RON', 'V_mag', 'T_eff', 'RV', 'F_H_alpha', 'F_H_alpha_err', 'F_H_beta', 'F_H_beta_err', 'BD', 'BD_err']
                res = list(obj_params.values()) + [H_alpha_line_mean, H_alpha_line_mean_err, H_beta_line_mean, H_beta_line_mean_err, balmer_dec, balmer_dec_err] 
                results.append(res)
            else:
                header = ['BD', 'BD_err']
                res = [balmer_dec, balmer_dec_err]
                results.append(res)
                
        # HARPS
        
        elif Instrument == 'HARPS':
            
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
            
            # Calculating doppler shift size using delta_lambda/lambda = v/c and the RV from either the CCF FITS file or the user given value.
           
            shift = ((obj_params['RV']/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 3)) # Using only 3 decimal places for the shift value since that's the precision of the wavelength in the .fits files!
            
            # Since the HARPS spectra have their individual spectral orders stitched together, we do not have to extract them separately as done for NARVAL. Thus for HARPS, the required region is extracted by slicing the spectrum with the index corresponding to the left and right continuum obtained using the 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, H_beta_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, H_alpha_line+2)
            
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
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx+1]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err_ron[left_idx:right_idx+1], unit=u.Jy))
                
            else:
                
                spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                                    flux=flx[left_idx:right_idx+1]*u.Jy,
                                    uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
            
            if print_stat:
                print('The doppler shift size using RV {} m/s and the Hα line of 656.2808nm is: {}nm'.format(obj_params['RV'], shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            # Calculating the fluxes
            
            ## H_alpha
            
            left_idx_alpha = find_nearest(spec1d.spectral_axis.value, H_alpha_line-(H_alpha_band/2)) 
            right_idx_alpha = find_nearest(spec1d.spectral_axis.value, H_alpha_line+(H_alpha_band/2))
                                                   
            H_alpha_line_region = [spec1d.spectral_axis.value[left_idx_alpha:right_idx_alpha+1], 
                                   spec1d.flux.value[left_idx_alpha:right_idx_alpha+1], 
                                   spec1d.uncertainty.array[left_idx_alpha:right_idx_alpha+1]]
            
            H_alpha_line_mean = np.round(np.mean(H_alpha_line_region[1]), 4) # Calculating mean of the flux within this bandwidth
            
            # Calculating the standard error on the mean flux calculated above.
            H_alpha_line_sum_err = [i**2 for i in H_alpha_line_region[2]]
            H_alpha_line_mean_err = np.round((np.sqrt(np.sum(H_alpha_line_sum_err))/len(H_alpha_line_sum_err)), 4)
            
            ## H_beta
            
            left_idx_beta = find_nearest(spec1d.spectral_axis.value, H_beta_line-(H_beta_band/2)) 
            right_idx_beta = find_nearest(spec1d.spectral_axis.value, H_beta_line+(H_beta_band/2))
                                                   
            H_beta_line_region = [spec1d.spectral_axis.value[left_idx_beta:right_idx_beta+1], 
                                  spec1d.flux.value[left_idx_beta:right_idx_beta+1], 
                                  spec1d.uncertainty.array[left_idx_beta:right_idx_beta+1]]
            
            H_beta_line_mean = np.round(np.mean(H_beta_line_region[1]), 4)
            
            H_beta_line_sum_err = [i**2 for i in H_beta_line_region[2]]
            H_beta_line_mean_err = np.round((np.sqrt(np.sum(H_beta_line_sum_err))/len(H_beta_line_sum_err)), 4)
            
            ## Balmer Decrement
            
            balmer_dec = np.round((H_alpha_line_mean/H_beta_line_mean), 4)
            
            balmer_dec_err = np.round((balmer_dec*np.sqrt((H_alpha_line_mean_err/H_alpha_line_mean)**2 + (H_beta_line_mean_err/H_beta_line_mean)**2)), 4)
            
            if print_stat:
                print('Mean of {} flux points in Hα: {}±{}'.format(len(H_alpha_line_region[1]), H_alpha_line_mean, H_alpha_line_mean_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('Mean of {} flux points in Hβ: {}±{}'.format(len(H_beta_line_region[1]), H_beta_line_mean, H_beta_line_mean_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('Calculated Balmer Decrement is: {}±{}'.format(balmer_dec, balmer_dec_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            # Plots spectrum
            
            if plot_spec:
                f, (ax1, ax2)  = plt.subplots(2, 1, figsize=(10,8))
                
                ax1.plot(spec1d.spectral_axis, spec1d.flux, '-k')  
                ax1.vlines(H_alpha_line, ymin=0, ymax=max(H_alpha_line_region[1]), linestyles='dotted', colors='red')
                ax1.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(H_alpha_line_region[1]), linestyles='--', colors='black', label='Hα band width = {}nm'.format(H_alpha_band))
                ax1.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(H_alpha_line_region[1]), linestyles='--', colors='black')
                ax1.set_xlabel('$\lambda (nm)$')
                ax1.set_ylabel("Flux (adu)")
                ax1.set_xlim((H_alpha_line-(H_alpha_band)/2)-0.1, (H_alpha_line+(H_alpha_band)/2)+0.1)
                ax1.yaxis.set_ticks_position('both')
                ax1.xaxis.set_ticks_position('both')
                ax1.tick_params(direction='in', which='both')
                ax1.legend()
               
                ax2.plot(spec1d.spectral_axis, spec1d.flux, '-k')  
                ax2.vlines(H_beta_line, ymin=0, ymax=max(H_beta_line_region[1]), linestyles='dotted', colors='blue')
                ax2.vlines(H_beta_line-(H_beta_band/2), ymin=0, ymax=max(H_beta_line_region[1]), linestyles='--', colors='black', label='Hβ band width = {}nm'.format(H_beta_band))
                ax2.vlines(H_beta_line+(H_beta_band/2), ymin=0, ymax=max(H_beta_line_region[1]), linestyles='--', colors='black')
                ax2.set_xlabel('$\lambda (nm)$')
                ax2.set_ylabel("Flux (adu)")
                ax2.set_xlim((H_beta_line-(H_beta_band)/2)-0.1, (H_beta_line+(H_beta_band)/2)+0.1)
                ax2.yaxis.set_ticks_position('both')
                ax2.xaxis.set_ticks_position('both')
                ax2.tick_params(direction='in', which='both')
                ax2.legend()
                
                f.tight_layout()
                
                if save_figs:
                    plt.savefig('{}.png'.format(save_figs_name), format='png', dpi=300)
                
            header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'BERV', 'OBS_DATE', 'PROG_ID', 'SNR', 'SIGDET', 'CONAD', 'RON', 'RV', 'F_H_alpha', 'F_H_alpha_err', 'F_H_beta', 'F_H_beta_err', 'BD', 'BD_err']
            res = list(obj_params.values()) + [H_alpha_line_mean, H_alpha_line_mean_err, H_beta_line_mean, H_beta_line_mean_err, balmer_dec, balmer_dec_err]
            results.append(res)
            
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
            
            shift = ((obj_params['RV']/ap.constants.c.value)*H_alpha_line)  
            shift = (round(shift, 3)) 
            
            # Same as the HARPS spectra, the HARPS-N spectra have their individual spectral orders stitched together and 
            # we do not have to extract them separately as done for NARVAL. Thus, the required region is extracted by slicing
            # the spectrum with the index corresponding to the left and right continuum obtained using the 
            # 'find_nearest' function. 
            
            left_idx = find_nearest(wvl, H_beta_line-2) # ± 2nm extra included for both!
            right_idx = find_nearest(wvl, H_alpha_line+2)
            
            with warnings.catch_warnings(): # Ignore warnings
                warnings.simplefilter('ignore')
                flx_err = [np.sqrt(flux) for flux in flx] # Using only photon noise as flx_err approx since no RON info available!
            
            # Slicing the data to contain only the region required for the index calculation as explained above and creating 
            # a spectrum class for it
            
            spec1d = Spectrum1D(spectral_axis=(wvl[left_idx:right_idx+1] - shift)*u.nm, 
                              flux=flx[left_idx:right_idx+1]*u.Jy,
                              uncertainty=StdDevUncertainty(flx_err[left_idx:right_idx+1], unit=u.Jy))
            
            if print_stat:
                print('The doppler shift size using RV {} m/s and the Hα line of 656.2808nm is: {}nm'.format(obj_params['RV'], shift))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('The spectral region used ranges from {}nm to {}nm. These values are doppler shift corrected and rounded off to 3 decimal places'.format(spec1d.spectral_axis[0].value, 
                                                                                                                                                              spec1d.spectral_axis[-1].value))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            # Calculating the fluxes
            
            ## H_alpha
            
            left_idx_alpha = find_nearest(spec1d.spectral_axis.value, H_alpha_line-(H_alpha_band/2)) 
            right_idx_alpha = find_nearest(spec1d.spectral_axis.value, H_alpha_line+(H_alpha_band/2))
                                                   
            H_alpha_line_region = [spec1d.spectral_axis.value[left_idx_alpha:right_idx_alpha+1], 
                                   spec1d.flux.value[left_idx_alpha:right_idx_alpha+1], 
                                   spec1d.uncertainty.array[left_idx_alpha:right_idx_alpha+1]]
            
            H_alpha_line_mean = np.round(np.mean(H_alpha_line_region[1]), 4) # Calculating mean of the flux within this bandwidth
            
            # Calculating the standard error on the mean flux calculated above.
            H_alpha_line_sum_err = [i**2 for i in H_alpha_line_region[2]]
            H_alpha_line_mean_err = np.round((np.sqrt(np.sum(H_alpha_line_sum_err))/len(H_alpha_line_sum_err)), 4)
            
            ## H_beta
            
            left_idx_beta = find_nearest(spec1d.spectral_axis.value, H_beta_line-(H_beta_band/2)) 
            right_idx_beta = find_nearest(spec1d.spectral_axis.value, H_beta_line+(H_beta_band/2))
                                                   
            H_beta_line_region = [spec1d.spectral_axis.value[left_idx_beta:right_idx_beta+1], 
                                  spec1d.flux.value[left_idx_beta:right_idx_beta+1], 
                                  spec1d.uncertainty.array[left_idx_beta:right_idx_beta+1]]
            
            H_beta_line_mean = np.round(np.mean(H_beta_line_region[1]), 4)
            
            H_beta_line_sum_err = [i**2 for i in H_beta_line_region[2]]
            H_beta_line_mean_err = np.round((np.sqrt(np.sum(H_beta_line_sum_err))/len(H_beta_line_sum_err)), 4)
            
            ## Balmer Decrement
            
            balmer_dec = np.round((H_alpha_line_mean/H_beta_line_mean), 4)
            
            balmer_dec_err = np.round((balmer_dec*np.sqrt((H_alpha_line_mean_err/H_alpha_line_mean)**2 + (H_beta_line_mean_err/H_beta_line_mean)**2)), 4)
            
            if print_stat:
                print('Mean of {} flux points in Hα: {}±{}'.format(len(H_alpha_line_region[1]), H_alpha_line_mean, H_alpha_line_mean_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('Mean of {} flux points in Hβ: {}±{}'.format(len(H_beta_line_region[1]), H_beta_line_mean, H_beta_line_mean_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('Calculated Balmer Decrement is: {}±{}'.format(balmer_dec, balmer_dec_err))
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
                
            # Plots spectrum
            
            if plot_spec:
                f, (ax1, ax2)  = plt.subplots(2, 1, figsize=(10,8))
                
                ax1.plot(spec1d.spectral_axis, spec1d.flux, '-k')  
                ax1.vlines(H_alpha_line, ymin=0, ymax=max(H_alpha_line_region[1]), linestyles='dotted', colors='red')
                ax1.vlines(H_alpha_line-(H_alpha_band/2), ymin=0, ymax=max(H_alpha_line_region[1]), linestyles='--', colors='black', label='Hα band width = {}nm'.format(H_alpha_band))
                ax1.vlines(H_alpha_line+(H_alpha_band/2), ymin=0, ymax=max(H_alpha_line_region[1]), linestyles='--', colors='black')
                ax1.set_xlabel('$\lambda (nm)$')
                ax1.set_ylabel("Flux (adu)")
                ax1.set_xlim((H_alpha_line-(H_alpha_band)/2)-0.1, (H_alpha_line+(H_alpha_band)/2)+0.1)
                ax1.yaxis.set_ticks_position('both')
                ax1.xaxis.set_ticks_position('both')
                ax1.tick_params(direction='in', which='both')
                ax1.legend()
               
                ax2.plot(spec1d.spectral_axis, spec1d.flux, '-k')  
                ax2.vlines(H_beta_line, ymin=0, ymax=max(H_beta_line_region[1]), linestyles='dotted', colors='blue')
                ax2.vlines(H_beta_line-(H_beta_band/2), ymin=0, ymax=max(H_beta_line_region[1]), linestyles='--', colors='black', label='Hβ band width = {}nm'.format(H_beta_band))
                ax2.vlines(H_beta_line+(H_beta_band/2), ymin=0, ymax=max(H_beta_line_region[1]), linestyles='--', colors='black')
                ax2.set_xlabel('$\lambda (nm)$')
                ax2.set_ylabel("Flux (adu)")
                ax2.set_xlim((H_beta_line-(H_beta_band)/2)-0.1, (H_beta_line+(H_beta_band)/2)+0.1)
                ax2.yaxis.set_ticks_position('both')
                ax2.xaxis.set_ticks_position('both')
                ax2.tick_params(direction='in', which='both')
                ax2.legend()
                
                f.tight_layout()
                
                if save_figs:
                    plt.savefig('{}.png'.format(save_figs_name), format='png', dpi=300)
                
            header = ['BJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'OBS_DATE', 'PROG_ID', 'RV', 'F_H_alpha', 'F_H_alpha_err', 'F_H_beta', 'F_H_beta_err', 'BD', 'BD_err']
            res = list(obj_params.values()) + [H_alpha_line_mean, H_alpha_line_mean_err, H_beta_line_mean, H_beta_line_mean_err, balmer_dec, balmer_dec_err]
            results.append(res)
                
        else:
            raise ValueError('Instrument type not recognised. Available options are "NARVAL", "HARPS" and "HARPS-N"')
    
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

## Defining a function to calculate the CaII IRT indices following M. MIttag et al. 2017 (2017, A&A, 607, A87)

def CaII_IRT(file_path,
             radial_velocity,
             IRT_1_line=849.8,
             IRT_1_band=0.1,
             IRT_1_F1_line=849.0,
             IRT_1_F1_band=0.2,
             IRT_1_F2_line=850.9,
             IRT_1_F2_band=0.2,
             IRT_2_line=854.2,
             IRT_2_band=0.1,
             IRT_2_F1_line=853.0,
             IRT_2_F1_band=0.2,
             IRT_2_F2_line=856.6,
             IRT_2_F2_band=0.2,
             IRT_3_line=866.2,
             IRT_3_band=0.1,
             IRT_3_F1_line=864.1,
             IRT_3_F1_band=0.2,
             IRT_3_F2_line=867.8,
             IRT_3_F2_band=0.2,
             plot_spec=True,
             print_stat=True,
             save_results=False, 
             results_file_name=None,
             save_figs=False,
             save_figs_name=None,
             out_file_path=None):
    
    """
    
    Calculates the CaII infrared triplet (IRT) indices following M. MIttag et al. 2017 (2017, A&A, 607, A87). This method is analogous to the
    H_alpha index calculation method by Boisse et al. 2009 in which the mean flux in the IRT line is divided by the mean flux in certain
    reference continuum bands. For the subtraction method, see Martínez-Arnáiz, R., López-Santiago, J., Crespo-Chacón, I., & Montes, D. 2011, MNRAS, 414, 2629.
    
    NOTE: This index is calculated ONLY for the NARVAL instrument due to its wider wavelength coverage containing these IRT lines. TheHARPS instrument does not cover 
    these lines.
    
    
    Parameters:
    -----------
    
    file_path: list, .s format (NARVAL)
    List containng the paths of the spectrum files 
    
    radial_velocity: int
    Stellar radial velocity along the line-of-sight. This value is used for doppler shifting the spectra to its rest frame.
    
    IRT_1_line: int, default: 849.8 nm
    IRT line 1 centre in nm.
    
    IRT_1_band: int, default: 0.1 nm
    Band width (nm) in which to calculate the mean flux.
    
    IRT_2_line: int, default: 854.2 nm
    IRT line 1 centre in nm.
    
    IRT_2_band: int, default: 0.1 nm
    Band width (nm) in which to calculate the mean flux.
    
    IRT_3_line: int, default: 866.2 nm
    IRT line 1 centre in nm.
    
    IRT_3_band: int, default: 0.1 nm
    Band width (nm) in which to calculate the mean flux.
    
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
    
    Returns:
    -----------
    HJD, RA, DEC, AIRMASS, Exposure time[s], No. of exposures, GAIN [e-/ADU], ReadOut Noise [e-], V_mag, T_eff[K], RV[m/s], IRT 1 index, IRT 1 index error, IRT 2 index, IRT 2 index error, IRT 3 index, IRT 3 index error.
    
    All values are type float() given inside a list.
    
    """
    
    results = [] # Empty list to which the run results will be appended
    
    # Creating a loop to go through each given file_path in the list of file paths
    
    # Using the tqdm function 'log_progress' to provide a neat progress bar in Jupyter Notebook which shows the total number of
    # runs, the run time per iteration and the total run time for all files!
    
    for i in log_progress(range(len(file_path)), desc='Calculating CaII IRT indices'):
            
        if out_file_path != None:
            
            # Using read_data from krome.spec_analysis to extract useful object parameters and all individual spectral orders
            
            obj_params, orders = read_data(file_path=file_path[i],
                                           out_file_path=out_file_path[i],
                                           Instrument='NARVAL',
                                           print_stat=print_stat,
                                           show_plots=False)
            
            obj_params['RV'] = radial_velocity # setting radial_velocity as part of the obj_params dictionary for continuity 
            
        else:
            
            orders = read_data(file_path=file_path[i],
                               Instrument='NARVAL',
                               print_stat=print_stat,
                               out_file_path=None,
                               show_plots=False)
            
            if print_stat:
                print('"out_file_path" not given as an argument. Run will only return the indices and their errros instead.')
                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            

        if print_stat:
            print('Total {} spectral orders extracted'.format(len(orders)))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        order_27 = orders[61-27] # This order contains the 849.8 IRT line
        order_26 = orders[61-26] # This order contains the other two IRT lines
        
        if print_stat:
            print('The #27 IRT_1 order wavelength read from .s file using pandas is: {}'.format(order_27[0].values))
            print('The #27 IRT_1 order intensity read from .s file using pandas is: {}'.format(order_27[1].values))
            print('The #27 IRT_1 order intensity error read from .s file using pandas is: {}'.format(order_27[2].values))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('The #26 IRT_23 order wavelength read from .s file using pandas is: {}'.format(order_26[0].values))
            print('The #26 IRT_23 order intensity read from .s file using pandas is: {}'.format(order_26[1].values))
            print('The #26 IRT_23 order intensity error read from .s file using pandas is: {}'.format(order_26[2].values))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        # The spectra is now doppler shift corrected in the wavelength axis using the stellar radial velocity and the rest wavelength of IRT 3 line; delta_lambda = (v/c)*lambda. Any of the IRT lines can be used for doppler shifting the spectrum since they do not produce a significant difference in the final index.
        
        shift = ((radial_velocity/ap.constants.c.value)*IRT_3_line)  
        shift = (round(shift, 4)) # Using only 4 decimal places for the shift value since that's the precision of the wavelength in the .s files!
        
        wvl_IRT_1 = np.round((order_27[0].values - shift), 4) 
        flx_IRT_1 = order_27[1].values 
        flx_err_IRT_1 = order_27[2].values
        
        wvl_IRT_23 = np.round((order_26[0].values - shift), 4) 
        flx_IRT_23 = order_26[1].values 
        flx_err_IRT_23 = order_26[2].values
        
        # Creating a spectrum object called 'spec1d' using 'Spectrum1D' from 'specutils'
        # Docs for 'specutils' are here; https://specutils.readthedocs.io/en/stable/ 
        
        # The spectral and flux axes are given units nm and Jy respectively using 'astropy.units'. 
        # The uncertainty has units Jy as well!
    
        spec1d_IRT_1 = Spectrum1D(spectral_axis=wvl_IRT_1*u.nm, 
                                  flux=flx_IRT_1*u.Jy, 
                                  uncertainty=StdDevUncertainty(flx_err_IRT_1, unit=u.Jy)) 
        
        spec1d_IRT_23 = Spectrum1D(spectral_axis=wvl_IRT_23*u.nm, 
                                  flux=flx_IRT_23*u.Jy, 
                                  uncertainty=StdDevUncertainty(flx_err_IRT_23, unit=u.Jy))
        
        # Printing info
        
        if print_stat:
            print('The doppler shift size using RV {} m/s and the CaII IRT line of 866.2nm is: {}nm'.format(radial_velocity, shift))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            
        # Plots spectrum
        
        if plot_spec:
            f, (ax1, ax2, ax3)  = plt.subplots(3, 1, figsize=(6, 12))
            
            ax1.plot(spec1d_IRT_1.spectral_axis, spec1d_IRT_1.flux, '-k', label='Order #27')  
            ax1.vlines(IRT_1_line, ymin=0, ymax=max(spec1d_IRT_1.flux.value), linestyles='dotted', colors='red')
            ax1.vlines(IRT_1_line-(IRT_1_band/2), ymin=0, ymax=max(spec1d_IRT_1.flux.value), linestyles='--', colors='black', label='CaII IRT_1 bandwidth = {}nm'.format(IRT_1_band))
            ax1.vlines(IRT_1_line+(IRT_1_band/2), ymin=0, ymax=max(spec1d_IRT_1.flux.value), linestyles='--', colors='black')
            ax1.set_xlabel('$\lambda (nm)$')
            ax1.set_ylabel("Normalized Flux")
            ax1.set_xlim((IRT_1_line-(IRT_1_band/2))-0.1, (IRT_1_line+(IRT_1_band/2))+0.1)
            ax1.yaxis.set_ticks_position('both')
            ax1.xaxis.set_ticks_position('both')
            ax1.tick_params(direction='in', which='both')
            ax1.legend()
           
            ax2.plot(spec1d_IRT_23.spectral_axis, spec1d_IRT_23.flux, '-k', label='Order # 26')  
            ax2.vlines(IRT_2_line, ymin=0, ymax=max(spec1d_IRT_23.flux.value), linestyles='dotted', colors='green')
            ax2.vlines(IRT_2_line-(IRT_2_band/2), ymin=0, ymax=max(spec1d_IRT_23.flux.value), linestyles='--', colors='black', label='CaII IRT_2 bandwidth = {}nm'.format(IRT_2_band))
            ax2.vlines(IRT_2_line+(IRT_2_band/2), ymin=0, ymax=max(spec1d_IRT_23.flux.value), linestyles='--', colors='black')
            ax2.set_xlabel('$\lambda (nm)$')
            ax2.set_ylabel("Normalized Flux")
            ax2.set_xlim((IRT_2_line-(IRT_2_band/2))-0.1, (IRT_2_line+(IRT_2_band/2))+0.1)
            ax2.yaxis.set_ticks_position('both')
            ax2.xaxis.set_ticks_position('both')
            ax2.tick_params(direction='in', which='both')
            ax2.legend()
            
            ax3.plot(spec1d_IRT_23.spectral_axis, spec1d_IRT_23.flux, '-k', label='Order # 26')  
            ax3.vlines(IRT_3_line, ymin=0, ymax=max(spec1d_IRT_23.flux.value), linestyles='dotted', colors='blue')
            ax3.vlines(IRT_3_line-(IRT_3_band/2), ymin=0, ymax=max(spec1d_IRT_23.flux.value), linestyles='--', colors='black', label='CaII IRT_3 bandwidth = {}nm'.format(IRT_3_band))
            ax3.vlines(IRT_3_line+(IRT_3_band/2), ymin=0, ymax=max(spec1d_IRT_23.flux.value), linestyles='--', colors='black')
            ax3.set_xlabel('$\lambda (nm)$')
            ax3.set_ylabel("Normalized Flux")
            ax3.set_xlim((IRT_3_line-(IRT_3_band/2))-0.1, (IRT_3_line+(IRT_3_band/2))+0.1)
            ax3.yaxis.set_ticks_position('both')
            ax3.xaxis.set_ticks_position('both')
            ax3.tick_params(direction='in', which='both')
            ax3.legend()
            
            f.tight_layout()
            
            if save_figs:
                plt.savefig('{}_CaII_IRT_lines_plot.pdf'.format(save_figs_name), format='pdf')
                
        # Extracting each of the IRT line region using 'extract_region' from 'specutils'
        
        ## IRT 1
        
        F_IRT_1_region = extract_region(spec1d_IRT_1, region=SpectralRegion((IRT_1_line-(IRT_1_band/2))*u.nm, (IRT_1_line+(IRT_1_band/2))*u.nm))
        F1_IRT_1_region = extract_region(spec1d_IRT_1, region=SpectralRegion((IRT_1_F1_line-(IRT_1_F1_band/2))*u.nm, (IRT_1_F1_line+(IRT_1_F1_band/2))*u.nm))
        F2_IRT_1_region = extract_region(spec1d_IRT_1, region=SpectralRegion((IRT_1_F2_line-(IRT_1_F2_band/2))*u.nm, (IRT_1_F2_line+(IRT_1_F2_band/2))*u.nm))
        
        ## IRT 2
        
        F_IRT_2_region = extract_region(spec1d_IRT_23, region=SpectralRegion((IRT_2_line-(IRT_2_band/2))*u.nm, (IRT_2_line+(IRT_2_band/2))*u.nm))
        F1_IRT_2_region = extract_region(spec1d_IRT_23, region=SpectralRegion((IRT_2_F1_line-(IRT_2_F1_band/2))*u.nm, (IRT_2_F1_line+(IRT_2_F1_band/2))*u.nm))
        F2_IRT_2_region = extract_region(spec1d_IRT_23, region=SpectralRegion((IRT_2_F2_line-(IRT_2_F2_band/2))*u.nm, (IRT_2_F2_line+(IRT_2_F2_band/2))*u.nm))
        
        ## IRT 3
        
        F_IRT_3_region = extract_region(spec1d_IRT_23, region=SpectralRegion((IRT_3_line-(IRT_3_band/2))*u.nm, (IRT_3_line+(IRT_3_band/2))*u.nm))
        F1_IRT_3_region = extract_region(spec1d_IRT_23, region=SpectralRegion((IRT_3_F1_line-(IRT_3_F1_band/2))*u.nm, (IRT_3_F1_line+(IRT_3_F1_band/2))*u.nm))
        F2_IRT_3_region = extract_region(spec1d_IRT_23, region=SpectralRegion((IRT_3_F2_line-(IRT_3_F2_band/2))*u.nm, (IRT_3_F2_line+(IRT_3_F2_band/2))*u.nm))
                                        
        
        regions = [F_IRT_1_region, F1_IRT_1_region, F2_IRT_1_region, 
                   F_IRT_2_region, F1_IRT_2_region, F2_IRT_3_region, 
                   F_IRT_3_region, F1_IRT_3_region, F2_IRT_3_region]
            
        # The indices are calculated using the 'calc_ind' function from krome.spec_analysis by inputting the extracted regions as shown below;
        
        I_IRT_1, I_IRT_1_err, I_IRT_2, I_IRT_2_err, I_IRT_3, I_IRT_3_err = calc_ind(regions=regions,
                                                                                    index_name='CaII_IRT',
                                                                                    print_stat=print_stat)
        
        if out_file_path != None:
            header = ['HJD', 'RA', 'DEC', 'AIRMASS', 'T_EXP', 'NUM_EXP', 'GAIN', 'RON', 'V_mag', 'T_eff', 'RV', 'I_IRT1', 'I_IRT1_err', 'I_IRT2', 'I_IRT2_err', 'I_IRT3', 'I_IRT3_err']
            res = list(obj_params.values()) + [I_IRT_1, I_IRT_1_err, I_IRT_2, I_IRT_2_err, I_IRT_3, I_IRT_3_err] 
            results.append(res)
        else:
            header = ['I_IRT1', 'I_IRT1_err', 'I_IRT2', 'I_IRT2_err', 'I_IRT3', 'I_IRT3_err']
            res = [I_IRT_1, I_IRT_1_err, I_IRT_2, I_IRT_2_err, I_IRT_3, I_IRT_3_err]
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