#!/usr/bin/env python
# coding: utf-8

"""
plotting.py: This python module contains functions to plot results obtained from the index_calc.py module.

"""

__author__ = "Mukul Kumar"
__email__ = "Mukul.k@uaeu.ac.ae, MXK606@alumni.bham.ac.uk"
__date__ = "17-03-2022"
__version__ = "1.2"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from krome.spec_analysis import read_data

## Defining a function that calculates and plots the Pearson R correlation between two datasets  

def corr_plot(x,
              y,
              xlabel,
              ylabel, 
              fmt='ok',
              ecolor='red',
              capsize=3, 
              fontsize=18,
              alpha=1.0,
              xerr=None,
              yerr=None,
              title=None,
              print_stat=True,
              save_fig=False,
              save_plot_name=None):
    
    """
    Calculates the Pearson R correlation coefficient between two datasets using the scipy.stats.pearsonr function and 
    plots a best fit line to the two datasets scatter plot.
    
    Parameters:
    -----------
    x: arr
    Array containing the first dataset
    
    y: arr
    Array containing the second dataset
    
    xlabel: str
    Label for the x-axis
    
    ylabel: str
    Label for the y-axis
    
    fmt: str, default='ok'
    Format for plotting the data points. Default is black dots
    
    ecolor: str, default='red'
    Error bar color
    
    capsize: int, default=3
    Error bar capsize
    
    fontsize: int, default=18
    Font size of the axis labels
    
    alpha: int, default=1.0
    Plot transparency
    
    xerr: list, default=None
    Array containing the error on the first dataset. 
    NOTE: The errors are used ONLY for plotting and are not used when calculating the correlation coefficient.
    
    yerr: list, defaulr=None
    Array containing the error on the second dataset
    
    title: str, default=None
    Plot title
    
    save_fig: bool, default=False
    Saves the plot as a PDF in the working  directory
    
    save_plot_name: str, default=None
    Name with which to save the plot
    
    Returns:
    --------
    
    Pearson’s correlation coefficient, Two-tailed p-value, slope of the best fit line and its intercept.
    
    All values are type float()
    
    """
    
    p, p_val = stats.pearsonr(x,y)
    p = np.round(p, 4)
    
    f, ax = plt.subplots()
    
    if xerr != None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, 
                    fmt=fmt, ecolor=ecolor, capsize=capsize,
                    alpha=alpha)
    else:
        plt.plot(x, y, fmt, alpha=alpha)
    
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), '-.k') # fitting a best fit line to the scatter plot
    plt.annotate(r'$\rho$ = {}'.format(np.round(p, 2)), xy=(0.05, 0.92), xycoords='axes fraction', size='large')
    plt.annotate('p-value = {:.2e}'.format(p_val), xy=(0.05, 0.85), xycoords='axes fraction', size='large')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    # ax.set_ylim(min(y)-(min(y)*0.1), max(y)+(max(y)*0.12))
    plt.minorticks_on()
    ax.tick_params(direction='in', which='both')
    f.tight_layout()
    slope, intercept = np.round(np.polyfit(x,y,1), 4)
    if save_fig:
        plt.savefig('{}.pdf'.format(save_plot_name), format='pdf')
        
        # Also saves the correlation results in a {save_plot_name}.meta text file as well.
        
        with open('{}.meta'.format(save_plot_name), 'w') as f:
            f.write('R:' + '\t' + str(p) + '\n')
            f.write('p-value:' + '\t' + str(p_val) + '\n')
            f.write('Slope:' + '\t' + str(slope) + '\n')
            f.write('Intercept:' + '\t' + str(intercept) + '\n')
        
        
    if print_stat:
        print('R: {}'.format(p))
        print('p-value: {:.4e}'.format(p_val))
        print('Slope: {} '.format(slope))
        print('Intercept: {} '.format(intercept))
    
    return p, p_val, slope, intercept

## Defining a function to plot the given activity indices against the system ephemerides!

def ephem_plot(ephem_file,
               index_file,
               index_col_name,
               save_fig=False):
    
    """
    Plots activity indices against their ephemerides.
    
    Parameters:
    -----------
    
    ephem_file:
    csv file containing the system ephemerides
    
    index_file:
    csv file containing the activity indices 
    
    index_col_name:
    Column name of the index to plot the ephemerides for
    
    save_fig: bool, default=False
    Saves the figures as a pdf in the working directory
    
    Returns:
    --------
    None. This is a void function.
    
    """
    
    ## Reading data using pandas
    
    ephem_data = pd.read_csv(ephem_file)
    index_data = pd.read_csv(index_file)
    
    ## Sorting both dataframes (df) by their JDs before plotting
    
    ephem_data = ephem_data.sort_values(by='JD')
    index_data = index_data.sort_values(by=index_data.columns[0]) ## Using df.columns to get the JD column name, i.e. either HJD, MJD or BJD
    
    ## Creating figure with two subplots 
    
    fig = plt.figure(figsize=(10,8)) 
    gs = fig.add_gridspec(2, 1, hspace=0.3, wspace=0.0)
    ax1, ax2 = gs.subplots()
    
    ## Assigning plotting variables
    
    xdata = ephem_data['Orbital_Phase']
    xdata1 = ephem_data['Rotational_Phase']
    ydata = index_data[index_col_name]
    ydata_err = index_data[index_col_name+'_err']
    
    ## Orbital phase plot
    
    ax1.tick_params(direction='in', axis='both')
    ax1.errorbar(xdata, ydata, yerr=ydata_err, fmt='.k', ecolor='black', capsize=3)
    ax1.set_ylabel(index_col_name)
    ax1.set_xlabel('Orbital Phase')  
    ax1.set_xlim(-0.05,1.0)
    
    ## Rotational phase plot
    
    ax2.tick_params(direction='in', axis='both')
    ax2.errorbar(xdata1, ydata, yerr=ydata_err, fmt='.k', ecolor='black', capsize=3)
    ax2.set_ylabel(index_col_name)
    ax2.set_xlabel('Rotational Phase')  
    ax2.set_xlim(-0.05,1.0)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('{}_vs_ephemerides.pdf'.format(index_col_name), format='pdf')
        
## Defining a function to overplot multiple spectrums!

def overplot(file_path,
             Instrument,
             save_fig=False,
             save_name=None):
    
    """
    Overplots multiple spectrums for further analysis.
    
    Parameters:
    -----------
    
    file_path: str
    List containing file paths of the .s/.fits files
    
    Instrument: str
    Instrument type used. Available options: ['NARVAL', 'HARPS', 'HARPS-N']
    
    save_fig: bool, default=False
    Saves the plot as a PDF in the working directory
    
    save_name: str, default=None
    Name with which to save the plot.
    
    Returns:
    --------
    None. This is a void function.
    
    """
    
    spec_all = []
    
    for file in file_path:
        
        if Instrument=='NARVAL':
            
            # Skipping first 2 rows of .s file and setting header to None to call columns by their index.
            # Assigning the sep manually and setting skipinitialspace to True to fix the issue of multiple leading spaces; 2 spaces upto 1000nm then 1 space!
            
            df = pd.read_csv(file, header=None, skiprows=2, sep=' ', skipinitialspace=True) 
            spec = [df[0].values, df[1].values]
            spec_all.append(spec)
            
        else:
            op, spec = read_data(file, Instrument, print_stat=False, show_plots=False)
            spec_all.append(spec)
            
    plt.figure(figsize=(10,4))
    for spec in spec_all:
        plt.plot(spec[0], spec[1])
    
    plt.xlabel(r'$\lambda$(nm)')
    
    if Instrument=='NARVAL':
        plt.ylabel('Normalized Flux')
    else:
        plt.ylabel('Flux (adu)')
    
    plt.title('Overplot of {} Individual spectrums'.format(len(file_path)))
    plt.tight_layout()
        
    if save_fig:
        plt.savefig('{}.pdf'.format(save_name), format='pdf')
    

## Defining a function to plot the reduced spectrum for a given index function!

def plot_spectrum(spec, 
                  lines, 
                  Index, 
                  Instrument, 
                  norm_spec, 
                  save_figs, 
                  save_figs_name,
                  CaI_index=None):
    
    """
    
    Insert Doc Here!
    
    """
    
    if Index == 'HaI':

        f, (ax1, ax2, ax3)  = plt.subplots(3, 1, figsize=(10,12))
        ax1.plot(spec.spectral_axis, spec.flux, '-k')  
        ax1.set_xlabel('$\lambda (nm)$')
        ax2.set_xlabel('$\lambda (nm)$')
        ax3.set_xlabel('$\lambda (nm)$')
        
        if Instrument=='NARVAL' or Instrument=='ESPADONS':
            ax1.set_ylabel("Normalized Flux")
            ax2.set_ylabel("Normalized Flux")
            ax3.set_ylabel("Normalized Flux")
        else:
            if norm_spec:
                ax1.set_ylabel("Normalized Flux")
                ax2.set_ylabel("Normalized Flux")
                ax3.set_ylabel("Normalized Flux")
            else:
                ax1.set_ylabel("Flux (adu)")
                ax2.set_ylabel("Flux (adu)")
                ax3.set_ylabel("Flux (adu)")
        
        ax1.axvline(lines[0]-(lines[1]/2), linestyle='--', color='black', label='Hα {}±{}nm'.format(lines[0], lines[1]/2))
        ax1.axvline(lines[0]+(lines[1]/2), linestyle='--', color='black')
        ax1.axvline(lines[2]-(lines[3]/2), linestyle='dotted', color='blue', label='Blue cont. {}±{}nm'.format(lines[2], lines[3]/2))
        ax1.axvline(lines[2]+(lines[3]/2), linestyle='dotted', color='blue')
        ax1.axvline(lines[4]-(lines[5]/2), linestyle='dashdot', color='red', label='Red cont. {}±{}nm'.format(lines[4], lines[5]/2))
        ax1.axvline(lines[4]+(lines[5]/2), linestyle='dashdot', color='red')
        
        if CaI_index:
            ax1.axvline(lines[6]-(lines[7]/2), linestyle='dashdot', color='black', label='CaI {}±{}nm'.format(lines[6], lines[7]/2))
            ax1.axvline(lines[6]+(lines[7]/2), linestyle='dashdot', color='black')
        
        ax1.set_xlim(lines[2]-1.1, lines[4]+1.1)
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.tick_params(direction='in', which='both')
        ax1.legend()
        
        # Plots the zoomed in regions around the H alpha line.
        
        ax2.plot(spec.spectral_axis, spec.flux)
        ax2.axvline(lines[0], ymin=0, linestyle='dotted', color='green')
        ax2.axvline(lines[0]-(lines[1]/2), linestyle='--', color='black', label='Hα band width = {}nm'.format(lines[1]))
        ax2.axvline(lines[0]+(lines[1]/2), linestyle='--', color='black')
        ax2.set_xlim(lines[0]-(lines[1]/2)-0.1, lines[0]+(lines[1]/2)+0.1)
        ax2.yaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')
        ax2.tick_params(direction='in', which='both')
        ax2.legend()        
                
        if CaI_index:
            
            # Plots the zoomed in regions around the CaI line.
            
            ax3.plot(spec.spectral_axis, spec.flux)
            ax3.axvline(lines[6], linestyle='dotted', color='green')
            ax3.axvline(lines[6]-(lines[7]/2), linestyle='--', color='black', label='CaI band width = {}nm'.format(lines[7]))
            ax3.axvline(lines[6]+(lines[7]/2), linestyle='--', color='black')
            ax3.set_xlim(lines[6]-(lines[7]/2)-0.1, lines[6]+(lines[7]/2)+0.1)
            ax3.yaxis.set_ticks_position('both')
            ax3.xaxis.set_ticks_position('both')
            ax3.tick_params(direction='in', which='both')
            ax3.legend()
            
        f.tight_layout()
        plt.minorticks_on()
            
        if save_figs:
                plt.savefig('{}_Hα_line_plot.pdf'.format(save_figs_name), format='pdf', dpi=300)
                
    elif Index=='HeI':
        
        f, (ax1, ax2)  = plt.subplots(2, 1, figsize=(10,8))
        ax1.plot(spec.spectral_axis, spec.flux, '-k')  
        ax1.set_xlabel('$\lambda (nm)$')
        ax2.set_xlabel('$\lambda (nm)$')
        
        if Instrument=='NARVAL' or Instrument=='ESPADONS':
            ax1.set_ylabel("Normalized Flux")
            ax2.set_ylabel("Normalized Flux")
        else:
            if norm_spec:
                ax1.set_ylabel("Normalized Flux")
                ax2.set_ylabel("Normalized Flux")
            else:
                ax1.set_ylabel("Flux (adu)")
                ax2.set_ylabel("Flux (adu)")
        
        ax1.axvline(lines[0]-(lines[1]/2), linestyle='--', color='black', label='HeID3 {}±{}nm'.format(lines[0], lines[1]/2))
        ax1.axvline(lines[0]+(lines[1]/2), linestyle='--', color='black')
        ax1.axvline(lines[2]-(lines[3]/2), linestyle='dotted', color='blue', label='Blue cont. {}±{}nm'.format(lines[2], lines[3]/2))
        ax1.axvline(lines[2]+(lines[3]/2), linestyle='dotted', color='blue')
        ax1.axvline(lines[4]-(lines[5]/2), linestyle='dashdot', color='red', label='Red cont. {}±{}nm'.format(lines[4], lines[5]/2))
        ax1.axvline(lines[4]+(lines[5]/2), linestyle='dashdot', color='red')
        ax1.set_xlim(lines[2]-0.5, lines[4]+0.5)
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.tick_params(direction='in', which='both')
        ax1.legend()
        
        # Plots the zoomed in regions around the HeI line.
        
        ax2.plot(spec.spectral_axis, spec.flux)
        ax2.axvline(lines[0], ymin=0, linestyle='dotted', color='green')
        ax2.axvline(lines[0]-(lines[1]/2), linestyle='--', color='black', label='HeID3 band width = {}nm'.format(lines[1]))
        ax2.axvline(lines[0]+(lines[1]/2), linestyle='--', color='black')
        ax2.set_xlim(lines[0]-(lines[1]/2)-0.1, lines[0]+(lines[1]/2)+0.1)
        ax2.yaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')
        ax2.tick_params(direction='in', which='both')
        ax2.legend()
        
        f.tight_layout()
        plt.minorticks_on()
        
        if save_figs:
            plt.savefig('{}_HeID3_line_plot.pdf'.format(save_figs_name), format='pdf')
            
    elif Index=='CaIIHK':
        
        f, (ax1, ax2, ax3)  = plt.subplots(3, 1, figsize=(10,12))
        ax1.plot(spec.spectral_axis, spec.flux, '-k')  
        ax1.set_xlabel('$\lambda (nm)$')
        ax2.set_xlabel('$\lambda (nm)$')
        ax3.set_xlabel('$\lambda (nm)$')
        
        if Instrument=='NARVAL' or Instrument=='ESPADONS':
            ax1.set_ylabel("Normalized Flux")
            ax2.set_ylabel("Normalized Flux")
            ax3.set_ylabel("Normalized Flux")
        else:
            if norm_spec:
                ax1.set_ylabel("Normalized Flux")
                ax2.set_ylabel("Normalized Flux")
                ax3.set_ylabel("Normalized Flux")
            else:
                ax1.set_ylabel("Flux (adu)")
                ax2.set_ylabel("Flux (adu)")
                ax3.set_ylabel("Flux (adu)")
        
        ax1.axvline(lines[0]-(lines[1]/2), linestyle='--', color='black', label='CaII H {}±{}nm'.format(lines[0], lines[1]/2))
        ax1.axvline(lines[0]+(lines[1]/2), linestyle='--', color='black')
        ax1.axvline(lines[2]-(lines[3]/2), linestyle='-.', color='black', label='CaII K {}±{}nm'.format(lines[2], lines[3]/2))
        ax1.axvline(lines[2]+(lines[3]/2), linestyle='-.', color='black')
        ax1.axvline(lines[4]-(lines[5]/2), linestyle='dotted', color='blue', label='Blue cont. {}±{}nm'.format(lines[4], lines[5]/2))
        ax1.axvline(lines[4]+(lines[5]/2), linestyle='dotted', color='blue')
        ax1.axvline(lines[6]-(lines[7]/2), linestyle='dashdot', color='red', label='Red cont. {}±{}nm'.format(lines[6], lines[7]/2))
        ax1.axvline(lines[6]+(lines[7]/2), linestyle='dashdot', color='red')
        ax1.set_xlim(lines[4]-1.5, lines[6]+1.5)
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.tick_params(direction='in', which='both')
        ax1.legend()
        
        # Plots the zoomed in regions around the H&K lines.
        
        ax2.plot(spec.spectral_axis, spec.flux)
        ax2.axvline(lines[0], ymin=0, linestyle='dotted', color='green')
        ax2.axvline(lines[0]-(lines[1]/2), linestyle='--', color='black', label='CaII H band width = {}nm'.format(lines[1]))
        ax2.axvline(lines[0]+(lines[1]/2), linestyle='--', color='black')
        ax2.set_xlim(lines[0]-(lines[1]/2)-0.1, lines[0]+(lines[1]/2)+0.1)
        ax2.yaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')
        ax2.tick_params(direction='in', which='both')
        ax2.legend()
        
        ax3.plot(spec.spectral_axis, spec.flux)
        ax3.axvline(lines[2], ymin=0, linestyle='dotted', color='green')
        ax3.axvline(lines[2]-(lines[3]/2), linestyle='-.', color='black', label='CaII K band width = {}nm'.format(lines[2]))
        ax3.axvline(lines[2]+(lines[3]/2), linestyle='-.', color='black')
        ax3.set_xlim(lines[2]-(lines[3]/2)-0.1, lines[2]+(lines[3]/2)+0.1)
        ax3.yaxis.set_ticks_position('both')
        ax3.xaxis.set_ticks_position('both')
        ax3.tick_params(direction='in', which='both')
        ax3.legend()
        
        f.tight_layout()
        plt.minorticks_on()
        
        if save_figs:
            plt.savefig('{}_CaIIHK_lines_plot.pdf'.format(save_figs_name), format='pdf')
        
    return


