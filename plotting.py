#!/usr/bin/env python
# coding: utf-8

"""
plotting.py: This python module contains functions to plot results obtained from the index_calc.py module.

"""

__author__ = "Mukul Kumar"
__email__ = "Mukul.k@uaeu.ac.ae, MXK606@alumni.bham.ac.uk"
__date__ = "24-02-2022"
__version__ = "1.1"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

## Defining a function that calculates and plots the Pearson R correlation between two datasets  

def plot_corr(x,
              xerr,
              y,
              yerr,
              xlabel,
              ylabel, 
              fmt='ok',
              ecolor='red',
              capsize=3, 
              alpha=1.0,
              title=None, 
              save_fig=False,
              save_plot_name=None):
    
    """
    Calculates the Pearson R correlation coefficient between two datasets using the scipy.stats.pearsonr function and 
    plots a best fit line to the two datasets scatter plot.
    
    Parameters:
    -----------
    x: arr
    Array containing the first dataset
    
    xerr: arr
    Array containing the error on the first dataset. 
    NOTE: The errors are used ONLY for plotting and are not used when calculating the correlation coefficient.
    
    y: arr
    Array containing the second dataset
    
    yerr: arr
    Array containing the error on the second dataset
    
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
    
    alpha: int, default=1.0
    Plot transparency
    
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
    
    f, ax = plt.subplots()
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, 
                fmt=fmt, ecolor=ecolor, capsize=capsize,
                alpha=alpha)
    
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), '-.k') # fitting a best fit line to the scatter plot
    plt.annotate(r'$\rho$ = {}'.format(np.round(p, 2)), xy=(0.05, 0.92), xycoords='axes fraction', size='large')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.minorticks_on()
    ax.tick_params(direction='in', which='both')
    plt.tight_layout()
    slope, intercept = np.polyfit(x,y,1)
    if save_fig:
        plt.savefig('{}.pdf'.format(save_plot_name), format='pdf')
    print('R: {}'.format(np.round(p, 4)))
    print('p-value: {:.4e}'.format(p_val))
    print('Slope: {} '.format(np.round(slope, 4)))
    print('Intercept: {} '.format(np.round(intercept, 4)))
    
    return p, p_val, slope, intercept
