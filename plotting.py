#!/usr/bin/env python
# coding: utf-8

"""
plotting.py: This python module contains functions to plot results obtained from the spec_analysis.py module.

"""

__author__ = "Mukul Kumar"
__email__ = "Mukul.k@uaeu.ac.ae, MXK606@alumni.bham.ac.uk"
__date__ = "07-02-2021"
__version__ = "1.0"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

## Defining a corr. plotting function

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
              save_plot_name=None, 
              save_fig=False):
    
    """
    Type Function Docstring Here!
    
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

## Defining a function for plotting the activity index against the JD



