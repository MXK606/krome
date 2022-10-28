"""
This is a Python package containing the modules; 'index_calc', 'spec_analysis' and 'plotting'.
"""

__author__ = "Mukul Kumar"
__email__ = "mukulkumar531@gmail.com, MXK606@alumni.bham.ac.uk"
__date__ = "11-10-2022"
__version__ = "1.3.1"
__all__ = ["index_calc", "spec_analysis", "plotting"] # Using this DUNDER function so that running 'from test_package import *' imports only these modules

## Prints author information when imported

print('KROME ' + __version__)
print('Author: ' + __author__)
print('Email: ' + __email__)