# -*- coding: utf-8 -*-
"""
Predicting fule poverty example

@author: ucbva19
"""

import os
import datetime as dt

import pandas as pd
import matplotlib.pyplot as plt

# from ochre import Dwelling, Analysis
# from ochre.utils import default_input_path
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Plotting figures default
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
# plt.rcParams['figure.figsize'] = (4,4) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

#%%

data_path = 'C:\\Users\\ucbva19\\OneDrive - University College London\\ESDA\\BENV0092 Energy Data Analytics in the Built Environment\\data sets'

fuel_poverty_df = pd.read_csv(f'{data_path}\FuelPoverty_2019_processed.csv')


#%%
# Scatter plots of potential factors

plt.scatter(fuel_poverty_df['hh_income'], fuel_poverty_df['In_fuel_Poverty'])
plt.show()

plt.scatter(fuel_poverty_df['fuel_costs'], fuel_poverty_df['In_fuel_Poverty'])
plt.show()

plt.scatter(fuel_poverty_df['In_fuel_Poverty'], fuel_poverty_df['Head_Working_Status'])
plt.show()