# -*- coding: utf-8 -*-
"""
Tutorial 1: Lumbed RC model parameter estimation from measurement data

@author: ucbva19
"""

import sys, os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600

#%%

# Load measurement data
data_path = 'C:\\Users\\ucbva19\\Git projects\\energy_analytics_built_env\\data raw'

# Buildings data
data = pd.read_csv(f'{data_path}\\10mins_solpap.csv', index_col = 0)

# Keep a subset of features
data = data[['T_External (degC)', 'P_tot (W)', 'Solar (W/m2)', 'T_Average (degC)']]


# Check missing data
print(data.isna().sum())

# Fill missing data with linear interpolation
data = data.interpolate('linear')

assert(data.isna().all().sum() == 0)

# Lead T_in values (target variable)
data['T_Average_lead_1'] = data['T_Average (degC)'].shift(-1)
data['T_diff'] = data['T_External (degC)'] - data['T_Average (degC)']

data = data.dropna()
#%%

dt_h = 1/6

Y = data['T_Average_lead_1']
X = data[['T_diff', 'P_tot (W)', 'Solar (W/m2)', 'T_Average (degC)']]

# Fit linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression(fit_intercept = False).fit(X, Y)


tau = dt_h/lr.coef_[0]

plt.plot(Y.values[:100], label = 'Actual')
plt.plot(lr.predict(X)[:100], label = 'Fitted')
plt.show()

print('In-sample RMSE')
print( np.sqrt(np.square(Y - lr.predict(X)).mean()) )

tau = dt_h/lr.coef_[0]
C = dt_h/lr.coef_[1]
R = tau/C

print('RC parameters')
print(f'tau:{tau}')
print(f'C:{C}')
print(f'R:{R}')

#!!!!!! Model parameters make not make sense here, as this not a proper RC model
#%%
from scipy.optimize import lsq_linear

A = X.values
b = Y.values

lsq_solution = lsq_linear(A, b, bounds = ([-np.inf, -np.inf, -np.inf, 1-1e-5], [np.inf, np.inf, np.inf, 1]) )

x_coeff = lsq_solution.x

tau = dt_h/x_coeff[0]
C = dt_h/x_coeff[1]
R = tau/C

T_fitted = x_coeff@A.T

plt.plot(b[:100], label = 'Actual')
plt.plot(x_coeff@A[:100].T, label = 'Fitted')
# plt.plot(x_coeff@A.T-b, label = 'Residuals')
plt.show()

print('In-sample RMSE')
print( np.sqrt(np.square(b - x_coeff@A.T).mean()) )

print('RC parameters')
print(f'tau:{tau}')
print(f'C:{C}')
print(f'R:{R}')

# Note: these agree with Shuwen's parameters

#%% Test 2, using alternative data set

# Load measurement data
data_path = 'C:\\Users\\ucbva19\\Git projects\\energy_analytics_built_env\\data raw\\856978_data_and_documentation\\SMETER_P2_Energy_Temp_RH_Weather_30homes\\Meter_TemperatureRH_Weather_30homes'

# Buildings data
house = 'HH25'
data = pd.read_csv(f'{data_path}\\{house}_all.csv', index_col = 0)

print('Percentage of missing values')
print(100*(data.isna().sum()/data.shape[0]))

