# -*- coding: utf-8 -*-
"""
Tutorial XX: Estimate parameters for lumped RC model from measurement data

@author: a.stratigakos@ucl.ac.uk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import lsq_linear

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600

#%% Load measurement data

data_path = 'C:\\Users\\ucbva19\\Git projects\\energy_analytics_built_env\\data raw'
data = pd.read_csv(f'{data_path}\\10mins_solpap.csv', index_col = 0)

# Keep a subset of features
data = data[['T_External (degC)', 'P_tot (W)', 'Solar (W/m2)', 'T_Average (degC)']]

#%% Pre-processing

# Check missing data
print(data.isna().sum())

# Visualization 
data['P_tot (W)'].plot(); plt.show()
data['T_External (degC)'].plot(); plt.show()
data['Solar (W/m2)'].plot(); plt.show()
data['T_Average (degC)'].plot(); plt.show()

# Fill missing data with linear interpolation
data = data.interpolate('linear')
assert(data.isna().all().sum() == 0)

# Lead T_in values (target variable)
data['T_Average_lead_1'] = data['T_Average (degC)'].shift(-1)
data['T_diff'] = data['T_External (degC)'] - data['T_Average (degC)']

# Drop NaNs created from the shift operator
data = data.dropna()

# Create target/ feature data, set common parameters

Y = data['T_Average_lead_1']
X = data[['T_diff', 'P_tot (W)', 'Solar (W/m2)', 'T_Average (degC)']]

# discretization step at 10 minutes
dt_h = 1/6
#%% Find parameters for lumped RC model

# Solve a least-squares optimization problem to estimate RC parameters
A = X.values
b = Y.values

lsq_solution = lsq_linear(A, b, bounds = ([-np.inf, -np.inf, -np.inf, 1-1e-5], [np.inf, np.inf, np.inf, 1]) )

x_coeff = lsq_solution.x

tau = dt_h/x_coeff[0]
C = dt_h/x_coeff[1]
R = tau/C
g = x_coeff[2]/x_coeff[1]

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
print(f'g:{g}')

# To confirm, check these values against Shuwen's paper

#%% Fit a linear regression model

lr = LinearRegression(fit_intercept = False).fit(X, Y)

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

#####!!!!!!! Do these parameters make sense?
#####!!!!!!! Check the values for tau and R. Can they be negative?

#%% Test 2, using alternative data set

# Load measurement data
data_path = 'C:\\Users\\ucbva19\\Git projects\\energy_analytics_built_env\\data raw\\856978_data_and_documentation\\SMETER_P2_Energy_Temp_RH_Weather_30homes\\Meter_TemperatureRH_Weather_30homes'

# Buildings data
house = 'HH25'
data = pd.read_csv(f'{data_path}\\{house}_all.csv', index_col = 0)

print('Percentage of missing values')
print(100*(data.isna().sum()/data.shape[0]))

