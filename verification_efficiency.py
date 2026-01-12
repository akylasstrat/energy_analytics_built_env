# -*- coding: utf-8 -*-
"""
Verification of energy efficiency measures

@author: a.stratigakos@ucl.ac.uk
"""

import sys, os
cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600

# Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def accuracy_metrics(actual, predictions):
    
    actual_copy = actual.copy().reshape(-1,1)
    predictions_copy = predictions.copy().reshape(-1,1)
    
    error = actual_copy - predictions_copy
    
    assert(error.shape[0] == len(actual_copy))
    if error.ndim > 1:        
        assert(error.shape[1] == 1)
    
    rmse = np.sqrt( np.square(error).mean() )
    mae = np.abs(error).mean()
    
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    
    return rmse, mae


#%% Explore the Building Data Genome 2 (BDG2) Data-Set

bdg2_path = 'C:\\Users\\ucbva19\\Git projects\\building-data-genome-project-2'

path_meters = f"{bdg2_path}\\data\\meters\\cleaned\\"
path_meta = f"{bdg2_path}\\data\\metadata\\"
path_weather = f"{bdg2_path}\\data\\weather\\"

# Buildings data
metadata = pd.read_csv(f'{path_meta}\\metadata.csv', index_col = 0)
metadata.info()

weather_data = pd.read_csv(f'{path_weather}\\weather.csv', index_col = 0, parse_dates=True)
weather_data.info()

elec_data = pd.read_csv(f'{path_meters}\\electricity_cleaned.csv', index_col = 0, parse_dates=True)
elec_data.info()
    
weather_data = reduce_mem_usage(weather_data)
elec_data = reduce_mem_usage(elec_data)

# #%%
# # Buildings data
# weather_data.to_csv(f'{path_weather}\\weather_memory_efficient.csv')
# elec_data.to_csv(f'{path_meters}\\electricity_cleaned_memory_efficient.csv')

#%%
# Candidates for verification tutorial
# Robin_education_Takako, Rat_education_Alfonso, Rat_education_Marcos, Rat_education_Patty, Rat_public_Mark, Rat_warehouse_Eloisa

selected_building = ['Rat_education_Marcos']

# Visualize a building with break points
fig, ax = plt.subplots()
elec_data[selected_building]['01-01-2017':].plot(ax=ax)
plt.ylabel('KWh')
plt.show()

fig, ax = plt.subplots()
elec_data[selected_building].plot(ax=ax)
plt.ylabel('KWh')
plt.show()

#%%

# Create target/ feature data
Y = elec_data[selected_building]

# Keep only weather in relevant site id

selected_site_id = metadata.loc[selected_building]['site_id']

X = weather_data[weather_data.site_id == selected_site_id.values[0]][['airTemperature', 'dewTemperature', 
                                                                     'precipDepth1HR', 'seaLvlPressure', 'windDirection', 'windSpeed']]

X ['Temp_sq'] = X['airTemperature']**2

X['Hour'] = X.index.hour
X['Month'] = X.index.month
X['Day_of_week'] = X.index.weekday
X['Day_of_year'] = X.index.day_of_year

# Hour: 0–23
X["Hour_sin"] = np.sin(2 * np.pi * X["Hour"] / 24)
X["Hour_cos"] = np.cos(2 * np.pi * X["Hour"] / 24)

# Month: 1–12
X["Month_sin"] = np.sin(2 * np.pi * (X["Month"] - 1) / 12)
X["Month_cos"] = np.cos(2 * np.pi * (X["Month"] - 1) / 12)

# Day of week: typically 1–7 (adjust if your data uses 0–6)
X["DayOfWeek_sin"] = np.sin(2 * np.pi * (X["Day_of_week"] - 1) / 7)
X["DayOfWeek_cos"] = np.cos(2 * np.pi * (X["Day_of_week"] - 1) / 7)

# Day of year: 1–365 (or 366 if leap years are included)
X["DayOfYear_sin"] = np.sin(2 * np.pi * (X["Day_of_year"] - 1) / 365)
X["DayOfYear_cos"] = np.cos(2 * np.pi * (X["Day_of_year"] - 1) / 365)


Y = Y.interpolate(method='linear')
X = X.interpolate(method='linear')

# X matrix has some missing indices, need to merge on index
Y = pd.merge(Y, X, left_index=True, right_index=True)[selected_building]

# assert(Y.isna().any().values)
# assert(X.isna().sum().sum()==0)

# !!!! Feature transformations
# Wind direction: from degrees to radians
# Calendar variables: from integers to (i) one-hot encoded, (ii) sin/cos signs

#%%

# !!!! For model selection, we need to create test set in the base period as well, evaluate out-of-sample, pick model, re-train using all the data
base_period_training_start = '2016-01-01'
base_period_training_end = '2017-02-01'
base_period_testing_start = '2017-02-02'
base_period_testing_end = '2017-06-01'

X_base = X[base_period_training_start:base_period_testing_end]
Y_base = Y[base_period_training_start:base_period_testing_end]

X_report = X[base_period_testing_end:]
Y_report = Y[base_period_testing_end:]

train_X_base = X_base[base_period_training_start:base_period_training_end].values
train_Y_base = Y_base[base_period_training_start:base_period_training_end].values

test_X_base = X_base[base_period_testing_start:base_period_testing_end].values
test_Y_base = Y_base[base_period_testing_start:base_period_testing_end].values

# base_period_start = '2016-01-01'
# base_period_end = '2017-06-01'

report_period_start = '2017-07-01'

#%% Fit model

from sklearn.ensemble import ExtraTreesRegressor

# X matrix has some missing indices, need to merge on index
et_model = ExtraTreesRegressor(n_estimators = 1000)

# fit model in base period
et_model.fit(train_X_base, train_Y_base)

Y_et_pred = et_model.predict(test_X_base)

_,_ = accuracy_metrics(test_Y_base, Y_et_pred)


#%% Linear model
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
# fit model in base period
lr_model.fit(train_X_base, train_Y_base)

Y_lr_pred = lr_model.predict(test_X_base)
_,_ = accuracy_metrics(test_Y_base, Y_lr_pred)

#%% Visualize forecasts

plt.plot(test_Y_base[-2*168:], '--', label = 'Actual', color = 'black')
plt.plot(Y_et_pred[-2*168:], label = 'ET_pred')
plt.plot(Y_lr_pred[-2*168:], label = 'LR_pred')
plt.legend()
plt.show()

#%% Counterfactual estimation

# Select your model

selected_model = et_model

selected_model.fit(X_base.values, Y_base.values)

# Predict counterfactual
Y_counterfactual = selected_model.predict(X_report)

# Re-train model using all the availabe data
plt.plot(Y_counterfactual[:1000])
plt.plot(Y[report_period_start:].values[:1000])
plt.show()

#%%
# Hourly Savings = Counterfactual Baseline (Predicted) - Actual Metered Consumption
hourly_savings = Y_counterfactual.reshape(-1,1) - Y_report

print(f'Average hourly savings: {hourly_savings.values.mean()} KWh')

hourly_savings.groupby(hourly_savings.index.weekday).mean().plot()
hourly_savings.groupby(hourly_savings.index.hour).mean().plot()

