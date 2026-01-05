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

metadata = reduce_mem_usage(metadata)
weather_data = reduce_mem_usage(weather_data)
elec_data = reduce_mem_usage(elec_data)

#%%
# Candidates for verification tutorial
# Robin_education_Takako, Rat_education_Alfonso, Rat_education_Marcos, Rat_education_Patty, Rat_public_Mark, Rat_warehouse_Eloisa

selected_building = ['Rat_education_Marcos']

# Visualize a building with break points
fig, ax = plt.subplots()
elec_data[selected_building]['01-01-2017':].plot(ax=ax)
plt.ylabel('KWh')

fig, ax = plt.subplots()
elec_data[selected_building].plot(ax=ax)
plt.ylabel('KWh')

#%%

# Create target/ feature data
Y = elec_data[selected_building]

# Keep only weather in relevant site id

selected_site_id = metadata.loc[selected_building]['site_id']

X = weather_data[weather_data.site_id == selected_site_id.values[0]][['airTemperature', 'dewTemperature', 
                                                                     'precipDepth1HR', 'seaLvlPressure', 'windDirection', 'windSpeed']]

X['Hour'] = X.index.hour
X['Month'] = X.index.month
X['Day_of_week'] = X.index.weekday
X['Day_of_year'] = X.index.day_of_year


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
base_period_start = '2016-01-01'
base_period_end = '2017-06-01'

report_period_start = '2017-07-01'

#%% Fit model

from sklearn.ensemble import ExtraTreesRegressor

# X matrix has some missing indices, need to merge on index
trainX = X[base_period_start:base_period_end]
trainY = Y[base_period_start:base_period_end]

et_model = ExtraTreesRegressor(n_estimators = 500)

# fit model in base period
et_model.fit(trainX, trainY)

#%%
Y_counterfactual = et_model.predict(X[report_period_start:])

plt.plot(Y_counterfactual[:1000])
plt.plot(Y[report_period_start:].values[:1000])
plt.show()

# Hourly Savings = Counterfactual Baseline (Predicted) - Actual Metered Consumption

hourly_savings = Y_counterfactual.reshape(-1,1) - Y[report_period_start:]

#%%

print(f'Average hourly savings: {hourly_savings.values.mean()} KWh')

hourly_savings.groupby(hourly_savings.index.weekday).mean().plot()
hourly_savings.groupby(hourly_savings.index.hour).mean().plot()

