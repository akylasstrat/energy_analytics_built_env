# -*- coding: utf-8 -*-
"""
Smart meter analytics data

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

sample_buildings = ['Rat_education_Alfonso', 'Cockatoo_public_Shad']
# Visualize a building with break points
fig, ax = plt.subplots()
elec_data[sample_buildings[1]]['01-01-2017':].plot(ax=ax)
plt.ylabel('KWh')

fig, ax = plt.subplots()
elec_data[sample_buildings[1]]['03-10-2017':'03-17-2017'].plot(ax=ax)
plt.ylabel('KWh')

# for c in elec_data.columns.values:
#     if 'Rat' in c:
#         elec_data[c].plot(legend = True)
#         plt.show()
#     else: 
#         continue
#%%

# Keep only data from a specific site

target_sites = ['Robin']

selected_buildings = metadata.query(f'site_id in {target_sites}').index

# select buildings within site
elec_reduced_data = elec_data[selected_buildings]

# del elec_reduced_data['Robin_office_Erma']

# Fill missing values with linear interpolation
elec_reduced_data = elec_reduced_data.interpolate('linear')
metadata_reduced = metadata.loc[selected_buildings]

# Assert there are no NaNs
assert(elec_reduced_data.isna().all().sum() == 0)

# Features from buildings metadata to add to meters dataset
# buildings_feat = metadata[["building_id","site_id","primaryspaceusage","timezone"]]

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
sc_elec_reduced_data = scaler.fit_transform(elec_reduced_data)

sc_elec_reduced_data = pd.DataFrame(data = sc_elec_reduced_data, columns = elec_reduced_data.columns, 
                                    index = elec_reduced_data.index)
#%%

# Average daily profile per usage

for i, temp_usage in enumerate(metadata.loc[selected_buildings].primaryspaceusage.unique()):
    
    print(f'Usage:{temp_usage}')
    
    mask = metadata_reduced.primaryspaceusage == temp_usage
    if mask.sum() == 0:
        continue
    
    sub_buildings = metadata_reduced.index[mask]
    
    print(f'Number of buildings: {mask.sum()}')
    
    # Weekly profile
    fig, ax = plt.subplots()    
    elec_reduced_data['01-01-2017':][sub_buildings].groupby(elec_reduced_data['01-01-2017':].index.weekday).mean().plot(legend = False, ax=ax)
    plt.title(f'{temp_usage}: weekly average')
    plt.xlabel('Weekday')
    
    # Weekly profile
    fig, ax = plt.subplots()    
    elec_reduced_data['01-01-2017':][sub_buildings].groupby(elec_reduced_data['01-01-2017':].index.hour).mean().plot(legend = False, ax=ax)
    plt.title(f'{temp_usage}: hourly average')
    plt.xlabel('Hour of the day')
    
    # Weekly average scaled profile
    ave_profiles = sc_elec_reduced_data['01-01-2017':][sub_buildings].groupby(sc_elec_reduced_data['01-01-2017':].index.weekday).mean().mean(1)
    all_profiles = sc_elec_reduced_data['01-01-2017':][sub_buildings].groupby(sc_elec_reduced_data['01-01-2017':].index.weekday).mean().values
    fig, ax = plt.subplots()    
    plt.plot(all_profiles, color = 'gray', alpha = 0.3)
    plt.plot(ave_profiles, color = 'black', linewidth = 3)
    plt.title(f'{temp_usage}: weekly average')
    plt.xlabel('Weekday')
    
    # Daily average scaled profile
    ave_profiles = sc_elec_reduced_data['01-01-2017':][sub_buildings].groupby(sc_elec_reduced_data['01-01-2017':].index.hour).mean().mean(1)
    all_profiles = sc_elec_reduced_data['01-01-2017':][sub_buildings].groupby(sc_elec_reduced_data['01-01-2017':].index.hour).mean().values
    fig, ax = plt.subplots()    
    plt.plot(all_profiles, color = 'gray', alpha = 0.3)
    plt.plot(ave_profiles, color = 'black', linewidth = 3)
    plt.title(f'{temp_usage}: hourly average')
    plt.xlabel('Hour of the day')

#%%

elec_reduced_data['Robin_office_Victor'].plot()
plt.show()

elec_reduced_data['Robin_office_Victor']['2017-03-01':'2017-03-07'].plot()
#%%
start_date = '2017-03-01'
end_date = '2017-03-31'

# Plot daily profiles for an office
col_1 = 'Robin_office_Victor'

elec_reduced_data[col_1][start_date:end_date].groupby([elec_reduced_data[start_date:end_date].index.weekday, 
                                                           elec_reduced_data[start_date:end_date].index.hour]).mean().plot()
plt.show()

ave_daily_prof = elec_reduced_data[col_1][start_date:end_date].groupby([elec_reduced_data[start_date:end_date].index.weekday, 
                                                           elec_reduced_data[start_date:end_date].index.hour]).mean()

plt.plot(ave_daily_prof.values.reshape(-1,24).T)
plt.show()

# Plot daily profile for lodging

col_2 = 'Robin_lodging_Renea'
elec_reduced_data[col_1][start_date:end_date].groupby([elec_reduced_data[start_date:end_date].index.weekday, 
                                                           elec_reduced_data[start_date:end_date].index.hour]).mean().plot()
elec_reduced_data[col_2][start_date:end_date].groupby([elec_reduced_data[start_date:end_date].index.weekday, 
                                                           elec_reduced_data[start_date:end_date].index.hour]).mean().plot()
plt.show()

ave_daily_prof = elec_reduced_data[col_2][start_date:end_date].groupby([elec_reduced_data[start_date:end_date].index.weekday, 
                                                           elec_reduced_data[start_date:end_date].index.hour]).mean()

plt.plot(ave_daily_prof.values.reshape(-1,24).T)
plt.show()

#%%
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

# Simple forecasting application
target_building_ids = ['Robin_lodging_Renea']
weather_feat = ['airTemperature', 'dewTemperature', 'seaLvlPressure', 'windDirection',
                'windSpeed']


train_end = '2017-06-01'

for build_id in target_building_ids:
    
    # target variable
    Y = elec_reduced_data[build_id]
    
    # feature data: weather + calendar
    X = weather_data[weather_data.site_id == 'Robin'][weather_feat]
    
    X['weekday'] = X.index.weekday
    X['hour_of_day'] = X.index.hour
    X['day_of_year'] = X.index.day
    X['month'] = X.index.month
    X = X.interpolate('linear')
    
    # X matrix has some missing indices, need to merge on index
    trainX = X[:train_end]
    trainY = Y[trainX.index]

    testX = X[train_end:]
    testY = Y[testX.index]

    etree_model = ExtraTreesRegressor(n_estimators=500).fit(trainX, trainY)    
    lr_model = LinearRegression().fit(trainX, trainY)    

    Y_pred = etree_model.predict(testX)    
    
#%%
plt.plot(testY.values[800:1000])
plt.plot(Y_pred[800:1000])
plt.plot(lr_model.predict(testX)[800:1000])


