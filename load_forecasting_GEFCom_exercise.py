# -*- coding: utf-8 -*-
"""
Load forecasting GEFCom2012

@author: ucbva19
"""

import sys, os
cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600

def evaluate_accuracy(actual, predictions):
    
    actual_copy = actual.copy().reshape(-1,1)
    predictions_copy = predictions.copy().reshape(-1,1)
    
    error = actual_copy - predictions_copy
    
    assert(error.shape[0] == len(actual_copy))
    if error.ndim > 1:        
        assert(error.shape[1] == 1)
    
    rmse = np.sqrt( np.square(error).mean() )
    mae = np.abs(error).mean()
    mape = (np.abs(error)/actual_copy).mean()
    
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'MAPE: {100*mape}')
    
    return rmse, mae

def create_vanilla_predictors(meteo_df, temp_col = 'station_ave', include_h_wd = False):
    ''' Function to create the predictors for the Vanilla model. 
        Also returns a list with columns targeted for attacks and columns that cannot be attacked'''

    Predictors = meteo_df['station_ave'].to_frame()
    Predictors = Predictors.rename(columns={"station_ave": "t"})

    target_pred = []
    fixed_pred = []

    encoder = OneHotEncoder()

    if include_h_wd == True:
        Predictors[['wd_'+str(i) for i in range(7)]] = encoder.fit_transform(Predictors.index.weekday.values.astype(str).reshape(-1,1)).toarray()
        Predictors[['h_'+str(i) for i in range(24)]] = encoder.fit_transform(Predictors.index.hour.values.astype(str).reshape(-1,1)).toarray()
        fixed_pred = fixed_pred + ['wd_'+str(i) for i in range(7)] + ['h_'+str(i) for i in range(24)]

    # One-hot encoded calendar variables
    weekday_indicator = pd.DataFrame(data = encoder.fit_transform(Predictors.index.weekday.values.astype(str).reshape(-1,1)).toarray(), columns = ['wd_'+str(i) for i in range(7)])
    hour_indicator = pd.DataFrame(data = encoder.fit_transform(Predictors.index.hour.values.astype(str).reshape(-1,1)).toarray(), columns = ['h_'+str(i) for i in range(24)])
    month_indicator = pd.DataFrame(data = encoder.fit_transform(Predictors.index.month.values.astype(str).reshape(-1,1)).toarray(), columns = ['m_'+str(i) for i in range(12)])

    # Polynomial temperature and trend
    Predictors['trend'] = np.arange(len(Predictors))
    Predictors['t_2'] = np.power(Predictors['t'].values,2)
    Predictors['t_3'] = np.power(Predictors['t'].values,3)
    target_pred = target_pred + ['trend', 't', 't_2', 't_3']

    # Temperature-hour interaction
    for c in ['h_'+str(i) for i in range(24)]:
        Predictors['t_'+c] = Predictors['t'].values*hour_indicator[c].values
        Predictors['t_2_'+c] = np.power(Predictors['t'].values,2)*hour_indicator[c].values
        Predictors['t_3_'+c] = np.power(Predictors['t'].values,3)*hour_indicator[c].values
        target_pred = target_pred + ['t_'+c, 't_2_'+c, 't_3_'+c]
    # Temperature-month interaction
    for c in ['m_'+str(i) for i in range(12)]:
        Predictors['t_'+c] = Predictors['t'].values*month_indicator[c].values
        Predictors['t_2_'+c] = np.power(Predictors['t'].values,2)*month_indicator[c].values
        Predictors['t_3_'+c] = np.power(Predictors['t'].values,3)*month_indicator[c].values
        target_pred = target_pred + ['t_'+c, 't_2_'+c, 't_3_'+c]

    # Indicator for month and hour*weekday
    Predictors[['m_'+str(i) for i in range(12)]] = encoder.fit_transform(Predictors.index.month.values.astype(str).reshape(-1,1)).toarray()
    hour_day_inter = Predictors.index.weekday.astype(str)+Predictors.index.hour.astype(str)
    Predictors[['h_d_int'+str(i) for i in range(24*7)]] = encoder.fit_transform(hour_day_inter.values.reshape(-1,1)).toarray()

    fixed_pred = fixed_pred + ['m_'+str(i) for i in range(12)] + ['h_d_int'+str(i) for i in range(24*7)]

    #for c in ['h_'+str(i) for i in range(24)]: del Predictors[c]
    #for d in ['wd_'+str(i) for i in range(7)]: del Predictors[d]
    return Predictors[target_pred + fixed_pred], target_pred, fixed_pred

#%% Explore the Building Data Genome 2 (BDG2) Data-Set

gefcom2012_path = 'C:\\Users\\ucbva19\\Git projects\\energy_analytics_built_env\\data processed'

load_df = pd.read_csv(gefcom2012_path+'\\gefcom12_load.csv', index_col=0, parse_dates=True)
meteo_df = pd.read_csv(gefcom2012_path+'\\gefcom12_temperature.csv', index_col=0, parse_dates=True)
meteo_df['station_ave'] = meteo_df.mean(axis=1)

#%%
Predictors, target_pred, fixed_pred = create_vanilla_predictors(meteo_df, temp_col = 'station_ave', include_h_wd = False)

target_scaler = MinMaxScaler()
pred_scaler = MinMaxScaler()

start = '2005-01-01'
split = '2007-01-01'

zone = 21

print('Zone: ', zone)
Y = load_df['Z'+str(zone)].to_frame()


train_Y = Y[start:split].values
test_Y = Y[split:].values
target_Y = Y[split:]

train_X = Predictors[start:split].values
test_X = Predictors[split:].values


sc_train_Y = target_scaler.fit_transform(Y[start:split])
sc_test_Y = target_scaler.transform(Y[split:])

sc_train_X = pred_scaler.fit_transform(Predictors[start:split])
sc_test_X = pred_scaler.transform(Predictors[split:])


#%%%% Linear models: linear regresion, lasso, ridge

# Set the parameters by cross-validation
param_grid = {"alpha": [10**pow for pow in range(-5,6)]}

# ridge = GridSearchCV(Ridge(fit_intercept = True), param_grid)
# ridge.fit(sc_train_X, sc_train_Y)

# lasso = GridSearchCV(Lasso(fit_intercept = True), param_grid)
# lasso.fit(sc_train_X, sc_train_Y)

lr = LinearRegression(fit_intercept = True)
lr.fit(sc_train_X, sc_train_Y)

#%%

lr_pred = target_scaler.inverse_transform(lr.predict(sc_test_X).reshape(-1,1))
# lasso_pred = target_scaler.inverse_transform(lasso.predict(sc_test_X).reshape(-1,1))
# ridge_pred = target_scaler.inverse_transform(ridge.predict(sc_test_X).reshape(-1,1))


print('LR')
_, _ =  evaluate_accuracy(lr_pred.reshape(-1,1), target_Y.values)
# print('Lasso')
# _, _ =  evaluate_accuracy(lasso_pred.reshape(-1,1), target_Y.values)
# print('Ridge')
# _, _ =  evaluate_accuracy(ridge_pred.reshape(-1,1), target_Y.values)
