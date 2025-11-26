# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 12:13:38 2025

@author: ucbva19
"""
import os
import datetime as dt

import matplotlib.pyplot as plt

from ochre import Dwelling
from ochre.utils import default_input_path  # for using sample files

# Plotting figures default
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
# plt.rcParams['figure.figsize'] = (4,4) # Height can be changed
# plt.rcParams['font.size'] = 7
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
# plt.rcParams["mathtext.fontset"] = 'dejavuserif'

#%%
dwelling_args = {
    # Timing parameters
    "start_time": dt.datetime(2018, 1, 1, 0, 0),  # (year, month, day, hour, minute)
    "time_res": dt.timedelta(minutes=10),         # time resolution of the simulation
    "duration": dt.timedelta(days=3),             # duration of the simulation

    # Input files
    "hpxml_file": os.path.join(default_input_path, "Input Files", "bldg0112631-up11.xml"),
    "hpxml_schedule_file": os.path.join(default_input_path, "Input Files", "bldg0112631_schedule.csv"),
    "weather_file": os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw"),
}

# Create Dwelling model
dwelling = Dwelling(**dwelling_args)


df, metrics, hourly = dwelling.simulate()

#%%
from ochre import Analysis

# calculate metrics from the time series results
metrics2 = Analysis.calculate_metrics(df)

#%%
from ochre import CreateFigures

# Plot results
fig = CreateFigures.plot_power_stack(df)


#%%
# Average daily profile
fig = CreateFigures.plot_daily_profile(df, 'Total Electric Power (kW)', plot_max=False, plot_min=False)

#%%
df['Hour'] = df.index.hour
df['Minute_hour'] = df.index.minute

fig, ax = plt.subplots()            
df.groupby(['Hour', 'Minute_hour'])['Total Electric Power (kW)'].mean().plot(ax=ax)
plt.title('Average daily profile')

#%%


#%% Simulating single piece of equipment

from ochre import ElectricVehicle

equipment_args = {
    "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    "time_res": dt.timedelta(minutes=15),
    "duration": dt.timedelta(days=10),
    "save_results": False,  # if True, must specify output_path
    # "output_path": os.getcwd(),
    "seed": 1,  # setting random seed to create consistent charging events

    # Equipment-specific parameters
    "vehicle_type": "BEV",
    "charging_level": "Level 1",
    "range": 200,
}

# Initialize equipment
equipment = ElectricVehicle(**equipment_args)

# Simulate equipment
df = equipment.simulate()

df.head()

fig = CreateFigures.plot_daily_profile(df, "EV Electric Power (kW)", plot_max=False, plot_min=False)

fig = CreateFigures.plot_time_series_detailed((df["EV SOC (-)"],))

#%% Simulating a water heater

import numpy as np
import pandas as pd
from ochre import ElectricResistanceWaterHeater

# Create water draw schedule
start_time = dt.datetime(2018, 1, 1, 0, 0)  # year, month, day, hour, minute
time_res = dt.timedelta(minutes=5)
duration = dt.timedelta(days=10)
times = pd.date_range(
    start_time,
    start_time + duration,
    freq=time_res,
    inclusive="left",
)
water_draw_magnitude = 12  # L/min
withdraw_rate = np.random.choice([0, water_draw_magnitude], p=[0.99, 0.01], size=len(times))
schedule = pd.DataFrame(
    {
        "Water Heating (L/min)": withdraw_rate,
        "Zone Temperature (C)": 20,
        "Mains Temperature (C)": 7,
    },
    index=times,
)

equipment_args = {
    "start_time": start_time,  # year, month, day, hour, minute
    "time_res": time_res,
    "duration": duration,
    "save_results": False,  # if True, must specify output_path
    # "output_path": os.getcwd(),
    # Equipment-specific parameters
    "Setpoint Temperature (C)": 51,
    "Tank Volume (L)": 250,
    "Tank Height (m)": 1.22,
    "UA (W/K)": 2.17,
    "schedule": schedule,
    }

# Initialize equipment
wh = ElectricResistanceWaterHeater(**equipment_args)

# Run simulation
df = wh.simulate()

# Show results
df.head()

fig = CreateFigures.plot_daily_profile(
    df, "Water Heating Electric Power (kW)", plot_max=False, plot_min=False
)

fig = CreateFigures.plot_time_series_detailed((df["Hot Water Outlet Temperature (C)"],))









