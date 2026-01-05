# -*- coding: utf-8 -*-
"""
Simulating residential building demand using OCHRE

@author: ucbva19
"""

import sys, os
import glob
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ochre import Dwelling, CreateFigures, Analysis
from ochre.utils import default_input_path

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600

#%%

# Pick any of the included HPXML + schedule files for building data
hpxml_file_1 = os.path.join(default_input_path, "Input Files", "bldg0112631-up11.xml")
hpxml_file_2 = os.path.join(default_input_path, "Input Files", "bldg0112631-up00.xml")
schedule_file = os.path.join(default_input_path, "Input Files", "bldg0112631_schedule.csv")
weather_file = os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw")

print("HPXML:", hpxml_file_1)
print("Schedule:", schedule_file)
print("Weather:", weather_file)

schedule = pd.read_csv(schedule_file)
weather = pd.read_csv(weather_file)


# Define arguments for dwelling object, common across all examples
common_dwelling_args = {
    "start_time": dt.datetime(2018, 1, 1, 0, 0),
    "time_res": dt.timedelta(minutes = 15),
    "duration": dt.timedelta(days = 365),

    "hpxml_schedule_file": schedule_file,
    "weather_file": weather_file,    
    "verbosity": 7,
    "metrics_verbosity": 3,}

# Base dwelling: base case with electric heat pump
dwelling_elec_args = common_dwelling_args.copy()
dwelling_elec_args['hpxml_file'] = hpxml_file_1

# Create model and simulate
dwelling_elec = Dwelling(**dwelling_elec_args)
df_elec, metrics_elec, _ = dwelling_elec.simulate()


# Plot stacked power
fig = CreateFigures.plot_power_stack(df_elec[500:1000])
fig = CreateFigures.plot_power_stack(df_elec)
fig = CreateFigures.plot_daily_profile(df_elec, 'Total Electric Power (kW)', plot_max=True, plot_min=True)

# Dwelling 2: base case with **gas** boiler (same arguments, different hpxml file)
dwelling_gas_args = common_dwelling_args.copy()
dwelling_gas_args['hpxml_file'] = hpxml_file_2

# Create model and simulate
dwelling_gas = Dwelling(**dwelling_gas_args)
df_gas, metrics_gas, _ = dwelling_gas.simulate()

# Plot stacked power
fig = CreateFigures.plot_power_stack(df_gas[:1000])
fig = CreateFigures.plot_power_stack(df_gas)
fig = CreateFigures.plot_daily_profile(df_gas, 'Total Electric Power (kW)', plot_max=True, plot_min=True)

#%%

# Some basic plots to explore results
fig, ax = plt.subplots()
for df, lab in [(df_elec,"dwelling_elec"), (df_elec,"dwelling_gas")]:
    ax.plot(df.index, df.get("Total Electric Power (kW)", np.nan), label=f"{lab} elec (kW)")
ax.set_ylabel("kW"); ax.legend(); ax.set_title("Total electric demand")

fig, ax = plt.subplots()
for df, lab in [(df_elec,"dwelling_elec"), (df_elec,"dwelling_gas")]:
    if "Total Gas Power (therms/hour)" in df:
        ax.plot(df.index, df["Total Gas Power (therms/hour)"], label=f"{lab} gas (therms/h)")
ax.set_ylabel("therms/h"); ax.legend(); ax.set_title("Total gas demand")

fig, ax = plt.subplots()
ax.plot(df_elec.index, df_elec.get("HVAC Heating Delivered (W)", np.nan), label="dwelling_elec delivered (W)")
ax.plot(df_gas.index, df_gas.get("HVAC Heating Delivered (W)", np.nan), label="dwelling_gas delivered (W)")
ax.set_ylabel("W"); ax.legend(); ax.set_title("Heating delivered (service)")

fig, ax = plt.subplots()
ax.plot(df_elec.index, df_elec.get("HVAC Heating Electric Power (kW)", np.nan), label="dwelling_elec HVAC elec (kW)")
ax.plot(df_gas.index, df_gas.get("HVAC Heating Gas Power (therms/hour)", np.nan), label="dwelling_gas HVAC gas (therms/h)")
ax.legend(); ax.set_title("Energy input to deliver heating")

# Scatterplot of heating input vs outdoor temperature to maintain the same internal temperature
Tout = 'Temperature - Outdoor (C)'
Tin  = 'Temperature - Indoor (C)'

fig, ax = plt.subplots(ncols = 2, sharex=True, figsize = (10, 4))
ax[0].scatter(df_elec[Tout], df_elec.get("HVAC Heating Electric Power (kW)", np.nan), s=4, alpha=0.3, label="dwelling_elec")
ax[0].set_xlabel("Outdoor temp (C)"); 
ax[0].set_ylabel("Heating input (kW)")
ax[0].legend();

ax[1].scatter(df_gas[Tout], df_gas.get("HVAC Heating Gas Power (therms/hour)", np.nan), s=4, alpha=0.3, label="dwelling_gas")
ax[1].set_xlabel("Outdoor temp (C)"); 
ax[1].set_ylabel("Heating input (therms/hour)")
ax[1].legend();

#%%
def plot_stack(df, title):
    cols = [c for c in [
        "HVAC Heating Electric Power (kW)",
        "HVAC Cooling Electric Power (kW)",
        "Water Heating Electric Power (kW)",
        "Other Electric Power (kW)",
        "Indoor Lighting Electric Power (kW)",
        "MELs Electric Power (kW)"
    ] if c in df.columns]
    fig, ax = plt.subplots()
    ax.stackplot(df.index, [df[c].values for c in cols], labels=cols)
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.set_ylabel("kW"); ax.set_title(title)

plot_stack(df_elec[:500], "dwelling_elec: electric end-use stack")
plot_stack(df_gas[:500], "dwelling_gas: electric end-use stack")

#%%

fig, ax = plt.subplots()
ax.plot(df_elec.index, df_elec["Total Electric Energy (kWh)"], label="dwelling_elec cum kWh")
ax.plot(df_gas.index, df_gas["Total Electric Energy (kWh)"], label="dwelling_gas gas kWh")
ax.set_ylabel("kWh"); ax.legend(); ax.set_title("Cumulative electric energy")

if "Total Gas Energy (therms)" in df_gas.columns or "Total Gas Energy (therms)" in df_elec.columns:
    fig, ax = plt.subplots()
    if "Total Gas Energy (therms)" in df_elec: ax.plot(df_elec.index, df_elec["Total Gas Energy (therms)"], label="dwelling_elec cum therms")
    if "Total Gas Energy (therms)" in df_gas: ax.plot(df_gas.index, df_gas["Total Gas Energy (therms)"], label="dwelling_gas cum therms")
    ax.set_ylabel("therms"); ax.legend(); ax.set_title("Cumulative gas energy")


fig, ax = plt.subplots()
ax.plot(np.sort(df_elec["Total Electric Power (kW)"].values)[::-1], label="dwelling_elec")
ax.plot(np.sort(df_gas["Total Electric Power (kW)"].values)[::-1], label="dwelling_gas")
ax.set_xlabel("Ranked timesteps"); ax.set_ylabel("kW")
ax.legend(); ax.set_title("Load duration curve (total electric)")

#%%%%%%%%%%%%%%%%%%%%%
############# Add new equipment to dwelling_elec

# Add PV, estimate reductions
new_equipment = { "PV": {"capacity": 5,}}
dwelling_elec_PV_args = {**dwelling_elec_args,  "Equipment": new_equipment,}

# Create new dwelling model and simulate
dwelling_elec_PV = Dwelling(**dwelling_elec_PV_args)
df_elec_PV, metrics_elec_PV, _ = dwelling_elec_PV.simulate()

# estimate the effect on the average profile
fig = CreateFigures.plot_daily_profile(df_elec[:'2018-01-30'], 'Total Electric Power (kW)', plot_max=True, plot_min=True)
fig = CreateFigures.plot_daily_profile(df_elec_PV[:'2018-01-30'], 'Total Electric Power (kW)', plot_max=True, plot_min=True)

#%%
fig, ax = plt.subplots(ncols = 2, figsize = (10,4), sharey = True)

plt.sca(ax[0])
df_elec[:'2018-01-30'].groupby(df_elec[:'2018-01-30'].index.hour)['Total Electric Power (kW)'].mean().plot(ax = ax[0], label = 'Baseline')
df_elec_PV[:'2018-01-30'].groupby(df_elec_PV[:'2018-01-30'].index.hour)['Total Electric Power (kW)'].mean().plot(linestyle = '-.', 
                                                                                                        ax = ax[0], color = 'tab:orange', label = 'Baseline-PV')
plt.xlabel('Hour of day')
plt.ylabel('Total Electric Power (kW)')
plt.title('Winter')

df_elec['2018-06-01':'2018-09-30'].groupby(df_elec['2018-06-01':'2018-09-30'].index.hour)['Total Electric Power (kW)'].mean().plot(ax = ax[1], label = 'Baseline')
df_elec_PV['2018-06-01':'2018-09-30'].groupby(df_elec_PV['2018-06-01':'2018-09-30'].index.hour)['Total Electric Power (kW)'].mean().plot(linestyle = '-.', 
                                                                                                                                         ax = ax[1], color = 'tab:orange', label = 'With PV')
plt.sca(ax[1])
plt.xlabel('Hour of day')
plt.title('Summer')
plt.legend()
plt.show()

#%% Add PV + Battery storage system

new_equipment = { "PV": {"capacity": 5,}, 
                  "Battery": {'capacity_kwh':10, 'capacity':5, 'self_consumption_mode': True,}}

dwelling_elec_PV_BESS_args = {**dwelling_elec_PV_args,  "Equipment": new_equipment,}

# Create dwelling model
dwelling_elec_PV_BESS = Dwelling(**dwelling_elec_PV_BESS_args)
df_elec_PV_BESS, metrics_elec_PV_BESS, _ = dwelling_elec_PV_BESS.simulate()

fig, ax = plt.subplots(ncols = 2, figsize = (10,4), sharey = True)

fig, ax = plt.subplots(ncols = 2, figsize = (10,4), sharey = True)

plt.sca(ax[0])
df_elec[:'2018-01-30'].groupby(df_elec[:'2018-01-30'].index.hour)['Total Electric Power (kW)'].mean().plot(ax = ax[0], label = 'Baseline')
df_elec_PV[:'2018-01-30'].groupby(df_elec_PV[:'2018-01-30'].index.hour)['Total Electric Power (kW)'].mean().plot(linestyle = '-.', 
                                                                                                        ax = ax[0], color = 'tab:orange', label = 'Baseline-PV')
df_elec_PV_BESS[:'2018-01-30'].groupby(df_elec_PV[:'2018-01-30'].index.hour)['Total Electric Power (kW)'].mean().plot(linestyle = '--', ax = ax[0], color = 'black', 
                                                                                                                      label = 'With PV+BESS')
plt.xlabel('Hour of day')
plt.ylabel('Total Electric Power (kW)')
plt.title('Winter')

df_elec['2018-06-01':'2018-09-30'].groupby(df_elec['2018-06-01':'2018-09-30'].index.hour)['Total Electric Power (kW)'].mean().plot(ax = ax[1], label = 'Baseline')
df_elec_PV['2018-06-01':'2018-09-30'].groupby(df_elec_PV['2018-06-01':'2018-09-30'].index.hour)['Total Electric Power (kW)'].mean().plot(linestyle = '-.', 
                                                                                                                                         ax = ax[1], color = 'tab:orange', label = 'With PV')
df_elec_PV_BESS['2018-06-01':'2018-09-30'].groupby(df_elec_PV['2018-06-01':'2018-09-30'].index.hour)['Total Electric Power (kW)'].mean().plot(linestyle = '--', ax = ax[1], 
                                                                                                                                              color = 'black', label = 'With PV+BESS')
plt.sca(ax[1])
plt.xlabel('Hour of day')
plt.title('Summer')
plt.legend()
plt.show()







