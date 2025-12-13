import sys, os
cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

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

# Base dwelling

# Pick any of the included HPXML + schedule files
hpxml_file_1 = os.path.join(default_input_path, "Input Files", "bldg0112631-up11.xml")
hpxml_file_2 = os.path.join(default_input_path, "Input Files", "bldg0112631-up00.xml")
schedule_file = os.path.join(default_input_path, "Input Files", "bldg0112631_schedule.csv")
weather_file = os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw")

print("HPXML:", hpxml_file_1)
print("Schedule:", schedule_file)
print("Weather:", weather_file)

schedule = pd.read_csv(schedule_file)
weather = pd.read_csv(weather_file)


# Define Dwelling arguments
common_dwelling_args = {
    "start_time": dt.datetime(2018, 1, 1, 0, 0),
    "time_res": dt.timedelta(minutes = 15),
    "duration": dt.timedelta(days = 30),

    "hpxml_schedule_file": schedule_file,
    "weather_file": weather_file,
    
    "verbosity": 7,
    "metrics_verbosity": 3,}

dwelling_1_args = common_dwelling_args
dwelling_1_args['hpxml_file'] = hpxml_file_1

# Create model and simulate
dwelling_1 = Dwelling(**dwelling_1_args)
# Simulate
df_1, metrics_1, _ = dwelling_1.simulate()

# Plot stacked power
fig = CreateFigures.plot_power_stack(df_1[:1000])
fig = CreateFigures.plot_power_stack(df_1)

fig = CreateFigures.plot_daily_profile(df_1, 'Total Electric Power (kW)', plot_max=True, plot_min=True)

# Dwelling 2
dwelling_2_args = common_dwelling_args
dwelling_2_args['hpxml_file'] = hpxml_file_2

# Create model and simulate
dwelling_2 = Dwelling(**dwelling_2_args)
# Simulate
df_2, metrics_2, _ = dwelling_2.simulate()

# Plot stacked power
fig = CreateFigures.plot_power_stack(df_2[:1000])
fig = CreateFigures.plot_power_stack(df_2)

#%%
fig, ax = plt.subplots()
for df, lab in [(df_1,"dwelling_1"), (df_2,"dwelling_2")]:
    ax.plot(df.index, df.get("Total Electric Power (kW)", np.nan), label=f"{lab} elec (kW)")
ax.set_ylabel("kW"); ax.legend(); ax.set_title("Total electric demand")

fig, ax = plt.subplots()
for df, lab in [(df_1,"dwelling_1"), (df_2,"dwelling_2")]:
    if "Total Gas Power (therms/hour)" in df:
        ax.plot(df.index, df["Total Gas Power (therms/hour)"], label=f"{lab} gas (therms/h)")
ax.set_ylabel("therms/h"); ax.legend(); ax.set_title("Total gas demand")


fig, ax = plt.subplots()
ax.plot(df_1.index, df_1.get("HVAC Heating Delivered (W)", np.nan), label="dwelling_1 delivered (W)")
ax.plot(df_2.index, df_2.get("HVAC Heating Delivered (W)", np.nan), label="dwelling_2 delivered (W)")
ax.set_ylabel("W"); ax.legend(); ax.set_title("Heating delivered (service)")

fig, ax = plt.subplots()
ax.plot(df_1.index, df_1.get("HVAC Heating Electric Power (kW)", np.nan), label="dwelling_1 HVAC elec (kW)")
ax.plot(df_2.index, df_2.get("HVAC Heating Gas Power (therms/hour)", np.nan), label="dwelling_2 HVAC gas (therms/h)")
ax.legend(); ax.set_title("Energy input to deliver heating")


#%%

Tout = 'Temperature - Outdoor (C)'
Tin  = 'Temperature - Indoor (C)'

fig, ax = plt.subplots()
ax.scatter(df_1[Tout], df_1.get("HVAC Heating Electric Power (kW)", np.nan), s=4, alpha=0.3, label="dwelling_1 (HP elec)")
ax.scatter(df_2[Tout], df_2.get("HVAC Heating Gas Power (therms/hour)", np.nan), s=4, alpha=0.3, label="dwelling_2 (gas)")
ax.set_xlabel("Outdoor temp (C)"); ax.set_ylabel("Heating input (kW or therms/h)")
ax.legend(); ax.set_title("Heating input vs outdoor temperature")


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

plot_stack(df_1, "dwelling_1: electric end-use stack")
plot_stack(df_2, "dwelling_2: electric end-use stack")

#%%
fig, ax = plt.subplots()
ax.plot(df_1.index, df_1["Total Electric Energy (kWh)"], label="dwelling_1 cum kWh")
ax.plot(df_2.index, df_2["Total Electric Energy (kWh)"], label="dwelling_2 cum kWh")
ax.set_ylabel("kWh"); ax.legend(); ax.set_title("Cumulative electric energy")

if "Total Gas Energy (therms)" in df_2.columns or "Total Gas Energy (therms)" in df_1.columns:
    fig, ax = plt.subplots()
    if "Total Gas Energy (therms)" in df_1: ax.plot(df_1.index, df_1["Total Gas Energy (therms)"], label="dwelling_1 cum therms")
    if "Total Gas Energy (therms)" in df_2: ax.plot(df_2.index, df_2["Total Gas Energy (therms)"], label="dwelling_2 cum therms")
    ax.set_ylabel("therms"); ax.legend(); ax.set_title("Cumulative gas energy")


fig, ax = plt.subplots()
ax.plot(np.sort(df_1["Total Electric Power (kW)"].values)[::-1], label="dwelling_1")
ax.plot(np.sort(df_2["Total Electric Power (kW)"].values)[::-1], label="dwelling_2")
ax.set_xlabel("Ranked timesteps"); ax.set_ylabel("kW")
ax.legend(); ax.set_title("Load duration curve (total electric)")

#%%

# ------------------------------------------
# Same building and equipment, different occupants

# Base dwelling

# Pick any of the included HPXML + schedule files
hpxml_file_1 = os.path.join(default_input_path, "Input Files", "bldg0112631-up11.xml")
hpxml_file_2 = os.path.join(default_input_path, "Input Files", "bldg0112631-up00.xml")
schedule_file = os.path.join(default_input_path, "Input Files", "bldg0112631_schedule.csv")
weather_file = os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw")

print("HPXML:", hpxml_file_1)
print("Schedule:", schedule_file)
print("Weather:", weather_file)

schedule = pd.read_csv(schedule_file)

upd_schedule = schedule.copy()
upd_schedule['occupants'] = 3.0

upd_schedule.to_csv('upd_bldg0112631_schedule.csv', index = False)
weather = pd.read_csv(weather_file)

upd_schedule_file = f'{cd}\\upd_bldg0112631_schedule.csv'

upd_dwelling_1_args = dwelling_1_args
upd_dwelling_1_args['hpxml_schedule_file'] = upd_schedule_file

# Create model and simulate
upd_dwelling_1 = Dwelling(**upd_dwelling_1_args)
# Simulate
upd_df_1, upd_metrics_1, _ = upd_dwelling_1.simulate()

#%%

fig = CreateFigures.plot_power_stack(df_1[:1000])
fig = CreateFigures.plot_power_stack(upd_df_1[:1000])


fig = CreateFigures.plot_daily_profile(df_1, 'Total Electric Power (kW)', plot_max=True, plot_min=True)
fig = CreateFigures.plot_daily_profile(upd_df_1, 'Total Electric Power (kW)', plot_max=True, plot_min=True)











