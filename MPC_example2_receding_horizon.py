# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 23:53:11 2025

@author: ucbva19
"""

import os
import datetime as dt

import pandas as pd
import matplotlib.pyplot as plt

from ochre import Dwelling, Analysis
from ochre.utils import default_input_path
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Plotting figures default
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
# plt.rcParams['figure.figsize'] = (4,4) # Height can be changed
# plt.rcParams['font.size'] = 7
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
# plt.rcParams["mathtext.fontset"] = 'dejavuserif'

# 3) Simple time-of-use price builder
def build_price_profile(time_index, off=0.08, mid=0.20, peak=0.50):
    """
    Given a pandas DatetimeIndex, build a 1D price array with:
      - 'off' during night,
      - 'mid' during working hours,
      - 'peak' during early evening.
    Adjust hours as you like.
    """
    price = np.zeros(len(time_index))
    
    for k, t in enumerate(time_index):
        h = t.hour
        if 17 <= h < 20:
            price[k] = peak
        elif 7 <= h < 17:
            price[k] = mid
        else:
            price[k] = off
    return price

def rc_from_alpha_gamma(alpha, gamma, dt_minutes=10):
    """
    Given discrete-time coefficients alpha, gamma (for a 1R-1C model)
    and the time step in minutes, return:
      R_th [°C/kW], C_th [kWh/°C], tau [h]
    satisfying:
      alpha = dt_h / tau
      gamma = dt_h / C_th
      tau = R_th * C_th
    """
    dt_h = dt_minutes / 60.0
    C_th = dt_h / gamma        # kWh/°C
    tau = dt_h / alpha         # h
    R_th = tau / C_th          # °C/kW
    return R_th, C_th, tau

#%%
# --------------------------------------------------------------
# 1) Use local sample files that ship with OCHRE
# --------------------------------------------------------------

# Explore what's available
print("Files under default_input_path:")
for root, dirs, files in os.walk(default_input_path):
    for f in files:
        print(os.path.join(root, f))

# Pick any of the included HPXML + schedule files
hpxml_file = os.path.join(default_input_path, "Input Files", "bldg0112631-up11.xml")
schedule_file = os.path.join(default_input_path, "Input Files", "bldg0112631_schedule.csv")
weather_file = os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw")

print("HPXML:", hpxml_file)
print("Schedule:", schedule_file)
print("Weather:", weather_file)

#%%
schedule = pd.read_csv(schedule_file)
weather = pd.read_csv(weather_file)

#%%
# --------------------------------------------------------------
# 2) Define Dwelling arguments
# --------------------------------------------------------------
dwelling_args = {
    "start_time": dt.datetime(2018, 1, 1, 0, 0),
    "time_res": dt.timedelta(minutes=10),
    "duration": dt.timedelta(days=10),

    "hpxml_file": hpxml_file,
    "hpxml_schedule_file": schedule_file,
    "weather_file": weather_file,
    # "schedule": schedule,
    # # Envelope: set initial indoor temperature
    # "Envelope": {
    #     "initial_temp_setpoint": 17.0  # °C, initial indoor zone temperature
    # },

    "verbosity": 7,
    "metrics_verbosity": 3,
}

# --------------------------------------------------------------
# 3) Simulate
# --------------------------------------------------------------
house = Dwelling(**dwelling_args)
df, metrics, hourly = house.simulate()

print("Simulation output:", df.columns[:12])
df.head()

# drop the first day to avoid outliers
df = df.iloc[144:]
#%%
# -------------------------------------------------------------------
# 4) Identify key columns
# -------------------------------------------------------------------
# Try to find the right columns by partial name
indoor_temp_col = [c for c in df.columns if "Temperature - Indoor" in c][0]
outdoor_temp_col = [c for c in df.columns if "Temperature - Outdoor" in c or "Outdoor Dry Bulb" in c][0]
heating_power_col = [c for c in df.columns if "HVAC Heating Main Power" in c or "HVAC Heating Power" in c][0]

print("Indoor temp column:", indoor_temp_col)
print("Outdoor temp column:", outdoor_temp_col)
print("Heating power column:", heating_power_col)

T_in  = df[indoor_temp_col].values          # °C
T_out = df[outdoor_temp_col].values         # °C
u     = df[heating_power_col].values        # kW
t_idx = df.index

#%%
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(t_idx, T_in, label="Indoor (°C)")
ax1.plot(t_idx, T_out, label="Outdoor (°C)", alpha=0.7)
ax1.set_ylabel("Temperature (°C)")
ax1.set_xlabel("Time")
ax1.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))
plt.step(t_idx, u, where="post")
plt.ylabel("Heating Power (kW)")
plt.xlabel("Time")
plt.title("HVAC Heating Power")
plt.tight_layout()
plt.show()

#%%######################## MPC part of the module

# -------------------------------------------------------------------
# 1) Discrete-time RC-ish model (from Lecture 1)
# -------------------------------------------------------------------

dt_minutes = 10
dt_hours = dt_minutes / 60

# Pick 1 day from the OCHRE simulation for our MPC horizon
start = 24 * 6        # e.g. skip the first day to avoid warm-up; adjust as needed
N_steps = 24 * 6      # 24 hours * 6 steps/hour = 144 steps at 10-min

T_out_MPC = T_out[start:start+N_steps]
time_index = t_idx[start:start+N_steps]

# Comfort band and bounds
T_min = 20.0
T_max = 23.5
u_min = 0.0
u_max = 8.0   # kW, adjust as needed

T0 = 21.0     # initial indoor temp for MPC model


# -------------------------------------------------------------------
# 2) Horizon and price profile
# -------------------------------------------------------------------
N_hours = 24
steps_per_hour = int(60 / dt_minutes)
N = N_hours * steps_per_hour

time_index = pd.date_range("2018-01-02", periods=N, freq=f"{dt_minutes}min")  # example date

# Time-of-use and fixed prices
tou_price = build_price_profile(time_index)
fxd_price =  0.20*np.ones(N_steps)

#%%

# -------------------------------------------------------------------
# 3) MPC problem (single-day open-loop optimization)
# -------------------------------------------------------------------
# --- RC model parameters ---
alpha = 0.0208   # envelope leak (10-min step)
gamma = 0.16     # heating gain (°C per kW per 10-min)
dt_minutes = 10
dt_hours = dt_minutes / 60

R_th, C_th, tau = rc_from_alpha_gamma(alpha, gamma)
print(f"R_th = {R_th:.2f} °C/kW, C_th = {C_th:.2f} kWh/°C, tau = {tau:.1f} h")

############ Problem formulation
# Decision variables
T = cp.Variable(N_steps + 1)
u = cp.Variable(N_steps)
slack = cp.Variable(N_steps)
# slack_neg = cp.Variable(N_steps, nonneg=True)

constraints = [T[0] == T0]

for k in range(N_steps):
    # dynamics with known disturbance T_out_MPC[k]
    constraints += [
        T[k+1] == T[k] + alpha*(T_out_MPC[k] - T[k]) + gamma*u[k],
        T_min <= T[k+1],
        T[k+1] <= T_max,
        u_min <= u[k],
        u[k] <= u_max,
    ]

# Cost: sum(price * power * dt)  -> € or £
cost = cp.sum(cp.multiply(tou_price, u)) * dt_hours

prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve()

print("Status:", prob.status)
print("Optimal cost:", prob.value)

T_opt = T.value
u_opt = u.value

# Temperature trajectory
plt.figure(figsize=(10,4))
plt.plot(time_index, T_opt[1:], label="Indoor temp")
plt.axhline(T_min, color="grey", linestyle="--", label="Comfort band")
plt.axhline(T_max, color="grey", linestyle="--")
plt.ylabel("Temperature (°C)")
plt.xlabel("Time")
plt.legend()
plt.title("MPC: Indoor temperature trajectory")
plt.tight_layout()
plt.show()

# Control and price
fig, ax1 = plt.subplots(figsize=(10,4))
ax1.step(time_index, u_opt, where="post", label="Heating power (kW)")
ax1.set_ylabel("Heating power (kW)")
ax1.set_xlabel("Time")

ax2 = ax1.twinx()
ax2.plot(time_index, tou_price, "--", alpha=0.7, label="Price")
ax2.set_ylabel("Price")

fig.legend(loc="upper right")
plt.title("MPC: Heating vs price")
plt.tight_layout()
plt.show()

# %% Want to try a lot of experiments, let's add this as a function

def solve_mpc_rc(T0, T_out, price, alpha, gamma, T_min, T_max, u_min, u_max,
    dt_minutes=10):
    
    T_out = np.asarray(T_out)
    price = np.asarray(price)
    N = len(T_out)
    dt_h = dt_minutes / 60.0

    T = cp.Variable(N + 1)
    u = cp.Variable(N)
    # s_plus = cp.Variable(N, nonneg=True)
    # s_minus = cp.Variable(N, nonneg=True)

    constraints = [T[0] == T0]

    for k in range(N):
        constraints += [
            T[k+1] == T[k] + alpha * (T_out[k] - T[k]) + gamma * u[k],
            u_min <= u[k],
            u[k] <= u_max,
            T_min <= T[k+1],
            T[k+1] <= T_max,
            ]

    energy_cost = cp.sum(cp.multiply(price, u)) * dt_h
    cost = energy_cost

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    
    print("Status:", prob.status)
    print("Optimal cost:", prob.value)
    
    return T.value, u.value, prob

def solve_mpc_rc_soft(T0, T_out, price, alpha, gamma, T_min, T_max, u_min, u_max,
    dt_minutes=10, lambda_slack = 10):
    
    T_out = np.asarray(T_out)
    price = np.asarray(price)
    N = len(T_out)
    dt_h = dt_minutes / 60.0

    T = cp.Variable(N + 1)
    u = cp.Variable(N)
    s_plus = cp.Variable(N, nonneg=True)
    s_minus = cp.Variable(N, nonneg=True)

    constraints = [T[0] == T0]

    for k in range(N):
        constraints += [
            T[k+1] == T[k] + alpha * (T_out[k] - T[k]) + gamma * u[k],
            u_min <= u[k],
            u[k] <= u_max,
            T_min - s_minus[k] <= T[k+1],
            T[k+1] <= T_max + s_plus[k],
            ]

    comfort_penalty = lambda_slack * (cp.sum_squares(s_plus) + cp.sum_squares(s_minus))
    energy_cost = cp.sum(cp.multiply(price, u)) * dt_h

    cost = energy_cost + comfort_penalty
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    
    print("Status:", prob.status)
    print("Optimal cost:", prob.value)
    
    return T.value, u.value, prob
#%% Assess the effect of prices

# solve with fixed price
T_fxd, u_fxd, prob_fxd = solve_mpc_rc(T0, T_out[:N_steps], fxd_price, alpha, gamma, T_min, T_max, 
                                      u_min, u_max, dt_minutes=10)


# solve with ToU price
T_tou, u_tou, prob_tou = solve_mpc_rc(T0, T_out[:N_steps], tou_price, alpha, gamma, T_min, T_max, 
                                      u_min, u_max, dt_minutes=10)

#%%
plt.figure(figsize=(10,4))
plt.plot(time_index, T_fxd[1:], label="Indoor temp (fixed)")
plt.plot(time_index, T_tou[1:], label="Indoor temp (ToU)")
plt.axhline(T_min, color="grey", linestyle="--", label="Comfort band")
plt.axhline(T_max, color="grey", linestyle="--")
plt.ylabel("Temperature (°C)")
plt.xlabel("Time")
plt.legend()
plt.title("MPC: Indoor temperature trajectory")
plt.tight_layout()
plt.show()


# Control and price
fig, ax1 = plt.subplots(figsize=(10,4))
ax1.step(time_index, u_fxd, where="post", label="Heating power, fixed price (kW)")
ax1.step(time_index, u_tou, where="post", label="Heating power, ToU price (kW)")
ax1.set_ylabel("Heating power (kW)")
ax1.set_xlabel("Time")

ax2 = ax1.twinx()
ax2.plot(time_index, tou_price, "--", alpha=0.7, label="ToU Price")
ax2.plot(time_index, fxd_price, "--", alpha=0.7, label="Fixed Price")
ax2.set_ylabel("Price")

fig.legend(loc="upper right")
plt.title("MPC: Heating vs price")
plt.tight_layout()
plt.show()

#%% What happens in extreme case

# MPC with hard constraints
_, _, _= solve_mpc_rc(T0, -10*T_out[:N_steps], tou_price, alpha, gamma, T_min, T_max, 
                                      u_min, u_max, dt_minutes=10)

# MPC with soft constraints and slacks
_, _, _= solve_mpc_rc_soft(T0, -10*T_out[:N_steps], tou_price, alpha, gamma, T_min, T_max, 
                                      u_min, u_max, dt_minutes=10, lambda_slack = 10)

# Now we can solve the problem

#%% Let's revisit the previous problem with soft penalties, consider ToU prices

# MPC with soft constraints and slacks
lambda_slack = .01

T_tou_soft, u_tou_soft, prob_tou_soft = solve_mpc_rc_soft(T0, T_out[:N_steps], tou_price, alpha, gamma, T_min, T_max, 
                                      u_min, u_max, dt_minutes=10, lambda_slack = lambda_slack)

# Note that optimal costs may not be comparable anymore


plt.figure(figsize=(10,4))
plt.plot(time_index, T_tou_soft[1:], label="Indoor temp (ToU, soft)")
plt.plot(time_index, T_tou[1:], label="Indoor temp (ToU)")
plt.axhline(T_min, color="grey", linestyle="--", label="Comfort band")
plt.axhline(T_max, color="grey", linestyle="--")
plt.ylabel("Temperature (°C)")
plt.xlabel("Time")
plt.legend()
plt.title(f"MPC: Indoor temperature trajectory, lambda = {lambda_slack}")
plt.tight_layout()
plt.show()


# Control and price
fig, ax1 = plt.subplots(figsize=(10,4))
ax1.step(time_index, u_tou, where="post", label="Heating power, ToU price (kW)")
ax1.step(time_index, u_tou_soft, where="post", label="Heating power, ToU, soft (kW)")
ax1.set_ylabel("Heating power (kW)")
ax1.set_xlabel("Time")

ax2 = ax1.twinx()
ax2.plot(time_index, tou_price, "--", alpha=0.7, label="ToU Price")
# ax2.plot(time_index, fxd_price, "--", alpha=0.7, label="Fixed Price")
ax2.set_ylabel("Price")

fig.legend(loc="upper right")
plt.title("MPC: Heating vs price")
plt.tight_layout()
plt.show()

# Find the smallest value of lambda that recovers the same solution as the hard constraints
