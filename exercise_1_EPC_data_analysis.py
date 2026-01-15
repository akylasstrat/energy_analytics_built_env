# -*- coding: utf-8 -*-
"""
Explore EPC data from Islington borough

Tasks:
    - Exploratory analysis
    - Identify most important factor of energy consumption
@author: ucbva19@ucl.ac.uk
"""

# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression


plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600

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

#%%
# -------------------------
# Load
# -------------------------

# path = 'C:\\Users\\ucbva19\\Git projects\\energy_analytics_built_env\\data raw'
# df = pd.read_csv(f"{path}\\Energy Performance Cetrificate data.csv")  # change path

path = 'C:\\Users\\ucbva19\\Git projects\\energy_analytics_built_env\\data raw\\epc-certificates-Islington'
df = pd.read_csv(f"{path}\\certificates.csv")  # change path

# drop columns with NaNs
df = df.dropna(axis=1)

print(df.head())
print(df.isna().sum())

assert(df.isna().sum().sum() == 0)

# for col in df.columns:    
#     if df.isna().sum().loc[col] > 500:
#         print(col)
#         del df[col]

#%%

# Outline of mini-tutorial or exercise 

# 1. Fit a regression model to predict annual heating cost using building characteristics.

# 2. Select one retrofit intervention (e.g. wall insulation upgrade).

# 3. For each dwelling, simulate the retrofit by modifying the relevant variable.

# 4. Estimate the average and distribution of cost savings.

# 5. Identify which types of dwellings benefit most.

# 6. Do not change floor area, occupancy, or fuel prices.

# variables_retained = ['ENERGY_CONSUMPTION_CURRENT', 'ENERGY_CONSUMPTION_POTENTIAL', 
#                       'HEATING_COST_CURRENT', 'HEATING_COST_POTENTIAL',
#                       'POTENTIAL_ENERGY_EFFICIENCY', 
#                       'CURRENT_ENERGY_EFFICIENCY', 
#                       'TOTAL_FLOOR_AREA', 'PROPERTY_TYPE', 'BUILT_FORM', 
#                       'WALLS_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 
#                       'LIGHTING_ENERGY_EFF', 'LIGHTING_COST_CURRENT', 'LIGHTING_COST_POTENTIAL']

# data = df[variables_retained].copy()

# Create variables to indicate savings
df['potential_energy_savings'] = df['ENERGY_CONSUMPTION_CURRENT'] - df['ENERGY_CONSUMPTION_POTENTIAL']
df['potential_heat_cost_savings'] = df['HEATING_COST_CURRENT'] - df['HEATING_COST_POTENTIAL']
df['potential_light_cost_savings'] = df['LIGHTING_COST_CURRENT'] - df['LIGHTING_COST_POTENTIAL']


#%% Step 1: Pre-process and fit regression model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import ExtraTreesRegressor

# Select target variable from ['ENERGY_CONSUMPTION_CURRENT', 'HEATING_COST_CURRENT', 'LIGHTING_COST_CURRENT']

target_current = 'ENERGY_CONSUMPTION_CURRENT'
target_potential = 'ENERGY_CONSUMPTION_POTENTIAL'

Y = df[[target_current, target_potential]]

numerical_features = ['CURRENT_ENERGY_EFFICIENCY', 'TOTAL_FLOOR_AREA']
ordinal_features = ['WALLS_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 
                    'LIGHTING_ENERGY_EFF', 'HOT_WATER_ENERGY_EFF']
categorical_features = ['PROPERTY_TYPE', 'BUILT_FORM']

X = df[numerical_features + categorical_features + ordinal_features]

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=0)

train_current_Y = train_Y[target_current]
test_current_Y = test_Y[target_current]

train_potential_Y = train_Y[target_potential]
test_potential_Y = test_Y[target_potential]
#%%
# ------------------
# Preprocessing
# ------------------

categories_list = [['Bungalow', 'Flat', 'House', 'Maisonette'], 
              ['Not Recorded', 'Detached', 'Enclosed End-Terrace', 'Enclosed Mid-Terrace','End-Terrace', 'Mid-Terrace', 'Semi-Detached']]

ord_list =   [['Very Poor', 'Poor', 'Average', 'Good', 'Very Good'], 
              ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good'],
              ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good'], 
              ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good']]

one_hot_preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                  ("ord", OrdinalEncoder(categories = ord_list), ordinal_features),
                  ("num", "passthrough", numerical_features),])

# ordinal_preprocessor = ColumnTransformer(
#     transformers=[("cat", OrdinalEncoder(categories = categories_list), categorical_features),
#                   ("num", "passthrough", numerical_features),])

# Select preprocesser 
preprocessor = one_hot_preprocessor

# ------------------
# LR Model pipeline
# ------------------
lr_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", LinearRegression()),
    ])

lr_model.fit(train_X, train_potential_Y)

y_lr_pred = lr_model.predict(test_X)

# ------------------
# ExtraTree Model pipeline
# ------------------
et_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", ExtraTreesRegressor()),
    ])

et_model.fit(train_X, train_potential_Y)
y_et_pred = et_model.predict(test_X)


#%%
_, _ = accuracy_metrics(test_current_Y.values, y_et_pred)
_, _ = accuracy_metrics(test_current_Y.values, y_lr_pred)

plt.plot(test_current_Y.values[:1000])
plt.plot(y_et_pred[:1000])
plt.show()
asfd
#%% Check feature importance 
# -----------------------------
# Feature importance
# -----------------------------
# get feature names after one-hot encoding
feature_names = et_model.named_steps["preprocess"].get_feature_names_out()
importances = et_model.named_steps["regressor"].feature_importances_

fi = pd.Series(importances, index = feature_names).sort_values(ascending=False)

# -----------------------------
# Plot (top 15)
# -----------------------------
plt.figure(figsize=(6, 4))
fi.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Feature importance (ExtraTrees)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

#%%
from sklearn.inspection import permutation_importance

r = permutation_importance(et_model, test_X, test_current_Y,
                           n_repeats=30, random_state=0)

#%%
sorted_importances_idx = r.importances_mean.argsort()
importances = pd.DataFrame( r.importances[sorted_importances_idx].T, columns = X.columns[sorted_importances_idx],)

ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()

#%%
fi = pd.Series(importances, index = feature_names).sort_values(ascending=False)

# -----------------------------
# Plot (top 15)
# -----------------------------
plt.figure(figsize=(6, 4))
fi.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Feature importance (ExtraTrees)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


#%%
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{diabetes.feature_names[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")
        
#%% Alternatively, use LabelEncoding instead of One-hot encoding


#%% Step 2. pick intervention for a specific type of house

# List of interventions
# 'LIGHTING_ENERGY_EFF' : From Poor/Average -> Very Good
# 'MAINHEAT_ENERGY_EFF': From Poor/Average -> Very Good


#### Example: let's see what happens when we improve heating efficiency from poor to very good 

# Find test observations with MAINHEAT_ENERGY_EFF == Poor

mask = test_X['MAINHEAT_ENERGY_EFF'].values == 'Poor'

pre_retrofit_X = test_X.iloc[mask]

retrofit_X = pre_retrofit_X.copy()
retrofit_X['MAINHEAT_ENERGY_EFF'] = 'Very Good'

# Predict counterfactual
counterfactual_Y = et_model.predict(retrofit_X)

# Estimate energy savings
delta_energy_predicted = test_current_Y.iloc[mask] - counterfactual_Y
delta_energy_dataset = test_current_Y.iloc[mask] - test_potential_Y.iloc[mask]

    
plt.scatter(delta_energy_predicted, delta_energy_dataset)
plt.show()

#%%
# -------------------------
# Plot distribution (rough ggplot freqpoly)
# -------------------------
plt.figure()
plt.hist(df["ENERGY_CONSUMPTION_CURRENT"], bins=200, density=True)
plt.axvline(df["ENERGY_CONSUMPTION_CURRENT"].mean(), linestyle="--", color = 'black')
plt.xlim(0, 3000)
plt.title("ENERGY_CONSUMPTION distribution")
plt.tight_layout()
plt.show()

# -------------------------
# EPC rating counts
# -------------------------
df["CURRENT_ENERGY_RATING"] = df["CURRENT_ENERGY_RATING"].astype("category")
print(df["CURRENT_ENERGY_RATING"].value_counts())

plt.figure()
for k, g in df.groupby("CURRENT_ENERGY_RATING"):
    plt.hist(g["ENERGY_CONSUMPTION_CURRENT"], bins=150, histtype="step", label=str(k))
plt.xlim(0, df["ENERGY_CONSUMPTION_CURRENT"].quantile(0.99))
plt.title("Consumption grouped by EPC rating")
plt.legend(ncol=2, fontsize=8)
plt.show()

#%%
# Boxplot by EPC
plt.figure()
cats = sorted(df["CURRENT_ENERGY_RATING"].dropna().unique())
data = [df.loc[df["CURRENT_ENERGY_RATING"] == c, "ENERGY_CONSUMPTION_CURRENT"].values for c in cats]
plt.boxplot(data, labels=cats, showfliers=False)
plt.title("Consumption by EPC rating (boxplot)")
plt.xlabel("EPC")
plt.ylabel("ENERGY_CONSUMPTION $(kWh/m^2)$")
plt.tight_layout()
plt.show()
#%%
# -------------------------
# Consumption by property type (basic descriptive stats)
# -------------------------
if "PROPERTY_TYPE" in df.columns:
    grp = df.groupby("PROPERTY_TYPE")["ENERGY_CONSUMPTION_CURRENT"]
    desc = grp.agg(["count", "mean", "std", "median"])
    print(desc.sort_values("mean"))
#%%
# -------------------------
# Floor area distribution + filter
# -------------------------
print(df["TOTAL_FLOOR_AREA"].describe())

plt.figure()
x = df["TOTAL_FLOOR_AREA"].dropna()
plt.hist(x, bins=150, density=True)
plt.axvline(x.mean(), linestyle="--", color = 'black')
plt.xlim(0, 500)
plt.title("TOTAL_FLOOR_AREA distribution")
plt.tight_layout()
plt.show()

#%%

# Area vs consumption scatter + regression line (quick)
plt.figure()
plt.scatter(df["TOTAL_FLOOR_AREA"], df["ENERGY_CONSUMPTION_CURRENT"], s=5, alpha=0.2)
# m, b = np.polyfit(df["TOTAL_FLOOR_AREA"], df["ENERGY_CONSUMPTION_CURRENT"], 1)
# xs = np.linspace(df["TOTAL_FLOOR_AREA"].min(), df["TOTAL_FLOOR_AREA"].max(), 200)
# plt.plot(xs, m * xs + b)
plt.title("Energy consumption vs total floor area")
plt.xlabel("TOTAL_FLOOR_AREA")
plt.ylabel("ENERGY_CONSUMPTION (kWh)")
plt.tight_layout()
plt.show()
#%%
# -------------------------
# Recode WALLS_DESCRIPTION into a few buckets (regex like grepl)
# -------------------------
if "WALLS_DESCRIPTION" in df.columns:
    w = df["WALLS_DESCRIPTION"].astype(str)

    def recode_walls(s):
        if "Cavity wall, as built, insulated" in s or "Cavity wall, filled cavity" in s:
            return "Insulated Cavity Wall"
        if "Cavity wall, as built, no insulation" in s:
            return "Uninsulated Cavity Wall"
        if "Solid brick, with external insulation" in s:
            return "Externally Insul. Solid Wall"
        if "Solid brick, with internal insulation" in s:
            return "Internally Insul. Solid Wall"
        if "Solid brick, as built, no insulation" in s:
            return "Uninsulated Solid Wall"
        return np.nan

    df["WALLS_DESCRIPTION_RECODED"] = w.map(recode_walls)

    keep = [
        "Insulated Cavity Wall",
        "Uninsulated Cavity Wall",
        "Externally Insul. Solid Wall",
        "Internally Insul. Solid Wall",
        "Uninsulated Solid Wall",
    ]
    df = df[df["WALLS_DESCRIPTION_RECODED"].isin(keep)]
    print(df["WALLS_DESCRIPTION_RECODED"].value_counts())

    # Mean consumption by wall type (bar)
    m = df.groupby("WALLS_DESCRIPTION_RECODED")["ENERGY_CONSUMPTION_CURRENT"].mean().sort_values()
    plt.figure()
    plt.bar(m.index, m.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean ENERGY_CONSUMPTION (kWh)")
    plt.title("Mean consumption by wall type")
    plt.tight_layout()
    plt.show()
#%%

# Load libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance

target_variable = 'ENERGY_CONSUMPTION_CURRENT'
feature_list = ['CURRENT_ENERGY_EFFICIENCY', 'TOTAL_FLOOR_AREA',
              'WALLS_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 'LIGHTING_ENERGY_EFF', 'HOT_WATER_ENERGY_EFF', 'PROPERTY_TYPE', 'BUILT_FORM']

# further seperate features into numerical, categorical, and ordinal

numerical_features = ['CURRENT_ENERGY_EFFICIENCY', 'TOTAL_FLOOR_AREA']
ordinal_features = ['WALLS_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 
                    'LIGHTING_ENERGY_EFF', 'HOT_WATER_ENERGY_EFF']
categorical_features = ['PROPERTY_TYPE', 'BUILT_FORM']

Y = df[target_variable]
X = df[feature_list]

# Training/ test split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=42)


# ------------------
# Preprocessing
# ------------------

categories_list = [['Bungalow', 'Flat', 'House', 'Maisonette'], 
                  ['Not Recorded', 'Detached', 'Enclosed End-Terrace', 'Enclosed Mid-Terrace','End-Terrace', 'Mid-Terrace', 'Semi-Detached']]

ord_list =   [['Very Poor', 'Poor', 'Average', 'Good', 'Very Good'], 
              ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good'],
              ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good'], 
              ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good']]

# create preprocessor that implements one-hot and ordinal encoding
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                  ("ord", OrdinalEncoder(categories = ord_list), ordinal_features),
                  ("num", "passthrough", numerical_features),])


# ExtraTree Model pipeline
et_model = Pipeline(
    steps=[("preprocess", preprocessor), ("regressor", ExtraTreesRegressor()),])

et_model.fit(train_X, train_Y)
y_et_pred = et_model.predict(test_X)

# evaluate accuracy 
_, _ = accuracy_metrics(test_Y.values, y_et_pred)

#%%
plt.plot(test_current_Y.values[:1000])
plt.plot(y_et_pred[:1000])
plt.show()
#%%
# Feature importance (default estimation in tree-based algorithms)
# get feature names after one-hot encoding
feature_names = et_model.named_steps["preprocess"].get_feature_names_out()
importances = et_model.named_steps["regressor"].feature_importances_

fi = pd.Series(importances, index = feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 4))
fi.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Feature importance (ExtraTrees)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Permutation importance
r = permutation_importance(et_model, test_X, test_Y, n_repeats=30, random_state=0)
sorted_importances_idx = r.importances_mean.argsort()
importances = pd.DataFrame( r.importances[sorted_importances_idx].T, columns = X.columns[sorted_importances_idx],)

ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show()