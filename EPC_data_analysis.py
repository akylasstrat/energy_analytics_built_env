# -*- coding: utf-8 -*-
"""
EPC data analysis

@author: ucbva19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


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

path = 'C:\\Users\\ucbva19\\Git projects\\energy_analytics_built_env\\data raw'
df = pd.read_csv(f"{path}\\Energy Performance Cetrificate data.csv")  # change path

print(df.head())
print(df.shape)
print(df.isna().sum())

for col in df.columns:
    if df.isna().sum().loc[col] > 1000:
        del df[col]

#%%

# Outline of mini-tutorial or exercise 

# 1. Fit a regression model to predict annual heating cost using building characteristics.

# 2. Select one retrofit intervention (e.g. wall insulation upgrade).

# 3. For each dwelling, simulate the retrofit by modifying the relevant variable.

# 4. Estimate the average and distribution of cost savings.

# 5. Identify which types of dwellings benefit most.

# 6. Do not change floor area, occupancy, or fuel prices.

df = df.dropna()
variables_retained = ['ENERGY_CONSUMPTION_CURRENT', 'ENERGY_CONSUMPTION_POTENTIAL', 
                      'HEATING_COST_CURRENT', 'HEATING_COST_POTENTIAL',
                      'POTENTIAL_ENERGY_EFFICIENCY', 
                      'CURRENT_ENERGY_EFFICIENCY', 
                      'TOTAL_FLOOR_AREA', 'PROPERTY_TYPE', 'BUILT_FORM', 
                      'WALLS_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 
                      'LIGHTING_ENERGY_EFF', 'LIGHTING_COST_CURRENT', 'LIGHTING_COST_POTENTIAL']


data = df[variables_retained]

# Create variables to indicate savings
data['potential_energy_savings'] = data['ENERGY_CONSUMPTION_CURRENT'] - data['ENERGY_CONSUMPTION_POTENTIAL']
data['potential_heat_cost_savings'] = data['HEATING_COST_CURRENT'] - data['HEATING_COST_POTENTIAL']
data['potential_light_cost_savings'] = data['LIGHTING_COST_CURRENT'] - data['LIGHTING_COST_POTENTIAL']


#%% Step 1: Pre-process and fit regression model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor

# Select target variable from ['ENERGY_CONSUMPTION_CURRENT', 'HEATING_COST_CURRENT', 'LIGHTING_COST_CURRENT']

target_current = 'ENERGY_CONSUMPTION_CURRENT'
target_potential = 'ENERGY_CONSUMPTION_POTENTIAL'

Y = data[[target_current, target_potential]]

numerical_features = ['CURRENT_ENERGY_EFFICIENCY', 'TOTAL_FLOOR_AREA']
categorical_features = ['PROPERTY_TYPE', 'BUILT_FORM', 'WALLS_ENERGY_EFF', 
                       'MAINHEAT_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']

X = data[numerical_features+categorical_features]

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=42)

train_current_Y = train_Y[target_current]
test_current_Y = test_Y[target_current]

train_potential_Y = train_Y[target_potential]
test_potential_Y = test_Y[target_potential]
#%%
# ------------------
# Preprocessing
# ------------------
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                  ("num", "passthrough", numerical_features),])


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

#ENERGY_CONSUMPTION_CURRENT is in KWH/m2 and it is the sum of electricity and gas consumption for each home
#create a new variable with consumption/m2 * m2, to get consumption.
# -------------------------
# Create total consumption (kWh)
# -------------------------
df["ENERGY_CONSUMPTION"] = df["ENERGY_CONSUMPTION_CURRENT"] * df["TOTAL_FLOOR_AREA"]
print(df["ENERGY_CONSUMPTION"].describe())

# -------------------------
# Filter outliers (Adan & Fuerst-like bounds)
# -------------------------
df = df[(df["ENERGY_CONSUMPTION"] > 3100) & (df["ENERGY_CONSUMPTION"] < 75000)]
print(df.shape)
print(df["ENERGY_CONSUMPTION"].describe())

#%%
# -------------------------
# Plot distribution (rough ggplot freqpoly)
# -------------------------
plt.figure()
x = df["ENERGY_CONSUMPTION"].dropna()
plt.hist(x, bins=200, density=True)
plt.axvline(x.mean(), linestyle="--", color = 'black')
plt.xlim(0, 200000)
plt.title("ENERGY_CONSUMPTION distribution")
plt.tight_layout()
plt.show()
#%%

# -------------------------
# EPC rating counts
# -------------------------
df["CURRENT_ENERGY_RATING"] = df["CURRENT_ENERGY_RATING"].astype("category")
print(df["CURRENT_ENERGY_RATING"].value_counts())

#%%
plt.figure()
for k, g in df.groupby("CURRENT_ENERGY_RATING"):
    plt.hist(g["ENERGY_CONSUMPTION"], bins=150, histtype="step", label=str(k))
plt.xlim(0, df["ENERGY_CONSUMPTION"].quantile(0.99))
plt.title("Consumption grouped by EPC rating")
plt.legend(ncol=2, fontsize=8)
plt.show()

#%%
# Boxplot by EPC
plt.figure()
cats = sorted(df["CURRENT_ENERGY_RATING"].dropna().unique())
data = [df.loc[df["CURRENT_ENERGY_RATING"] == c, "ENERGY_CONSUMPTION"].values for c in cats]
plt.boxplot(data, labels=cats, showfliers=False)
plt.title("Consumption by EPC rating (boxplot)")
plt.xlabel("EPC")
plt.ylabel("ENERGY_CONSUMPTION (kWh)")
plt.tight_layout()
plt.show()
#%%
# -------------------------
# Consumption by property type (basic descriptive stats)
# -------------------------
if "PROPERTY_TYPE" in df.columns:
    grp = df.groupby("PROPERTY_TYPE")["ENERGY_CONSUMPTION"]
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
df = df[(df["TOTAL_FLOOR_AREA"] <= 300) & (df["TOTAL_FLOOR_AREA"] > 20)]
print(df.shape)

# Area vs consumption scatter + regression line (quick)
plt.figure()
plt.scatter(df["TOTAL_FLOOR_AREA"], df["ENERGY_CONSUMPTION"], s=5, alpha=0.2)
m, b = np.polyfit(df["TOTAL_FLOOR_AREA"], df["ENERGY_CONSUMPTION"], 1)
xs = np.linspace(df["TOTAL_FLOOR_AREA"].min(), df["TOTAL_FLOOR_AREA"].max(), 200)
plt.plot(xs, m * xs + b)
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
    m = df.groupby("WALLS_DESCRIPTION_RECODED")["ENERGY_CONSUMPTION"].mean().sort_values()
    plt.figure()
    plt.bar(m.index, m.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean ENERGY_CONSUMPTION (kWh)")
    plt.title("Mean consumption by wall type")
    plt.tight_layout()
    plt.show()
#%%
# -------------------------
# Filter heating system descriptions (regex OR list)
# -------------------------
if "MAINHEAT_DESCRIPTION" in df.columns:
    pat = (
        "Boiler and radiators, mains gas|"
        "Boiler and radiators, oil|"
        "Room heaters, electricity|"
        "Air source heat pump, radiators, electric|"
        "Electric underfloor heating|"
        "Portable electric heaters assumed for most rooms|"
        "Portable electric heating assumed for most rooms"
    )
    df = df[df["MAINHEAT_DESCRIPTION"].astype(str).str.contains(pat, regex=True, na=False)]
    print(df["MAINHEAT_DESCRIPTION"].value_counts())

    m = df.groupby("MAINHEAT_DESCRIPTION")["ENERGY_CONSUMPTION"].mean().sort_values()
    plt.figure()
    plt.barh(m.index, m.values)
    plt.xlabel("Mean ENERGY_CONSUMPTION (kWh)")
    plt.title("Mean consumption by heating type")
    plt.tight_layout()
    plt.show()

# -------------------------
# Built form (drop NO DATA!)
# -------------------------
if "BUILT_FORM" in df.columns:
    df["BUILT_FORM"] = df["BUILT_FORM"].astype("category")
    df = df[df["BUILT_FORM"] != "NO DATA!"]
    print(df["BUILT_FORM"].value_counts())

    m = df.groupby("BUILT_FORM")["ENERGY_CONSUMPTION"].mean().sort_values()
    plt.figure()
    plt.barh(m.index, m.values)
    plt.xlabel("Mean ENERGY_CONSUMPTION (kWh)")
    plt.title("Mean consumption by built form")
    plt.tight_layout()
    plt.show()

# -------------------------
# Regression models (lm(...) equivalents)
# NOTE: in R they accidentally compare raw consumption without controlling for floor area etc.
# We'll replicate their setup first, then you can fix it by modelling ENERGY_CONSUMPTION_CURRENT or adding floor area.
# -------------------------
keep_cols = ["ENERGY_CONSUMPTION", "BUILT_FORM", "WALLS_ENERGY_EFF", "ROOF_ENERGY_EFF",
             "MAINHEAT_ENERGY_EFF", "LIGHTING_ENERGY_EFF"]
df_reg = df[[c for c in keep_cols if c in df.columns]].dropna()

# Remove "N/A"
for c in ["WALLS_ENERGY_EFF", "ROOF_ENERGY_EFF", "MAINHEAT_ENERGY_EFF", "LIGHTING_ENERGY_EFF"]:
    if c in df_reg.columns:
        df_reg = df_reg[df_reg[c] != "N/A"]

y = df_reg["ENERGY_CONSUMPTION"]
# sequential models
models = [
    ["WALLS_ENERGY_EFF"],
    ["WALLS_ENERGY_EFF", "ROOF_ENERGY_EFF"],
    ["WALLS_ENERGY_EFF", "ROOF_ENERGY_EFF", "LIGHTING_ENERGY_EFF"],
    ["WALLS_ENERGY_EFF", "ROOF_ENERGY_EFF", "LIGHTING_ENERGY_EFF", "BUILT_FORM"],
]

# same train/test split for comparability
train_idx, test_idx = train_test_split(df_reg.index, test_size=0.3, random_state=101)
def fit_and_eval(feats):
    X = df_reg[feats]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feats)], remainder="drop")
    pipe = Pipeline([("pre", pre), ("lr", LinearRegression())])
    pipe.fit(X.loc[train_idx], y.loc[train_idx])

    pred_tr = pipe.predict(X.loc[train_idx])
    pred_te = pipe.predict(X.loc[test_idx])

    r2_tr = r2_score(y.loc[train_idx], pred_tr)
    r2_te = r2_score(y.loc[test_idx], pred_te)
    rss_tr = np.sum((y.loc[train_idx] - pred_tr) ** 2)
    rss_te = np.sum((y.loc[test_idx] - pred_te) ** 2)
    rmse_te = mean_squared_error(y.loc[test_idx], pred_te, squared=False)

    return {"features": feats, "R2_train": r2_tr, "R2_test": r2_te, "RSS_train": rss_tr, "RSS_test": rss_te, "RMSE_test": rmse_te}

results = [fit_and_eval(m) for m in models]
print(pd.DataFrame(results))

# quick "anova-like" check: RSS should drop as you add predictors (on train set)
# (R's anova(lm1,lm2) is a formal nested-model F-test; this is a simple diagnostic.)
