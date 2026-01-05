"""
Tutorial 3 - Fuel Poverty (Socio-econ variables) - Python version

Implements the same pipeline as the provided R script:
- Load FuelPoverty_2019.csv
- Rename columns
- Recode categorical variables (Region, Tenure, DWtype, Working status, Ethnic origin, EPC, hhcompx, dfvuln)
- Remove non-residential ("converted and non residential")
- Train/test split (70/30, stratified)
- Logistic regression (one-hot encoding for categoricals)
- Evaluate: confusion matrix, accuracy, precision, recall, F1, ROC curve + AUC
- Random oversampling of minority class to N=13000 total training samples
- Refit and re-evaluate

@author: a.stratigakos@ucl.ac.uk
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

# Plotting figures default
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
#%%
# -----------------------------
# Helpers
# -----------------------------
def mcfadden_r2(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    """
    McFadden pseudo-R^2 = 1 - (LL_model / LL_null)

    LL_model: log-likelihood using predicted probabilities p_hat.
    LL_null: log-likelihood using a constant-only model with p = mean(y).
    """
    eps = 1e-15
    y = y_true.astype(float)
    p = np.clip(p_hat.astype(float), eps, 1 - eps)

    ll_model = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    p0 = np.clip(np.mean(y), eps, 1 - eps)
    ll_null = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))

    return 1.0 - (ll_model / ll_null)

def plot_bar(df: pd.DataFrame, col: str, title: str | None = None) -> None:
    """Simple count bar plot (similar to ggplot geom_bar)."""
    counts = df[col].value_counts(dropna=False)
    plt.figure()
    counts.plot(kind="bar")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.title(title or col)
    plt.tight_layout()
    plt.show()

def oversample_minority_to_N(
    X: pd.DataFrame, y: pd.Series, N: int, random_state: int = 101
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Random oversampling (with replacement) to achieve total size N,
    increasing only the minority class (like ROSE::ovun.sample(method="over", N=...)).

    If current size already >= N, returns original.
    """
    rng = np.random.default_rng(random_state)

    if len(y) >= N:
        return X.copy(), y.copy()

    # assume binary classes {0,1} (fuel poverty flag)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Expected binary target with exactly 2 classes.")

    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]

    idx_min = np.where(y.values == minority_class)[0]
    idx_maj = np.where(y.values == majority_class)[0]

    current_n = len(y)
    needed = N - current_n
    if needed <= 0:
        return X.copy(), y.copy()

    # sample minority indices with replacement
    sampled_min = rng.choice(idx_min, size=needed, replace=True)

    new_idx = np.concatenate([np.arange(current_n), sampled_min])
    rng.shuffle(new_idx)

    X_over = X.iloc[new_idx].reset_index(drop=True)
    y_over = y.iloc[new_idx].reset_index(drop=True)

    return X_over, y_over

def evaluate_binary_classifier(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
    title_prefix: str = "",
    plot_roc: bool = True,
) -> dict:
    """Compute metrics + optionally plot ROC curve."""
    y_pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    auc = roc_auc_score(y_true, proba)

    results = {
        "confusion_matrix_[0,1]": cm,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "mcfadden_r2": mcfadden_r2(y_true, proba),
    }

    print("\n" + ("=" * 60))
    if title_prefix:
        print(title_prefix)
    print("Confusion matrix (rows=true [0,1], cols=pred [0,1]):\n", cm)
    print(
        f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}"
    )
    print(f"McFadden pseudo-R²={results['mcfadden_r2']:.3f}")

    if plot_roc:
        dfr, tpr, _ = roc_curve(y_true, proba)
        plt.figure()
        plt.plot(dfr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve {(' - ' + title_prefix) if title_prefix else ''} (AUC={auc:.2f})")
        plt.tight_layout()
        plt.show()

    return results

#%%

# -----------------------------
# Main script
# -----------------------------
data_path = 'C:\\Users\\ucbva19\\OneDrive - University College London\\ESDA\\BENV0092 Energy Data Analytics in the Built Environment\\data sets'
csv_path=f"{data_path}\\FuelPoverty_2019.csv"
random_state = 42

df = pd.read_csv(csv_path)

# Rename columns
df = df.rename(columns={"fpLILEEflg": "In_fuel_Poverty", "fpfullinc": "hh_income", "fuelexpn": "fuel_costs",
        "Unoc": "Under_occupied", "gorehs": "Region",
        "tenure4x": "Tenure", "emphrp3x": "Head_Working_Status",
        "ethhrp2x": "Head_Ethnic_Origin","Ageyng": "Age_of_youngest","Ageold": "Age_of_oldest",})

# Ensure target is usable as binary (0/1)
df["In_fuel_Poverty"] = df["In_fuel_Poverty"].values.astype(int)

#%%
# --- Recode categorical variables (mirrors recode_factor calls)
region_map = {
    1: "North East",
    2: "North West",
    4: "Yorkshire and the Humber",
    5: "East Midlands",
    6: "West Midlands",
    7: "East",
    8: "London",
    9: "South East",
    10: "South West",
}
tenure_map = {
    1: "owner occupied",
    2: "private rented",
    3: "local authority",
    4: "housing association",
}
dwtype_map = {
    1: "end terrace",
    2: "mid terrace",
    3: "semi detached",
    4: "detached",
    5: "purpose built",
    6: "converted and non residential",
}
work_map = {1: "working", 2: "unemployed", 3: "inactive"}
eth_map = {1: "white", 2: "ethnic minority"}
epc_map = {1: "A/B/C", 2: "D", 3: "E", 4: "F", 5: "G"}
hhcomp_map = {
    1: "couple, no child(ren) under 60",
    2: "couple, no child(ren) 60 or over",
    3: "couple with child(ren)",
    4: "lone parent with child(ren)",
    5: "other multi-person households",
    6: "one person under 60",
    7: "one person aged 60 or over",
}
dfvuln_map = {1: "Vulnerable", 0: "Not vulnerable"}

# Convert to numeric first (R had strings like "1"); then map.
for col, mp in [
    ("Region", region_map),
    ("Tenure", tenure_map),
    ("DWtype", dwtype_map),
    ("Head_Working_Status", work_map),
    ("Head_Ethnic_Origin", eth_map),
    ("EPC", epc_map),
    ("hhcompx", hhcomp_map),
    ("dfvuln", dfvuln_map),]:
    
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].map(mp).astype("category")

    # --- Basic exploration plots (similar to p1..p6 in R)
    for col in ["Region", "Tenure", "DWtype", "Head_Working_Status", "Head_Ethnic_Origin", "EPC"]:
        if col in df.columns:
            plot_bar(df, col, title=f"Distribution of {col}")

# --- Remove non-residential category (same as subset(!DWtype == "..."))
if "DWtype" in df.columns:
    df = df.loc[df["DWtype"] != "converted and non residential"].copy()

#%%
feat_cols = [ "hh_income", "fuel_costs", "Under_occupied", "Region", "Tenure", 
             "Head_Working_Status", "Head_Ethnic_Origin", "Age_of_youngest", "Age_of_oldest"]


Y = df["In_fuel_Poverty"]
X = df[feat_cols]

# --- Train/test split: 70/30 stratified (like caTools::sample.split)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.30, random_state=random_state, stratify=Y,)

print("\nTrain class balance:\n", train_Y.value_counts())
print("Test class balance:\n", test_Y.value_counts())

# --- Preprocess: one-hot encode categoricals
categorical_cols = list(X.columns)
preprocessor = ColumnTransformer(
    transformers=[ ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),],remainder="drop",)

#%% Model 1: logistic regression (binomial GLM analogue)

lr_model = LogisticRegression( max_iter=2000,solver="lbfgs",)
clf1 = Pipeline(steps=[("preprocess", preprocessor), ("model", lr_model)])
clf1.fit(train_X, train_Y)


# Predict probabilities for class 1
proba_test_1 = clf1.predict_proba(test_X)[:, 1]
evaluate_binary_classifier( y_true = test_Y.values,
    proba=proba_test_1, threshold=0.5, title_prefix="Logistic regression", plot_roc=True,)

#%% Model 2: ExtraTrees
from sklearn.ensemble import ExtraTreesClassifier

et_model = ExtraTreesClassifier(n_estimators = 500, min_samples_leaf = 2, bootstrap = True, n_jobs = -1, 
                                class_weight = {0: 1, 1: 5})
clf2 = Pipeline(steps=[("preprocess", preprocessor), ("model", et_model)])
clf2.fit(train_X, train_Y)

# Predict probabilities for class 1
proba_test_2 = clf2.predict_proba(test_X)[:, 1]
evaluate_binary_classifier( y_true = test_Y.values,
    proba = proba_test_2, threshold=0.5, title_prefix="ExtraTrees", plot_roc=True,)

#%% Change sample weight to penalize false-negatives

weighted_et_model = ExtraTreesClassifier(n_estimators = 500, min_samples_leaf = 2, bootstrap = True, n_jobs = -1, 
                                class_weight = {0: 1, 1: 5})
clf2 = Pipeline(steps=[("preprocess", preprocessor), ("model", weighted_et_model)])
clf2.fit(train_X, train_Y)

# Predict probabilities for class 1
proba_test_2 = clf2.predict_proba(test_X)[:, 1]
evaluate_binary_classifier( y_true = test_Y.values,
    proba = proba_test_2, threshold=0.5, title_prefix="ExtraTrees", plot_roc=True,)

