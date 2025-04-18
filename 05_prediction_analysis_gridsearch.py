# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 17:04:14 2025
Script to create regression models with GridSearchCV + progress bar
@author: leengu001
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Load Data ---
data = pd.read_csv("results//pixel_table/pixel_table_final_filt.csv")

# --- Feature Sets & Targets ---
ps_features = ['blue_PS', 'green_PS', 'red_PS', 'NIR_PS', 'SWIR1_PS', 'SWIR2_PS', 'NDVI_PS']
ls_features = ['blue_LS', 'green_LS', 'red_LS', 'NIR_LS', 'SWIR1_LS', 'SWIR2_LS', 'NDVI_LS']
targets = ['med_CHM', 'crown_cov']

# --- Cross-validation ---
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# --- Expanded Hyperparameter Grid (48 combinations) ---
param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 6],
    'max_features': ['sqrt', 'log2']
}

# --- Model Evaluation with GridSearchCV ---
def evaluate_model(X, y, label):
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    preds = cross_val_predict(best_model, X, y, cv=cv)

    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)

    print(f"\n🟩 {label}")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R²:   {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")

    return best_model, preds, grid_search.best_params_, r2, rmse, mae

# --- Run models ---
results = {}
metrics_list = []

print("🔁 Running GridSearchCV for 4 models...")
with tqdm(total=4, desc="Modeling Progress") as pbar:
    for label, feats in zip(["PS", "LS"], [ps_features, ls_features]):
        X = data[feats]
        for target in targets:
            y = data[target]
            model_key = f"{target}_{label}"
            model, preds, best_params, r2, rmse, mae = evaluate_model(X, y, model_key)
            results[model_key] = {"true": y.values, "pred": preds}
            metrics_list.append({
                "Model": model_key,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "BestParams": best_params
            })
            pbar.update(1)

# --- Export metrics + best params ---
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("results//model_gridsearch_report_full2.csv", index=False)
print("✅ Report saved to results//model_gridsearch_report_full.csv")

# --- Metrics Helper ---
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

# --- Scatter Plots (with hue and no hue) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

titles = {
    "med_CHM_PS": "CHM - Peak Summer",
    "med_CHM_LS": "CHM - Late Summer",
    "crown_cov_PS": "Crown Cov - Peak Summer",
    "crown_cov_LS": "Crown Cov - Late Summer"
}

for idx, (key, res) in enumerate(results.items()):
    true = res["true"]
    pred = res["pred"]
    r2, rmse = get_metrics(true, pred)
    sns.scatterplot(x=true, y=pred, alpha=0.3, ax=axes[idx])
    color_vals = data["crown_cov_bin"].values
    sns.scatterplot(x=true, y=pred, hue=color_vals, palette="viridis", alpha=0.6, ax=axes[idx], legend=False)
    axes[idx].plot([true.min(), true.max()], [true.min(), true.max()], 'r--')
    axes[idx].set_title(f"{titles[key]}\nR²={r2:.2f}, RMSE={rmse:.2f} (10 folds)")
    axes[idx].set_xlabel("True")
    axes[idx].set_ylabel("Predicted")

plt.tight_layout()
plt.show()

# --- Fancy Plot with Hue ---
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 0.05, 1, 0.05], wspace=0.4, hspace=0.3)

crown_cov_vals = data["crown_cov"].values
chm_vals = data["med_CHM"].values

norm_cc = mcolors.Normalize(vmin=crown_cov_vals.min(), vmax=crown_cov_vals.max())
norm_chm = mcolors.Normalize(vmin=chm_vals.min(), vmax=chm_vals.max())
cmap_cc = cm.viridis
cmap_chm = cm.plasma

sm_cc = cm.ScalarMappable(norm=norm_cc, cmap=cmap_cc)
sm_chm = cm.ScalarMappable(norm=norm_chm, cmap=cmap_chm)

for i, (key, res) in enumerate(results.items()):
    row = i // 2
    col = 0 if "CHM" in key else 2
    ax = fig.add_subplot(gs[row, col])
    true = res["true"]
    pred = res["pred"]
    r2, rmse = get_metrics(true, pred)

    color_vals = crown_cov_vals if "CHM" in key else chm_vals
    norm = norm_cc if "CHM" in key else norm_chm
    cmap = cmap_cc if "CHM" in key else cmap_chm
    colors = cmap(norm(color_vals))

    ax.scatter(true, pred, c=colors, alpha=0.6)
    ax.plot([true.min(), true.max()], [true.min(), true.max()], 'r--')
    ax.set_title(f"{titles[key]}\nR²={r2:.2f}, RMSE={rmse:.2f}")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    if "crown_cov" in key:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

cbar_ax_mid = fig.add_subplot(gs[:, 1])
fig.colorbar(sm_cc, cax=cbar_ax_mid).set_label("Crown Cover (%)")

cbar_ax_right = fig.add_subplot(gs[:, 3])
fig.colorbar(sm_chm, cax=cbar_ax_right).set_label("CHM (m)")

plt.show()

# --- Feature Importance ---
importances = {}
for label, feats in zip(["PS", "LS"], [ps_features, ls_features]):
    X = data[feats]
    for target in targets:
        y = data[target]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances[f"{target}_{label}"] = pd.Series(model.feature_importances_, index=feats)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
for i, key in enumerate(importances):
    imp = importances[key].sort_values(ascending=True)
    sns.barplot(x=imp.values, y=imp.index, ax=axes[i])
    axes[i].set_title(f"Feature Importance: {key.replace('_', ' ')}")
plt.tight_layout()
plt.show()

importance_df = pd.DataFrame(importances).T
importance_df.index.name = "Model"
importance_df.to_csv("results//feature_importance_full_datapool.csv")
print("✅ Feature importances saved to results//feature_importance_full_datapool.csv")
