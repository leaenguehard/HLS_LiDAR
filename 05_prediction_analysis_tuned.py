# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 17:04:14 2025
Script to create regression models with best parameters and reload option
@author: leengu001
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === CONFIG ===
LOAD_SAVED_MODELS = False  # Set to True to load saved models instead of retraining
SAVE_DIR = "results/Regression_models/Tuned_20folds_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load Data ===
data = pd.read_csv("results//pixel_table/pixel_table_final_filt.csv")
ps_features = ['blue_PS', 'green_PS', 'red_PS', 'NIR_PS', 'SWIR1_PS', 'SWIR2_PS', 'NDVI_PS']
ls_features = ['blue_LS', 'green_LS', 'red_LS', 'NIR_LS', 'SWIR1_LS', 'SWIR2_LS', 'NDVI_LS']
targets = ['med_CHM', 'crown_cov']

cv = KFold(n_splits=20, shuffle=True, random_state=42)

best_params_dict = {
    "med_CHM_PS":  {'max_depth': 40,  'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200},
    "crown_cov_PS": {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200},
    "med_CHM_LS":  {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200},
    "crown_cov_LS": {'max_depth': 40,  'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
}

# === Load or Train Models ===
if LOAD_SAVED_MODELS:
    print("🧠 Loading pre-trained models and predictions...")
    saved_models = {}
    results = {}
    for model_name in best_params_dict:
        model_path = os.path.join(SAVE_DIR, f"{model_name}.joblib")
        saved_models[model_name] = joblib.load(model_path)

    with open(os.path.join(SAVE_DIR, "model_predictions.pkl"), "rb") as f:
        results = pickle.load(f)

    print("✅ Models and predictions loaded from disk.")

else:
    print("🔁 Training models with best parameters...")
    results = {}
    with tqdm(total=4, desc="Modeling Progress") as pbar:
        for label, feats in zip(["PS", "LS"], [ps_features, ls_features]):
            X = data[feats]
            for target in targets:
                y = data[target]
                model_key = f"{target}_{label}"
                params = best_params_dict[model_key]
                model = RandomForestRegressor(**params, random_state=42)
                preds = cross_val_predict(model, X, y, cv=cv)
                model.fit(X, y)
                results[model_key] = {"true": y.values, "pred": preds}

                joblib.dump(model, os.path.join(SAVE_DIR, f"{model_key}.joblib"))
                pbar.update(1)

    with open(os.path.join(SAVE_DIR, "model_predictions.pkl"), "wb") as f:
        pickle.dump(results, f)

    print("✅ Models and predictions saved.")

# === Metrics Helper ===
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

# === Custom Titles ===
titles_map = {
    "med_CHM_PS": "Canopy Height - Peak Summer",
    "med_CHM_LS": "Canopy Height - Late Summer",
    "crown_cov_PS": "Crown Cover - Peak Summer",
    "crown_cov_LS": "Crown Cover - Late Summer"
}

# === Scatter Plot ===
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for idx, (key, res) in enumerate(results.items()):
    true = res["true"]
    pred = res["pred"]
    r2, rmse = get_metrics(true, pred)
    color_vals = data["crown_cov_bin"].values
    sns.scatterplot(x=true, y=pred, alpha=0.3, ax=axes[idx])
    sns.scatterplot(x=true, y=pred, hue=color_vals, palette="viridis", alpha=0.6, ax=axes[idx], legend=False)
    axes[idx].plot([true.min(), true.max()], [true.min(), true.max()], 'r--')
    axes[idx].set_title(f"{titles_map[key]}\nR²={r2:.2f}, RMSE={rmse:.2f} (10 folds)")
    axes[idx].set_xlabel("True")
    axes[idx].set_ylabel("Predicted")

plt.tight_layout()
plt.show()

# === Plot with Hue Gradient  ===
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 0.05, 1, 0.05], wspace=0.4, hspace=0.3)

crown_cov_vals = data["crown_cov"].values
chm_vals = data["med_CHM"].values

# Normalize and colormaps
norm_cc = mcolors.Normalize(vmin=crown_cov_vals.min(), vmax=crown_cov_vals.max())
norm_chm = mcolors.Normalize(vmin=chm_vals.min(), vmax=chm_vals.max())
cmap_cc = cm.viridis
cmap_chm = cm.plasma

sm_cc = cm.ScalarMappable(norm=norm_cc, cmap=cmap_cc)
sm_chm = cm.ScalarMappable(norm=norm_chm, cmap=cmap_chm)

# Scatter plots with hue
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
    ax.set_title(f"{titles_map[key]}\nR²={r2:.2f}, RMSE={rmse:.2f}")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")

    if "crown_cov" in key:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

# Add colorbars
cbar_ax_mid = fig.add_subplot(gs[:, 1])
fig.colorbar(sm_cc, cax=cbar_ax_mid).set_label("Crown Cover (%)")

cbar_ax_right = fig.add_subplot(gs[:, 3])
fig.colorbar(sm_chm, cax=cbar_ax_right).set_label("Canopy Height (m)")
plt.show()


# === Feature Importances ===
importances = {}
print("📊 Calculating feature importances...")

for label, feats in zip(["PS", "LS"], [ps_features, ls_features]):
    X = data[feats]
    for target in targets:
        y = data[target]
        model_key = f"{target}_{label}"
        params = best_params_dict[model_key]
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X, y)
        importances[model_key] = pd.Series(model.feature_importances_, index=feats)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
color = "#1f77b4"

for i, key in enumerate(importances):
    imp = importances[key].sort_values(ascending=True)
    sns.barplot(x=imp.values, y=imp.index, ax=axes[i], color=color)
    axes[i].set_title(f"Feature Importance: {titles_map[key]}")

plt.tight_layout()
plt.show()

importance_df = pd.DataFrame(importances).T
importance_df.index.name = "Model"
importance_df.to_csv("results/Regression_models/Tuned_20folds_final/feature_importance_20folds_tuned.csv")
print("✅ Feature importances saved ")

# Save performance metrics
metrics = []
for key, res in results.items():
    r2, rmse = get_metrics(res["true"], res["pred"])
    mae = mean_absolute_error(res["true"], res["pred"])
    metrics.append({
        "Model": key,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("results/Regression_models/Tuned_20folds_final/model_metrics_20folds.csv", index=False)
print("✅ Metrics saved ")

# === Save Predictions to CSV ===
print("📄 Saving predictions as CSV files...")

predictions_dir = os.path.join(SAVE_DIR, "predictions_csv")
os.makedirs(predictions_dir, exist_ok=True)

for model_key, result in results.items():
    df = pd.DataFrame({
        "True": result["true"],
        "Predicted": result["pred"]
    })
    out_path = os.path.join(predictions_dir, f"{model_key}_predictions.csv")
    df.to_csv(out_path, index=False)

print(f"✅ Predictions saved to CSVs in: {predictions_dir}")


