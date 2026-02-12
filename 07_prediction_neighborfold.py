# -*- coding: utf-8 -*-
"""
Author: leengu001
Date  : 2025-07-07
"""
import os, pickle, joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ───────────────────── Settings ─────────────────────────────────────────────
LOAD_SAVED_MODELS = True
SAVE_DIR = "results/Regression_models/rev/BlockCV_models100TCPC"
os.makedirs(SAVE_DIR, exist_ok=True)

block_size_m  = 100
n_folds_max   = 20
verbose_plots = True

# ───────────────────── Data ─────────────────────────────────────────────────
data = pd.read_csv("results/pixel_table/pixel_table_final_filt.csv")

# ───────────── Predictor sets ─────────────────────────
PS_feats = [
    'blue_PS', 'green_PS', 'red_PS', 'NIR_PS',
    'SWIR1_PS', 'SWIR2_PS', 'NDVI_PS','TCB_PS', 'TCW_PS', 'TCG_PS'
]

LS_feats = [
    'blue_LS', 'green_LS', 'red_LS', 'NIR_LS',
    'SWIR1_LS', 'SWIR2_LS', 'NDVI_LS','TCB_LS', 'TCW_LS', 'TCG_LS'
]

feature_map = {"PS": PS_feats, "LS": LS_feats}

targets       = ["med_CHM", "crown_cov"]
seasons       = ["PS", "LS"]
target_models = [f"{t}_{s}" for t in targets for s in seasons]

# ───────────── Spatial block CV ─────────────────────────────────────────────
METRES_PER_DEG = 111_000
cell_deg = block_size_m / METRES_PER_DEG

min_lon, min_lat = data.longitude.min(), data.latitude.min()
block_x = np.floor((data.longitude - min_lon) / cell_deg).astype(int)
block_y = np.floor((data.latitude  - min_lat) / cell_deg).astype(int)
block_id = block_x * 10_000 + block_y

n_blocks = np.unique(block_id).size
n_splits = min(n_folds_max, n_blocks)
cv = GroupKFold(n_splits=n_splits)

# ───────────── Hyperparameters ──────────────────────────────────────────────
best_params = {
    "med_CHM_PS":   {'max_features': 'sqrt','min_samples_leaf': 2, 'min_samples_split': 2,'n_estimators': 500, 'random_state': 42},
    "med_CHM_LS":  {'max_features': 'sqrt','min_samples_leaf': 2, 'min_samples_split': 2,'n_estimators': 500, 'random_state': 42},
    "crown_cov_PS":  {'max_features': 'sqrt','min_samples_leaf': 2, 'min_samples_split': 2,'n_estimators': 500, 'random_state': 42},
    "crown_cov_LS":  {'max_features': 'sqrt','min_samples_leaf': 2, 'min_samples_split': 2,'n_estimators': 500, 'random_state': 42},
}

# ───────────── Training & block CV ──────────────────────────────────────────
results, saved_models = {}, {}

print("Training models with spatial block cross-validation …")
with tqdm(total=len(target_models), desc="Training") as pbar:
    for key in target_models:
        target, season = key.rsplit("_", 1)

        X = data[feature_map[season]]
        y = data[target]

        model = RandomForestRegressor(**best_params[key])

        y_pred = cross_val_predict(
            model, X, y,
            cv=cv, groups=block_id, n_jobs=-1
        )

        model.fit(X, y)

        results[key] = {"true": y.values, "pred": y_pred}
        saved_models[key] = model

        joblib.dump(model, os.path.join(SAVE_DIR, f"{key}.joblib"))
        with open(os.path.join(SAVE_DIR, f"{key}_pred.pkl"), "wb") as f:
            pickle.dump(results[key], f)

        pbar.update(1)

print("Models and predictions saved.")

# ───────────── Metrics ──────────────────────────────────────────────────────
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    rmse_pct = (rmse / np.mean(y_true)) * 100

    return r2, rmse, rmse_pct, mae


print(f"\nBlock CV results (block ≈ {block_size_m} m, folds={n_splits})")

metrics = []
for key, res in results.items():
    r2, rmse, rmse_pct, mae = get_metrics(res["true"], res["pred"])

    metrics.append({
        "Model": key,
        "R2": r2,
        "RMSE": rmse,
        "RMSE_%": rmse_pct,
        "MAE": mae
    })

    print(
        f"{key:<15s}  "
        f"R²={r2:5.3f}  "
        f"RMSE={rmse:6.2f} ({rmse_pct:5.1f}%)  "
        f"MAE={mae:6.2f}")


metrics_df = pd.DataFrame(metrics)
metrics_path = os.path.join(SAVE_DIR, "model_metrics_blockcv.csv")
metrics_df.to_csv(metrics_path, index=False)

# ───────────── Scatter plots ────────────────────────────────────────────────
if verbose_plots:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    title_map = {
        "med_CHM_PS":   "Canopy Height – Peak Summer",
        "med_CHM_LS":   "Canopy Height – Late Summer",
        "crown_cov_PS": "Crown Cover – Peak Summer",
        "crown_cov_LS": "Crown Cover – Late Summer"
    }

    crown_vals = data["crown_cov"].values
    chm_vals   = data["med_CHM"].values

    norm_cc  = mcolors.Normalize(vmin=crown_vals.min(), vmax=crown_vals.max())
    norm_chm = mcolors.Normalize(vmin=chm_vals.min(),   vmax=chm_vals.max())

    cmap_cc  = cm.viridis
    cmap_chm = cm.plasma

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, key in enumerate(target_models):
        true = results[key]["true"]
        pred = results[key]["pred"]
        r2, rmse, rmse_pct, mae = get_metrics(true, pred)

        cvals = crown_vals if "CHM" in key else chm_vals
        norm  = norm_cc    if "CHM" in key else norm_chm
        cmap  = cmap_cc    if "CHM" in key else cmap_chm

        ax = axes[idx]
        ax.scatter(true, pred, c=cmap(norm(cvals)), alpha=0.6, s=14)
        ax.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', lw=1)

        ax.set_title(f"{title_map[key]}\nR²={r2:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
        if "med_CHM" in key:
            ax.set_xlabel("Actual Canopy Height (m)")
            ax.set_ylabel("Estimated Canopy Height (m)")
        elif "crown_cov" in key:
            ax.set_xlabel("Actual Crown Cover (%)")
            ax.set_ylabel("Estimated Crown Cover (%)")

        if "crown_cov" in key:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

# ───────────── Feature importance ───────────────────────────────────────────
importances = {}
for key, model in saved_models.items():
    season = key.rsplit("_", 1)[1]
    importances[key] = pd.Series(
        model.feature_importances_,
        index=feature_map[season]
    )

importance_df = pd.DataFrame(importances).T
importance_df.index.name = "Model"

imp_path = os.path.join(SAVE_DIR, "feature_importances_blockcv.csv")
importance_df.to_csv(imp_path)

# ───────────── Feature importance plots ─────────────────────────────────────
titles_map = {
    "med_CHM_PS":   "Canopy Height – Peak Summer",
    "med_CHM_LS":   "Canopy Height – Late Summer",
    "crown_cov_PS": "Crown Cover – Peak Summer",
    "crown_cov_LS": "Crown Cover – Late Summer"
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
bar_colour = "#B3B3B3"

for i, key in enumerate(target_models):
    imp = importance_df.loc[key].sort_values(ascending=True)

    clean_names = []
    for f in imp.index:
        f = f.replace("_PS", "").replace("_LS", "").lower()
        if "swir1" in f:
            clean_names.append("SWIR1")
        elif "swir2" in f:
            clean_names.append("SWIR2")
        elif "ndvi" in f:
            clean_names.append("NDVI")
        elif "nir" in f:
            clean_names.append("NIR")
        elif "pc" in f:
            clean_names.append(f.upper())
        elif "tcb" in f:
            clean_names.append("TC Brightness")
        elif "tcg" in f:
            clean_names.append("TC Greenness")
        elif "tcw" in f:
            clean_names.append("TC Wetness")
        else:
            clean_names.append(f.capitalize())

    sns.barplot(x=imp.values, y=clean_names, ax=axes[i], color=bar_colour)
    axes[i].set_title(titles_map[key])
    axes[i].set_xlabel("Importance")
    axes[i].set_ylabel("Feature")
    axes[i].set_xlim(0, 0.20)

plt.tight_layout()
plt.show()

# ───────────── Save prediction CSVs ─────────────────────────────────────────
pred_dir = os.path.join(SAVE_DIR, "predictions_csv")
os.makedirs(pred_dir, exist_ok=True)

for key, res in results.items():
    pd.DataFrame(res).to_csv(
        os.path.join(pred_dir, f"{key}_predictions.csv"),
        index=False
    )

print("\nAll outputs saved in:", SAVE_DIR)
print(" • Metrics           → model_metrics_blockcv.csv")
print(" • Feature importances → feature_importances_blockcv.csv")
