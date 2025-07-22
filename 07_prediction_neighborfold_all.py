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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error

LOAD_SAVED_MODELS = False                
SAVE_DIR = "results/Regression_models/BlockCV_4models100m2leafmin"
os.makedirs(SAVE_DIR, exist_ok=True)

block_size_m  = 100                       # block width/height in metres
n_folds_max   = 20                     # upper limit on #folds (≤ n_blocks)
verbose_plots = True                     # quick on/off for scatter plots

# ───────────────────── Data ─────────────────────────────────────────────────
data = pd.read_csv("results/pixel_table/pixel_table_final_filt.csv")

PS_feats = ['blue_PS', 'green_PS', 'red_PS', 'NIR_PS',
            'SWIR1_PS', 'SWIR2_PS', 'NDVI_PS']
LS_feats = ['blue_LS', 'green_LS', 'red_LS', 'NIR_LS',
            'SWIR1_LS', 'SWIR2_LS', 'NDVI_LS']

targets      = ["med_CHM", "crown_cov"]
seasons      = ["PS", "LS"]
target_models = [f"{t}_{s}" for t in targets for s in seasons] 

feature_map = {"PS": PS_feats, "LS": LS_feats}

# ───────────── Build block  ────────────────────────────
METRES_PER_DEG = 111_000                 # coarse conversion 1° ≈ 111 km
cell_deg = block_size_m / METRES_PER_DEG

min_lon, min_lat = data.longitude.min(), data.latitude.min()
block_x = np.floor((data.longitude - min_lon) / cell_deg).astype(int)
block_y = np.floor((data.latitude  -  min_lat) / cell_deg).astype(int)
block_id = block_x * 10_000 + block_y                    # unique int

n_blocks = np.unique(block_id).size
n_splits = min(n_folds_max, n_blocks)                    # ≤ n_blocks
cv = GroupKFold(n_splits=n_splits)

# ───────────── Hyper-parameters ─────────────────────────────
best_params = {
    "med_CHM_PS":   {'max_depth': 40,   'max_features': 'sqrt',
                     'min_samples_leaf': 2, 'min_samples_split': 2,
                     'n_estimators': 200, 'random_state': 42},
    "med_CHM_LS":   {'max_depth': None, 'max_features': 'sqrt',
                     'min_samples_leaf': 2, 'min_samples_split': 2,
                     'n_estimators': 200, 'random_state': 42},
    "crown_cov_PS": {'max_depth': None, 'max_features': 'sqrt',
                     'min_samples_leaf': 2, 'min_samples_split': 2,
                     'n_estimators': 200, 'random_state': 42},
    "crown_cov_LS": {'max_depth': 40,   'max_features': 'sqrt',
                     'min_samples_leaf': 2, 'min_samples_split': 2,
                     'n_estimators': 200, 'random_state': 42},
}

results, saved_models = {}, {}
if LOAD_SAVED_MODELS:
    print("Loading saved models/predictions …")
    for key in target_models:
        m_path = os.path.join(SAVE_DIR, f"{key}.joblib")
        p_path = os.path.join(SAVE_DIR, f"{key}_pred.pkl")
        saved_models[key] = joblib.load(m_path)
        with open(p_path, "rb") as f:
            results[key] = pickle.load(f)
    print("Done.")
else:
    print("Training models with block CV …")
    with tqdm(total=len(target_models), desc="Training") as pbar:
        for key in target_models:
            target, season = key.rsplit("_", 1)
            X = data[feature_map[season]]
            y = data[target]

            model = RandomForestRegressor(**best_params[key])
            y_pred = cross_val_predict(model, X, y,
                                       cv=cv, groups=block_id, n_jobs=-1)
            model.fit(X, y)

            results[key] = {"true": y.values, "pred": y_pred}
            saved_models[key] = model

            # save artefacts
            joblib.dump(model, os.path.join(SAVE_DIR, f"{key}.joblib"))
            with open(os.path.join(SAVE_DIR, f"{key}_pred.pkl"), "wb") as f:
                pickle.dump(results[key], f)
            pbar.update(1)
    print("Models & predictions saved.")

def get_metrics(y_true, y_pred):
    return (r2_score(y_true, y_pred),
            np.sqrt(mean_squared_error(y_true, y_pred)))

print("Block-level CV results"
      f"  (block ≈ {block_size_m} m, folds={n_splits})")
for key in target_models:
    r2, rmse = get_metrics(results[key]["true"], results[key]["pred"])
    print(f"\n  {key:<13s}  R²={r2:5.3f}   RMSE={rmse:6.2f}")

# ───────────── Scatter plots (all 4 models) ────────────────────────────────
if verbose_plots:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    title_map = {
        "med_CHM_PS": "Canopy Height – Peak Summer",
        "med_CHM_LS": "Canopy Height – Late Summer",
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
        r2, rmse = get_metrics(true, pred)
        mae = mean_absolute_error(true, pred) 

        cvals = crown_vals if "CHM" in key else chm_vals
        norm  = norm_cc    if "CHM" in key else norm_chm
        cmap  = cmap_cc    if "CHM" in key else cmap_chm
        colors = cmap(norm(cvals))

        ax = axes[idx]
        ax.scatter(true, pred, c=colors, alpha=0.6, s=14)
        ax.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', linewidth=1)
        ax.set_title(f"{title_map[key]}\nR²={r2:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f} ")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

        if "crown_cov" in key:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

    # add colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35]) 
    cbar_ax2 = fig.add_axes([0.92, 0.10, 0.02, 0.35])
    fig.colorbar(cm.ScalarMappable(norm=norm_cc, cmap=cmap_cc), cax=cbar_ax1).set_label("Crown Cover (%)")
    fig.colorbar(cm.ScalarMappable(norm=norm_chm, cmap=cmap_chm), cax=cbar_ax2).set_label("Canopy Height (m)")

    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    plt.show()


###Feature importance
importances = {}
for key, model in saved_models.items():
    season = key.rsplit("_", 1)[1]        
    feats   = feature_map[season]
    importances[key] = pd.Series(model.feature_importances_, index=feats)

# --- Plot 2×2 bar charts ----------------------------------------------------
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
    imp = importances[key].sort_values(ascending=True)

    # prettify feature names
    cleaned = []
    for f in imp.index:
        f_clean = f.replace("_PS", "").replace("_LS", "").lower()
        if "swir1" in f_clean:
            cleaned.append("SWIR1")
        elif "swir2" in f_clean:
            cleaned.append("SWIR2")
        elif "ndvi"  in f_clean:
            cleaned.append("NDVI")
        elif "nir"   in f_clean:
            cleaned.append("NIR")
        else:
            cleaned.append(f_clean.capitalize())

    sns.barplot(x=imp.values, y=cleaned, ax=axes[i], color=bar_colour)
    axes[i].set_title(titles_map[key])
    axes[i].set_xlabel("Importance")
    axes[i].set_ylabel("Feature")
    axes[i].set_xlim(0, 0.20)        

plt.tight_layout()
plt.show()

importance_df = pd.DataFrame(importances).T
importance_df.index.name = "Model"
imp_path = os.path.join(SAVE_DIR, "feature_importances_blockcv.csv")
importance_df.to_csv(imp_path)
print(f"\n Feature importances saved → {imp_path}")

metrics = []
for key, res in results.items():
    r2, rmse = get_metrics(res["true"], res["pred"])
    mae = mean_absolute_error(res["true"], res["pred"])
    metrics.append({"Model": key, "R2": r2, "RMSE": rmse, "MAE": mae})

metrics_df = pd.DataFrame(metrics)
met_path = os.path.join(SAVE_DIR, "model_metrics_blockcv.csv")
metrics_df.to_csv(met_path, index=False)
print(f"\n Metrics saved → {met_path}")

# ───────────── predictions -----------------------------------
pred_dir = os.path.join(SAVE_DIR, "predictions_csv")
os.makedirs(pred_dir, exist_ok=True)

for key, res in results.items():
    pd.DataFrame({"True": res["true"], "Predicted": res["pred"]}) \
        .to_csv(os.path.join(pred_dir, f"{key}_predictions.csv"), index=False)

print(f"\n Predictions saved to CSVs in: {pred_dir}")


# Reload Feature Importances 
importance_path = os.path.join(SAVE_DIR, "feature_importance_20folds_tuned.csv")
importance_df = pd.read_csv(importance_path, index_col=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
color = "#B3B3B3"

for i, (key, row) in enumerate(importance_df.iterrows()):
    imp = row.sort_values(ascending=True)
    
    clean_feature_names = []
    for feat in imp.index:
        feat_clean = feat.replace("_PS", "").replace("_LS", "").lower()
        if "swir1" in feat_clean:
            clean_feature_names.append("SWIR1")
        elif "swir2" in feat_clean:
            clean_feature_names.append("SWIR2")
        elif "ndvi" in feat_clean:
            clean_feature_names.append("NDVI")
        elif "nir" in feat_clean:
            clean_feature_names.append("NIR")
        else:
            clean_feature_names.append(feat_clean.capitalize())
    
    sns.barplot(x=imp.values, y=clean_feature_names, ax=axes[i], color=color)
    axes[i].set_title(f"{titles_map[key]}")
    axes[i].set_xlabel("Importance")
    axes[i].set_ylabel("Feature")
    axes[i].set_xlim(0, 0.2)  

plt.savefig("results/figures/01_Regression_full_dataset/feature_importance.svg", format="svg", bbox_inches="tight")
plt.tight_layout()
plt.show()















