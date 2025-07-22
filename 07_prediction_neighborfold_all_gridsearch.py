# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:28:50 2025

@author: leengu001
"""
import os
import pickle
import joblib
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ───────── CONFIG ───────────────────────────────────────────────────────────
BLOCK_SIZE_M      = 100                  # block width/height in metres
N_FOLDS_MAX       = 20                   # GroupKFold upper-bound (≤ n_blocks)
RANDOM_STATE      = 42
N_JOBS            = -1                   # all available cores
SAVE_DIR          = "results/Regression_models/GridSearch_4models100m"
os.makedirs(SAVE_DIR, exist_ok=True)

PARAM_GRID = {
    "n_estimators":      [100, 300],
    "max_depth":         [None, 20, 40],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [2, 6],
    "max_features":      ["sqrt", "log2"],
}

# ───────── Data ─────────────────────────────────────────────────────────────
DATA_PATH = "results/pixel_table/pixel_table_final_filt.csv"
data = pd.read_csv(DATA_PATH)

PS_FEATS = ['blue_PS', 'green_PS', 'red_PS', 'NIR_PS',
            'SWIR1_PS', 'SWIR2_PS', 'NDVI_PS']
LS_FEATS = ['blue_LS', 'green_LS', 'red_LS', 'NIR_LS',
            'SWIR1_LS', 'SWIR2_LS', 'NDVI_LS']

TARGETS  = ["med_CHM", "crown_cov"]
SEASONS  = ["PS", "LS"]
FEATURE_MAP = {"PS": PS_FEATS, "LS": LS_FEATS}
MODEL_KEYS = [f"{t}_{s}" for t in TARGETS for s in SEASONS]   # 4 total

# ───────── Build block groups (100 m grid) ──────────────────────────────────
METRES_PER_DEG = 111_000                
cell_deg = BLOCK_SIZE_M / METRES_PER_DEG

min_lon, min_lat = data.longitude.min(), data.latitude.min()
block_x  = np.floor((data.longitude - min_lon) / cell_deg).astype(int)
block_y  = np.floor((data.latitude  -  min_lat) / cell_deg).astype(int)
block_id = block_x * 10_000 + block_y                      

n_blocks = block_id.nunique()
n_splits = min(N_FOLDS_MAX, n_blocks)
cv = GroupKFold(n_splits=n_splits)

print(f"\n Spatial block-CV: {n_blocks} blocks  →  {n_splits} folds")

def metrics(y_true, y_pred):
    """Return R², RMSE, MAE (in that order)."""
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# ───────── Grid Search ─────────────────────────────────────────
report_rows = []
best_models = {}

print("Running GridSearchCV for 4 models (block-aware)…")
with tqdm(total=len(MODEL_KEYS), desc="Grid Searching") as pbar:
    for key in MODEL_KEYS:
        target, season = key.rsplit("_", 1)
        X  = data[FEATURE_MAP[season]]
        y  = data[target]

        rf = RandomForestRegressor(random_state=RANDOM_STATE)
        gs = GridSearchCV(
            estimator=rf,
            param_grid=PARAM_GRID,
            cv=cv,
            scoring="r2",
            n_jobs=N_JOBS,
            verbose=0,
        )
        gs.fit(X, y, groups=block_id)             

        best_model = gs.best_estimator_
        y_pred = cross_val_predict(
            best_model, X, y,
            cv=cv, groups=block_id, n_jobs=N_JOBS
        )

        r2, rmse, mae = metrics(y, y_pred)

        model_path = os.path.join(SAVE_DIR, f"{key}_best.joblib")
        joblib.dump(best_model, model_path)

        preds_path = os.path.join(SAVE_DIR, f"{key}_pred.pkl")
        with open(preds_path, "wb") as f:
            pickle.dump({"true": y.values, "pred": y_pred}, f)

        report_rows.append({
            "Model": key,
            "R2":   round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE":  round(mae, 4),
            "BestParams": gs.best_params_,
        })
        best_models[key] = best_model

        pbar.update(1)

# ───────── Export CSV summary ───────────────────────────────────────────────
report_df = pd.DataFrame(report_rows)
csv_path  = os.path.join(SAVE_DIR, "gridsearch_metrics_blockcv.csv")
report_df.to_csv(csv_path, index=False)

print("Block-CV Grid-Search Results"
      f"  (block ≈ {BLOCK_SIZE_M} m, folds={n_splits})")
for row in report_rows:
    print(f"\n  {row['Model']:<13}  "
          f"R²={row['R2']:.3f}   RMSE={row['RMSE']:.2f}   "
          f"MAE={row['MAE']:.2f}")
    print(f"     Best → {row['BestParams']}")

print(f"\n Metrics & parameters written to {csv_path}")
print(f"\n Individual best-model pickles in  {SAVE_DIR}/")
