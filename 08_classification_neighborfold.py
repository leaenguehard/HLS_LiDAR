# -*- coding: utf-8 -*-
"""
RF classification (PS + LS models)

Author  : leengu001
Updated : 2025-07-07
"""
import os, pickle, joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.multiclass import OneVsRestClassifier

LOAD_SAVED_MODELS = False             
SAVE_DIR = "results/Classification_models/rev/BlockCV_20folds100mTCs"
os.makedirs(SAVE_DIR,           exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "predictions_csv"), exist_ok=True)

block_size_m  = 100                    
n_folds_max   = 20                    

data = pd.read_csv("results/pixel_table/pixel_table_final_filt.csv")

PS_feats = [
    'blue_PS', 'green_PS', 'red_PS', 'NIR_PS',
    'SWIR1_PS', 'SWIR2_PS', 'NDVI_PS','TCB_PS', 'TCW_PS', 'TCG_PS'
]

LS_feats = [
    'blue_LS', 'green_LS', 'red_LS', 'NIR_LS',
    'SWIR1_LS', 'SWIR2_LS', 'NDVI_LS','TCB_LS', 'TCW_LS', 'TCG_LS'
]

feature_map = {"PS": PS_feats, "LS": LS_feats}

# encode labels
label_encoder  = LabelEncoder()
data["cat_encoded"] = label_encoder.fit_transform(data["cat"])
class_names    = label_encoder.classes_
n_classes      = len(class_names)

METRES_PER_DEG = 111_000                      
cell_deg = block_size_m / METRES_PER_DEG

min_lon, min_lat = data.longitude.min(), data.latitude.min()
block_x = np.floor((data.longitude - min_lon) / cell_deg).astype(int)
block_y = np.floor((data.latitude  -  min_lat) / cell_deg).astype(int)
block_id = block_x * 10_000 + block_y

n_blocks = np.unique(block_id).size
n_splits = min(n_folds_max, n_blocks)
cv = GroupKFold(n_splits=n_splits)

#  Best hyper-parameters (pre-tuned) ─────────────────────────────
best_params_dict = {
    "PS": {'max_depth': 20,  'max_features': 'sqrt',
           'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300},
    "LS": {'max_depth': None,'max_features': 'sqrt',
           'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}
}
season_title_mapping = {"PS": "Peak Summer", "LS": "Late Summer"}

all_metrics, feature_importance = [], {}
conf_matrices, roc_curves, results = {}, {}, {}

# ─────────── Train per season ───────────────────────────────────────
for season in ["PS", "LS"]:
    print(f"\n  Season: {season}")
    feats = feature_map[season]
    X     = data[feats].values
    y     = data["cat_encoded"].values
    mdl_p = os.path.join(SAVE_DIR, f"model_{season}.joblib")
    predp = os.path.join(SAVE_DIR, f"{season}_predictions.pkl")

    if LOAD_SAVED_MODELS and os.path.exists(mdl_p) and os.path.exists(predp):
        print("Loading saved model & predictions …")
        model  = joblib.load(mdl_p)
        with open(predp, "rb") as fh:
            y_pred = pickle.load(fh)
    else:
        print("Training RF with block CV …")
        model  = RandomForestClassifier(**best_params_dict[season],
                                        random_state=42)
        y_pred = cross_val_predict(model, X, y,
                                   cv=cv, groups=block_id, n_jobs=-1)
        model.fit(X, y)
        joblib.dump(model, mdl_p)
        with open(predp, "wb") as fh:
            pickle.dump(y_pred, fh)
        print(" Model & predictions saved")

    acc   = accuracy_score(y, y_pred)
    rep   = classification_report(y, y_pred,
                                  target_names=class_names, output_dict=True)
    cm    = confusion_matrix(y, y_pred)
    conf_matrices[season] = cm

    all_metrics.append(
        {"Season": season, "Accuracy": acc,
         **{f"{lbl}_{m}": rep[lbl][m]
            for lbl in class_names
            for m in ["precision", "recall", "f1-score"]}}
    )

    print(f" Accuracy: {acc:.3f}")
    print(classification_report(y, y_pred, target_names=class_names))

    y_bin   = label_binarize(y, classes=np.arange(n_classes))
    ovr_mdl = OneVsRestClassifier(model)
    y_score = cross_val_predict(
                ovr_mdl, X, y_bin, cv=cv, groups=block_id,
                method='predict_proba', n_jobs=-1)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_curves[season] = (fpr, tpr, roc_auc)

    feature_importance[season] = pd.Series(model.feature_importances_,
                                           index=feats)

    out_csv = os.path.join(SAVE_DIR, "predictions_csv",
                           f"{season}_predictions.csv")
    pd.DataFrame({
        "True_Label": label_encoder.inverse_transform(y),
        "Predicted_Label": label_encoder.inverse_transform(y_pred)
    }).to_csv(out_csv, index=False)

# ─────────── Plot confusion & ROC per season ───────────────────────────────

custom_palette = [
    "#FC8D62",  # 5–20%
    "#8DA0CB",  # 20–50%
    "#E78AC3",  # 50–80% >5
    "#66C2A5",  # 50–80% <5
    "#FFD92F",  # 80–100% <5
    "#A6D854"   # 80–100% >5
]

categories = [
    '5–20%', '20–50%', '50–80% >5', 
    '50–80% <5', '80–100% <5', '80–100% >5'
]
palette_dict = dict(zip(categories, custom_palette))
category_to_index = {cat: i for i, cat in enumerate(class_names)}

for season in ["PS", "LS"]:
    cm  = conf_matrices[season]
    fpr, tpr, roc_auc = roc_curves[season]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=class_names, yticklabels=class_names, ax=ax[0])
    ax[0].set_title(season_title_mapping[season])
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    # ROC Curve - plot in desired category order
    for cat in categories:
        i = category_to_index[cat]
        ax[1].plot(fpr[i], tpr[i],
                   label=f"{cat} (AUC={roc_auc[i]:.2f})",
                   color=palette_dict[cat])

    ax[1].plot([0, 1], [0, 1], 'k--')
    ax[1].set_title(season_title_mapping[season])
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

# ─────────── Save metrics & importances ─────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(SAVE_DIR, "classification_metrics_block.csv"),
                  index=False, encoding="utf-8-sig")
feature_df = pd.DataFrame(feature_importance)
feature_df.to_csv(os.path.join(SAVE_DIR, "feature_importances_block.csv"),
                  encoding="utf-8-sig")

print("\n All outputs saved in:", SAVE_DIR)
print(f"   • Metrics  → classification_metrics_blockcv.csv")
print(f"   • Features → feature_importances_blockcv.csv")
print(f"   • Predictions per season in predictions_csv/")


# ───────────── Feature importance plots (PS & LS) ───────────────────────────

titles_map = {
    "PS": "Peak Summer",
    "LS": "Late Summer"
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
bar_colour = "#B3B3B3"

for ax, season in zip(axes, ["PS", "LS"]):
    imp = feature_importance[season].sort_values(ascending=True)

    # Clean feature names (same logic as your other script)
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

    sns.barplot(
        x=imp.values,
        y=clean_names,
        ax=ax,
        color=bar_colour
    )

    ax.set_title(titles_map[season])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, 0.20)

plt.tight_layout()
plt.show()

