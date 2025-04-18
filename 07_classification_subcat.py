# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:52:40 2025

@author: leengu001
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

# --- Load Data ---
data = pd.read_csv("results//pixel_table/pixel_table_final_filt.csv")

# --- Feature Sets ---
ps_features = ['blue_PS', 'green_PS', 'red_PS', 'NIR_PS', 'SWIR1_PS', 'SWIR2_PS', 'NDVI_PS']
ls_features = ['blue_LS', 'green_LS', 'red_LS', 'NIR_LS', 'SWIR1_LS', 'SWIR2_LS', 'NDVI_LS']

# --- Label Encoding ---
label_encoder = LabelEncoder()
data["cat_encoded"] = label_encoder.fit_transform(data["cat"])
class_names = label_encoder.classes_
n_classes = len(class_names)

# --- Cross-validation setup ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Store results ---
all_metrics = []
feature_importance = {}

# --- CLASSIFICATION LOOP ---
for season, feats in zip(["PS", "LS"], [ps_features, ls_features]):
    print(f"\n📅 Season: {season}")

    X = data[feats].values
    y = data["cat_encoded"].values

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"✅ Accuracy: {acc:.3f}")
    print("\n📋 Classification Report:")
    print(classification_report(y, y_pred, target_names=class_names))

    # --- Save metrics ---
    all_metrics.append({
        "Season": season,
        "Accuracy": acc,
        **{f"{label}_{metric}": report[label][metric]
           for label in class_names
           for metric in ["precision", "recall", "f1-score"]}
    })

    # --- Confusion Matrix Plot ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {season}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # --- ROC Curve (One-vs-Rest) ---
    y_bin = label_binarize(y, classes=np.arange(n_classes))
    model_ovr = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    y_score = cross_val_predict(model_ovr, X, y_bin, cv=cv, method='predict_proba')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve (One-vs-Rest) - {season}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # --- Fit on full data for feature importances ---
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feats)
    feature_importance[season] = importances

# --- Feature Importance Plot ---
for season in feature_importance:
    plt.figure(figsize=(8, 6))
    imp = feature_importance[season].sort_values(ascending=True)
    sns.barplot(x=imp.values, y=imp.index)
    plt.title(f"Feature Importance - {season}")
    plt.tight_layout()
    plt.show()

# --- Save Results ---
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("results/classification_metrics_by_season.csv", index=False)
print("📁 Saved classification metrics ")

feat_imp_df = pd.DataFrame(feature_importance)
feat_imp_df.to_csv("results/Classification/feature_importance_classification.csv")
print("📁 Saved feature importances")



# --- Compare PS vs LS Accuracy + F1 ---
# Melt the F1-scores only for plotting
f1_cols = [col for col in metrics_df.columns if "f1-score" in col]
f1_df = metrics_df.melt(id_vars=["Season"], value_vars=f1_cols,
                        var_name="Class", value_name="F1-score")

# Clean up class labels
f1_df["Class"] = f1_df["Class"].str.replace("_f1-score", "")

plt.figure(figsize=(10, 6))
sns.barplot(data=f1_df, x="Class", y="F1-score", hue="Season")
plt.title("F1-score per Class for Peak Season vs Late Summer")
plt.ylabel("F1-score")
plt.xlabel("Crown Cover Bin")
plt.legend(title="Season")
plt.tight_layout()
plt.show()

# Also plot overall accuracy comparison
plt.figure(figsize=(6, 4))
sns.barplot(data=metrics_df, x="Season", y="Accuracy")
plt.title("Overall Accuracy: Peak vs Late Summer")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
