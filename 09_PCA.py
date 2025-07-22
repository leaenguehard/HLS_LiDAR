# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:50:27 2025
PCA of spectral bands
@author: leengu001
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
from matplotlib.patches import Ellipse
import numpy as np

# Load data
data_f = pd.read_csv("results//pixel_table/pixel_table_final_filt.csv")

# Spectral band lists
sp_bands_PS = ["blue_PS", "green_PS", "red_PS", "NIR_PS", "SWIR1_PS", "SWIR2_PS", "NDVI_PS"]
sp_bands_LS = ["blue_LS", "green_LS", "red_LS", "NIR_LS", "SWIR1_LS", "SWIR2_LS", "NDVI_LS"]

# Custom palette for 'cat'
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

# Manual offsets for arrow labels
label_offsets = {
    'blue_LS': (0.25, -0.25),
    'blue_PS': (0.35, -0.25),
    'SWIR2_PS': (0.2, 0.0),
    'SWIR1_PS': (0.2, 0.0),
    'SWIR2_LS': (0.25, 0.2),
    'SWIR1_LS': (0.25, 0.0),
    'red_LS': (0.35, 0.1),
    'red_PS': (0.3, -0.1),
    'NIR_LS': (0.25, 0.3),
    'NIR_PS': (0.2, 0.3),
    'NDVI_LS': (-0.2, 0.5),
    'NDVI_PS': (-1.7, 0.5),
    'green_PS': (0.2, 0.2),
    'green_LS': (0.2, 0.4),
}

# --- Run PCA only once to get pca_ps, pca_ls and loadings
def compute_pca(data, band_list):
    X = StandardScaler().fit_transform(data[band_list])
    pca = PCA()
    pcs = pca.fit_transform(X)
    return pca, pcs

pca_ps, pcs_ps = compute_pca(data_f, sp_bands_PS)
pca_ls, pcs_ls = compute_pca(data_f, sp_bands_LS)

# Save loadings for later
loadings_ps = pd.DataFrame(pca_ps.components_.T, index=sp_bands_PS)
loadings_ls = pd.DataFrame(pca_ls.components_.T, index=sp_bands_LS)


# Create plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Peak Summer PCA plot
ax = axes[0]
for cat in categories:
    idx = data_f["cat"] == cat
    if idx.any():
        ax.scatter(pcs_ps[idx, 0], pcs_ps[idx, 1],
                   label=cat, color=palette_dict[cat], s=10, alpha=0.7)

# Arrows for loadings
for band in loadings_ps.index:
    x_loading = loadings_ps.loc[band, 0] * 5
    y_loading = loadings_ps.loc[band, 1] * 5
    ax.arrow(0, 0, x_loading, y_loading, color='black', alpha=1, head_width=0.2, head_length=0.2)
    dx, dy = label_offsets.get(band, (0, 0))
    
    # Remove "_PS" or "_LS" and format nicely
    clean_band = band.replace('_PS', '').replace('_LS', '')
    clean_band = clean_band.replace('SWIR1', 'SWIR 1').replace('SWIR2', 'SWIR 2')
    clean_band = clean_band.upper() if clean_band.lower() in ['nir', 'ndvi', 'swir 1', 'swir 2'] else clean_band.capitalize()
    
    ax.text(x_loading*0.96 + dx, y_loading*0.96 + dy, clean_band, color='black', fontsize=12, fontweight='bold')

ax.set_title("Peak Summer", fontsize=15)
ax.set_xlabel(f"PC1 ({pca_ps.explained_variance_ratio_[0]*100:.1f}%)", fontsize=15)
ax.set_ylabel(f"PC2 ({pca_ps.explained_variance_ratio_[1]*100:.1f}%)", fontsize=15)
ax.grid(True, alpha=0.3)

# --- Late Summer PCA plot
ax = axes[1]
for cat in categories:
    idx = data_f["cat"] == cat
    if idx.any():
        ax.scatter(pcs_ls[idx, 0], pcs_ls[idx, 1],
                   label=cat, color=palette_dict[cat], s=10, alpha=0.7)

# Arrows for loadings
for band in loadings_ls.index:
    x_loading = loadings_ls.loc[band, 0] * 5
    y_loading = loadings_ls.loc[band, 1] * 5
    ax.arrow(0, 0, x_loading, y_loading, color='black', alpha=1, head_width=0.2, head_length=0.2)
    dx, dy = label_offsets.get(band, (0, 0))
    
    # Remove "_PS" or "_LS" and format nicely
    clean_band = band.replace('_PS', '').replace('_LS', '')
    clean_band = clean_band.replace('SWIR1', 'SWIR 1').replace('SWIR2', 'SWIR 2')
    clean_band = clean_band.upper() if clean_band.lower() in ['nir', 'ndvi', 'swir 1', 'swir 2'] else clean_band.capitalize()
    
    ax.text(x_loading*0.96 + dx, y_loading*0.96 + dy, clean_band, color='black', fontsize=12, fontweight='bold')

ax.set_title("Late Summer", fontsize=15)
ax.set_xlabel(f"PC1 ({pca_ls.explained_variance_ratio_[0]*100:.1f}%)", fontsize=15)
ax.set_ylabel(f"PC2 ({pca_ls.explained_variance_ratio_[1]*100:.1f}%)", fontsize=15)
ax.grid(True, alpha=0.3)

# --- Updated legend position ---
# Get handles and labels
handles, labels = axes[1].get_legend_handles_labels()

# Add legend INSIDE the second axis (Late Summer plot)
axes[1].legend(
    handles, labels,
    title="Category",
    loc='upper right',  # Inside top right of subplot
    frameon=True,       # Box around legend
    fontsize=10,
    title_fontsize=11
)

# plt.savefig("results/figures/PCA.svg", format="svg", bbox_inches="tight")
plt.tight_layout()
plt.show()

##############

# Create loadings table for PC1–PC3
def get_loadings_table(pca, band_list, label):
    n_components = pca.components_.shape[0]
    loadings = pca.components_.T  # shape: (features, components)
    explained_var = pca.explained_variance_ratio_ * 100
    columns = [f"PC{i+1} ({explained_var[i]:.1f}%)" for i in range(n_components)]
    return pd.DataFrame(loadings, index=band_list, columns=columns)

pd.set_option("display.max_columns", None)  # show all PC columns
pd.set_option("display.width", 0)           # auto-fit wide tables

# Show full loadings tables
print("\n📊 PCA Loadings Table — Peak Summer")
loadings_ps = get_loadings_table(pca_ps, sp_bands_PS, "PS")
display(loadings_ps)

print("\n📊 PCA Loadings Table — Late Summer")
loadings_ls = get_loadings_table(pca_ls, sp_bands_LS, "LS")
display(loadings_ls)

# # Export all loadings to CSV
# loadings_ps.to_csv("results/pca_loadings_peak_summer.csv")
# loadings_ls.to_csv("results/pca_loadings_late_summer.csv")


# ## Save PCs to pixel table

# # Compute PC scores for Peak Summer
# X_ps = StandardScaler().fit_transform(data_f[sp_bands_PS])
# pcs_ps = PCA().fit_transform(X_ps)
# data_f[["PC1_PS", "PC2_PS", "PC3_PS"]] = pcs_ps[:, :3]

# # Compute PC scores for Late Summer
# X_ls = StandardScaler().fit_transform(data_f[sp_bands_LS])
# pcs_ls = PCA().fit_transform(X_ls)
# data_f[["PC1_LS", "PC2_LS", "PC3_LS"]] = pcs_ls[:, :3]

# # Save updated table with PC scores
# data_f.to_csv("results/pixel_table/pixel_table_final_filt_with_pcs.csv", index=False, encoding='utf-8-sig')


##### Ellipses 
# --- Helper function to plot ellipses
def plot_category_ellipse(ax, x, y, color, alpha=0.8):
    if len(x) < 2:
        return
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=lambda_[0]*4, height=lambda_[1]*4,
        angle=np.rad2deg(np.arccos(v[0, 0])),
        edgecolor=color,
        facecolor='none',  # No fill
        lw=3,
        alpha=alpha,
        linestyle='solid'
    )
    ax.add_patch(ell)

# --- Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Peak Summer plot
ax = axes[0]
for cat in categories:
    idx = data_f["cat"] == cat
    if idx.any():
        ax.scatter(pcs_ps[idx, 0], pcs_ps[idx, 1],
                   label=cat, color=palette_dict[cat], s=10, alpha=0.7)
        plot_category_ellipse(ax, pcs_ps[idx, 0], pcs_ps[idx, 1], palette_dict[cat])

# Add loadings arrows
for band in loadings_ps.index:
    x_loading = loadings_ps.loc[band, 0] * 5
    y_loading = loadings_ps.loc[band, 1] * 5
    ax.arrow(0, 0, x_loading, y_loading, color='black', alpha=1, head_width=0.2, head_length=0.2)
    dx, dy = label_offsets.get(band, (0, 0))
    clean_band = band.replace('_PS', '').replace('_LS', '')
    clean_band = clean_band.replace('SWIR1', 'SWIR 1').replace('SWIR2', 'SWIR 2')
    clean_band = clean_band.upper() if clean_band.lower() in ['nir', 'ndvi', 'swir 1', 'swir 2'] else clean_band.capitalize()
    ax.text(x_loading*0.96 + dx, y_loading*0.96 + dy, clean_band, color='black', fontsize=12, fontweight='bold')

ax.set_title("Peak Summer", fontsize=15)
ax.set_xlabel(f"PC1 ({pca_ps.explained_variance_ratio_[0]*100:.1f}%)", fontsize=15)
ax.set_ylabel(f"PC2 ({pca_ps.explained_variance_ratio_[1]*100:.1f}%)", fontsize=15)
ax.grid(True, alpha=0.3)

# --- Late Summer plot
ax = axes[1]
for cat in categories:
    idx = data_f["cat"] == cat
    if idx.any():
        ax.scatter(pcs_ls[idx, 0], pcs_ls[idx, 1],
                   label=cat, color=palette_dict[cat], s=10, alpha=0.7)
        plot_category_ellipse(ax, pcs_ls[idx, 0], pcs_ls[idx, 1], palette_dict[cat])

# Add loadings arrows
for band in loadings_ls.index:
    x_loading = loadings_ls.loc[band, 0] * 5
    y_loading = loadings_ls.loc[band, 1] * 5
    ax.arrow(0, 0, x_loading, y_loading, color='black', alpha=1, head_width=0.2, head_length=0.2)
    dx, dy = label_offsets.get(band, (0, 0))
    clean_band = band.replace('_PS', '').replace('_LS', '')
    clean_band = clean_band.replace('SWIR1', 'SWIR 1').replace('SWIR2', 'SWIR 2')
    clean_band = clean_band.upper() if clean_band.lower() in ['nir', 'ndvi', 'swir 1', 'swir 2'] else clean_band.capitalize()
    ax.text(x_loading*0.96 + dx, y_loading*0.96 + dy, clean_band, color='black', fontsize=12, fontweight='bold')

ax.set_title("Late Summer", fontsize=15)
ax.set_xlabel(f"PC1 ({pca_ls.explained_variance_ratio_[0]*100:.1f}%)", fontsize=15)
ax.set_ylabel(f"PC2 ({pca_ls.explained_variance_ratio_[1]*100:.1f}%)", fontsize=15)
ax.grid(True, alpha=0.3)

# --- Legend inside Late Summer plot
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(
    handles, labels,
    title="Category",
    loc='upper right',
    frameon=True,
    fontsize=12,
    title_fontsize=14
)
# plt.savefig("results/figures/PCA.svg", format="svg", bbox_inches="tight")
plt.tight_layout()
plt.show()








