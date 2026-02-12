# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:07:18 2025
Figures
@author: leengu001
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

# Load data
data = pd.read_csv("results//pixel_table/pixel_table_final_filt.csv")
#data = pd.read_csv("results//pixel_table/pixel_table_with_above.csv")

#data = data_f[~data_f['UAV_tiff'].isin(['EN23688_final.tif', 'EN23689_final.tif'])]

# Define custom palette matching categories
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

# Define band sets
ps_bands = ['blue_PS', 'green_PS', 'red_PS', 'NIR_PS', 'SWIR1_PS', 'SWIR2_PS']
ls_bands = ['blue_LS', 'green_LS', 'red_LS', 'NIR_LS', 'SWIR1_LS', 'SWIR2_LS']

#### Figure spectral bands
# Melt and label Peak Summer (PS)
ps_melted = data.melt(id_vars='cat', value_vars=ps_bands,
                      var_name='Band', value_name='Reflectance')
ps_melted['Season'] = 'Peak Summer'

# Melt and label Late Summer (LS)
ls_melted = data.melt(id_vars='cat', value_vars=ls_bands,
                      var_name='Band', value_name='Reflectance')
ls_melted['Season'] = 'Late Summer'

# Combine both
plot_data = pd.concat([ps_melted, ls_melted])

# Clean band names
plot_data['Band'] = plot_data['Band'].str.replace('_PS', '', regex=False)
plot_data['Band'] = plot_data['Band'].str.replace('_LS', '', regex=False)

# Map to proper names (capitalize correctly)
band_name_mapping = {
    'blue': 'Blue',
    'green': 'Green',
    'red': 'Red',
    'NIR': 'NIR',
    'SWIR1': 'SWIR1',
    'SWIR2': 'SWIR2'
}
plot_data['Band'] = plot_data['Band'].map(band_name_mapping)

# Enforce the correct band order
band_order = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
cat_type = CategoricalDtype(categories=band_order, ordered=True)
plot_data['Band'] = plot_data['Band'].astype(cat_type)

# Plot using FacetGrid
g = sns.FacetGrid(plot_data, col='Season', height=6, aspect=1.2, hue='cat', palette=palette_dict)

# Line plot
g.map_dataframe(sns.lineplot, x='Band', y='Reflectance', linewidth=2)

# Set titles and axis labels
g.set_titles("{col_name}")  # Ignore fontsize here
g.set_axis_labels("Spectral Band", "Mean Reflectance")
g.set_xlabels("Spectral Band", fontsize=14)
g.set_ylabels("Mean Reflectance", fontsize=14)

# Force bigger facet titles
for ax in g.axes.flatten():
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_title(ax.get_title(), fontsize=14)  
g.add_legend(title='Category')
g._legend.set_bbox_to_anchor((0.95, 0.8))
g._legend.set_frame_on(True)
g._legend.set_title('Category', prop={'size': 14})  # Legend title
for text in g._legend.texts:
    text.set_fontsize(12)  # Legend labels


plt.savefig("results/figures/reflectance_cat.svg", format="svg", bbox_inches="tight")
plt.tight_layout()
plt.show()

############### Plot ch, and crown cover distribution
# KDE plot with custom color and labels
a = sns.displot(data, x="med_CHM", kind="kde", color="#B3B3B3")  # Set curve color
a.ax.set_title("Median CHM per Pixel")
a.ax.set_xlabel("Median Canopy Height (m)")
a.ax.set_ylabel("Density")
plt.show()

# Create figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- First plot: med_CHM
sns.kdeplot(data=data, x="med_CHM", ax=axes[0], color="#B3B3B3", fill=False)
axes[0].set_title("Median CHM per Pixel")
axes[0].set_xlabel("Median Canopy Height (m)")
axes[0].set_ylabel("Density")
axes[0].set_xlim(0, 30) 
axes[0].grid(False)# alpha =0.1)

# --- Second plot: crown_cov
sns.kdeplot(data=data, x="crown_cov", ax=axes[1], color="#B3B3B3", fill=False)
axes[1].set_title("Crown Cover (%)")
axes[1].set_xlabel("Crown Cover (%)")
axes[1].set_ylabel("Density")
axes[1].set_xlim(5, 100) 
axes[1].grid(False)# alpha = 0.1)

plt.tight_layout()
plt.show()

############### Plot distribution NDVI/category
#sns.scatterplot(data = data, x = "med_CHM", y = "NDVI_PS", hue ="cat", palette = palette_dict).set_title("Peak Summer")#, style = "UTM_zone")

plt.figure(figsize=(10, 6))
ax = sns.scatterplot(
    data=data,
    x="med_CHM",
    y="TCG_PS",
    hue="cat",
    palette=palette_dict
)

# ax.set_title("Distribution of CHM and Red PS")
ax.set_xlabel("Canopy height (m)")
ax.set_ylabel("TCG")
plt.legend(title="Category")
plt.show()

###
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(
    data=data,
    x="med_CHM",
    y="TCG_PS",
    hue="cat",
    palette=palette_dict,
    ax=axes[0]
)
# add regression line 
sns.regplot(
    data=data,
    x="med_CHM",
    y="TCG_PS",
    scatter=False,
    ax=axes[0],
    line_kws={"color": "black", "linewidth": 1.5}
)

axes[0].set_title("Peak Summer")
axes[0].set_xlabel("Canopy height (m)")
axes[0].set_ylabel("TCG")
axes[0].legend_.remove()

# Second plot: Late Summer
sns.scatterplot(
    data=data,
    x="med_CHM",
    y="TCG_LS",
    hue="cat",
    palette=palette_dict,
    ax=axes[1]
)
# add regression line 
sns.regplot(
    data=data,
    x="med_CHM",
    y="TCG_LS",
    scatter=False,
    ax=axes[1],
    line_kws={"color": "black", "linewidth": 1.5}
)
axes[1].set_title("Late Summer")
axes[1].set_xlabel("Canopy height (m)")
axes[1].set_ylabel("TCG")
axes[1].legend(title="Category")

# Adjust layout
plt.tight_layout()
plt.show()

###### Crown cover plots
import matplotlib as mpl

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(right=0.88, wspace = 0.15)   # leave the right 12% empty

norm = mpl.colors.Normalize(data["med_CHM"].min(), data["med_CHM"].max())
sm   = mpl.cm.ScalarMappable(cmap="plasma", norm=norm)
sm.set_array([])

for ax, ycol, title in zip(
    axes,
    ["TCG_PS", "TCG_LS"],
    ["Peak Summer", "Late Summer"]
):
    sns.scatterplot(
        data=data,
        x="crown_cov",
        y=ycol,
        hue="med_CHM",
        palette="plasma",
        hue_norm=norm,
        legend=False,
        ax=ax,
        alpha=0.7,
        edgecolor="none",
    )
    sns.regplot(
        data=data,
        x="crown_cov",
        y=ycol,
        scatter=False,
        ax=ax,
        line_kws={"color": "black", "linewidth": 1.5},
    )
    ax.set_title(title)
    ax.set_xlabel("Crown cover (%)")
    ax.set_ylabel("TCG")

cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coords
fig.colorbar(sm, cax=cax, orientation="vertical", label="Canopy Height (m)")

plt.show()

## Correlation coefficients
from scipy.stats import pearsonr

structure_vars = {
    "Crown Cover": "crown_cov",
    "Canopy Height": "med_CHM"
}
seasons = {
    "PS": "Peak Summer",
    "LS": "Late Summer"
}
spectral_vars = {
    "NDVI": "NDVI",
    "TCW": "TCW"
}

results = []
print("\n========== Pearson Correlation Results ==========\n")

for struct_name, struct_col in structure_vars.items():
    for season_code, season_name in seasons.items():
        for spec_name, spec_prefix in spectral_vars.items():
            
            spec_col = f"{spec_prefix}_{season_code}"
            
            subset = data[[struct_col, spec_col]].dropna()
            
            r, p = pearsonr(subset[struct_col], subset[spec_col])
            
            print(f"{struct_name} ({season_name}) vs {spec_name}")
            print(f"   r = {r:.3f} | p = {p:.4f}")
            print("-" * 60)
            
            results.append({
                "Structure Variable": struct_name,
                "Season": season_name,
                "Spectral Index": spec_name,
                "Pearson r": round(r, 3),
                "p-value": round(p, 4)
            })


corr_df = pd.DataFrame(results)
# Save to CSV
corr_df.to_csv("pearson_correlations_structure_vs_spectral.csv", index=False)
print("\nSaved table as: pearson_correlations_structure_vs_spectral.csv")
corr_df



#sns.scatterplot(data = data, x = "crown_cov", y = "NDVI_LS", hue ="med_CHM").set_title("Distribution of crown cover and NDVI LS")#, style = "UTM_zone")


############### Plot frequencies
cc_LS = pd.read_csv("results/Regression_models/rev/BlockCV_models100TCPC/predictions_csv/crown_cov_LS_predictions.csv")
cc_PS = pd.read_csv("results/Regression_models/rev/BlockCV_models100TCPC/predictions_csv/crown_cov_PS_predictions.csv")
ch_LS = pd.read_csv("results/Regression_models/rev/BlockCV_models100TCPC/predictions_csv/med_CHM_LS_predictions.csv")
ch_PS = pd.read_csv("results/Regression_models/rev/BlockCV_models100TCPC/predictions_csv/med_CHM_PS_predictions.csv")

## without shrubs
# cc_LS = pd.read_csv("results/Regression_models/Tuned_20folds_final/predictions_csv/crown_cov_LS_predictions.csv")
# cc_PS = pd.read_csv("results/Regression_models/Tuned_20folds_final/predictions_csv/crown_cov_PS_predictions.csv")
# ch_LS = pd.read_csv("results/Regression_models/Tuned_20folds_final/predictions_csv/med_CHM_LS_predictions.csv")
# ch_PS = pd.read_csv("results/Regression_models/Tuned_20folds_final/predictions_csv/med_CHM_PS_predictions.csv")

# sns.scatterplot(x='True', y = 'Predicted', data = cc_LS)
# sns.histplot(data=cc_LS, x="True", kde=True, color="blue", label="True", alpha=0.5)
# sns.kdeplot(data=cc_LS, x="True", color="blue", label="True", linewidth=2)
# sns.kdeplot(data=cc_LS, x="Predicted", color="orange", label="Predicted", linewidth=2)

# Create figure and 4 subplots
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ── X-AXIS LIMITS ───────────────────────────────────────────────
axes[0,0].set_xlim(0, 30)       # Canopy Height – Peak Summer
axes[1,0].set_xlim(0, 30)     # Canopy Height – Late  Summer
axes[0,1].set_xlim(5, 100)      # Crown  Cover  – Peak Summer
axes[1,1].set_xlim(5, 100)      # Crown  Cover  – Late  Summer
# ────────────────────────────────────────────────────────────────

# Top-left: Canopy Height – Peak Summer
sns.kdeplot(data=ch_PS, x="true",      color="#CBC705", label="Measured",
            linewidth=2, ax=axes[0,0])
sns.kdeplot(data=ch_PS, x="pred", color="#327686", label="Predicted",
            linewidth=2, ax=axes[0,0])
axes[0,0].set_title('Canopy Height – Peak Summer')
axes[0,0].set_xlabel('Canopy Height (m)')
axes[0,0].legend()

# Top-right: Crown Cover – Peak Summer
sns.kdeplot(data=cc_PS, x="true",      color="#CBC705", label="Measured",
            linewidth=2, ax=axes[0,1])
sns.kdeplot(data=cc_PS, x="pred", color="#327686", label="Predicted",
            linewidth=2, ax=axes[0,1])
axes[0,1].set_title('Crown Cover – Peak Summer')
axes[0,1].set_xlabel('Crown Cover (%)')
axes[0,1].legend()

# Bottom-left: Canopy Height – Late Summer
sns.kdeplot(data=ch_LS, x="true",      color="#CBC705", label="Measured",
            linewidth=2, ax=axes[1,0])
sns.kdeplot(data=ch_LS, x="pred", color="#327686", label="Predicted",
            linewidth=2, ax=axes[1,0])
axes[1,0].set_title('Canopy Height – Late Summer')
axes[1,0].set_xlabel('Canopy Height (m)')
axes[1,0].legend()

# Bottom-right: Crown Cover – Late Summer
sns.kdeplot(data=cc_LS, x="true",      color="#CBC705", label="Measured",
            linewidth=2, ax=axes[1,1])
sns.kdeplot(data=cc_LS, x="pred", color="#327686", label="Predicted",
            linewidth=2, ax=axes[1,1])
axes[1,1].set_title('Crown Cover – Late Summer')
axes[1,1].set_xlabel('Crown Cover (%)')
axes[1,1].legend()

plt.tight_layout()
plt.show()


######## above vs HLS crown cov
plt.figure(figsize=(8, 6))
sns.kdeplot(data=data, x="crown_cov", color="#CBC705", label="HLS Crown Cover", linewidth=2)
sns.kdeplot(data=data, x="ABoVE_tcc", color="#327686", label="ABoVE TCC", linewidth=2)

# Add labels and legend
plt.xlabel("Crown Cover (%)")
plt.ylabel("Probability density")
plt.title("Distribution of Crown Cover Values")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

### frequency

plt.figure(figsize=(8, 6))

# Plot histograms with count instead of density
sns.histplot(data=data, x="crown_cov", color="#CBC705", label="HLS Crown Cover",
             bins=30, kde=False, stat="count", element="step", fill=False, linewidth=2)

sns.histplot(data=data, x="ABoVE_tcc", color="#327686", label="ABoVE TCC",
             bins=30, kde=False, stat="count", element="step", fill=False, linewidth=2)

# Customize plot
plt.xlabel("Crown Cover (%)")
plt.ylabel("Frequency")
plt.title("Frequency, of Crown Cover Values")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
