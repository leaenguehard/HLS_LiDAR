# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:24:03 2025
Tasseled Cap coefficients
@author: leengu001
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("results//pixel_table/pixel_table_final_filt.csv")

# # Tasseled Cap Coefficients
# coefficients = {
#     'TCB': [0.3690, 0.4271, 0.4689, 0.5073, 0.3824, 0.2406],   # Brightness
#     'TCG': [-0.2870, -0.2685, -0.4087, 0.8145, 0.0637, -0.1052], # Greenness
#     'TCW': [0.0382, 0.2137, 0.3536, 0.2270, -0.6108, -0.6351]    # Wetness
# }

# # Compute tasseled cap
# def compute_tc(df, bands, coeffs):
#     return sum(df[band] * coef for band, coef in zip(bands, coeffs))

# # Band names
# bands_PS = ['blue_PS', 'green_PS', 'red_PS', 'NIR_PS', 'SWIR1_PS', 'SWIR2_PS']
# bands_LS = ['blue_LS', 'green_LS', 'red_LS', 'NIR_LS', 'SWIR1_LS', 'SWIR2_LS']

# # Compute TCB
# data['TCB_PS'] = compute_tc(data, bands_PS, coefficients['TCB'])
# data['TCB_LS'] = compute_tc(data, bands_LS, coefficients['TCB'])

# # Compute TCW
# data['TCW_PS'] = compute_tc(data, bands_PS, coefficients['TCW'])
# data['TCW_LS'] = compute_tc(data, bands_LS, coefficients['TCW'])

# # Compute TCG (for TCA only)
# data['TCG_PS']  = compute_tc(data, bands_PS, coefficients['TCG'])
# data['TCG_LS']  = compute_tc(data, bands_LS, coefficients['TCG'])

# # Compute TCA
# data['TCA_PS'] = np.arctan(data['TCB_PS'] / data['TCG_PS'] )
# data['TCA_LS'] = np.arctan(data['TCB_LS'] / data['TCG_LS'] )

# # Save the updated DataFrame to a new CSV file
# data.to_csv("results//pixel_table/pixel_table_final_filtTC.csv", index=False, encoding='utf-8-sig')

##### PLOT
custom_palette = [
    "#FC8D62", "#8DA0CB", "#E78AC3", "#66C2A5", "#FFD92F", "#A6D854"
]
categories = ['5–20%', '20–50%', '50–80% >5', '50–80% <5', '80–100% <5', '80–100% >5']
palette_dict = dict(zip(categories, custom_palette))

# Compute axis limits for top row (TCB vs TCG)
x_top = pd.concat([data['TCB_PS'], data['TCB_LS']])
y_top = pd.concat([data['TCG_PS'], data['TCG_LS']])
x_top_lim = [x_top.min(), x_top.max()]
y_top_lim = [y_top.min(), y_top.max()]

# Compute axis limits for bottom row (TCG vs TCW) with padding
x_bot = pd.concat([data['TCG_PS'], data['TCG_LS']])
y_bot = pd.concat([data['TCW_PS'], data['TCW_LS']])

# Padding: 5% of data range
x_bot_range = x_bot.max() - x_bot.min()
y_bot_range = y_bot.max() - y_bot.min()
pad_ratio = 0.05

x_bot_lim = [x_bot.min() - pad_ratio * x_bot_range, x_bot.max() + pad_ratio * x_bot_range]
y_bot_lim = [y_bot.min() - pad_ratio * y_bot_range, y_bot.max() + pad_ratio * y_bot_range]

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
sns.set(style="ticks")

ax00, ax01, ax10, ax11 = axes.flatten()

# Top left: Peak Summer (TCB vs TCG)
sns.scatterplot(
    data=data,
    x='TCB_PS',
    y='TCG_PS',
    hue='cat',
    palette=palette_dict,
    alpha=0.7,
    s=40,
    edgecolor='k',
    ax=ax00,
    legend=False
)
ax00.set_title("Peak Summer", fontsize=22)
ax00.set_xlabel("Brightness (TCB)", fontsize=20)
ax00.set_ylabel("Greenness (TCG)", fontsize=20)
ax00.set_xlim(x_top_lim)
ax00.set_ylim(y_top_lim)

# Top right: Late Summer (TCB vs TCG)
sns.scatterplot(
    data=data,
    x='TCB_LS',
    y='TCG_LS',
    hue='cat',
    palette=palette_dict,
    alpha=0.7,
    s=40,
    edgecolor='k',
    ax=ax01,
    legend=True
)
ax01.set_title("Late Summer", fontsize=22)
ax01.set_xlabel("Brightness (TCB)", fontsize=20)
ax01.set_ylabel("")
ax01.set_xlim(x_top_lim)
ax01.set_ylim(y_top_lim)
legend = ax01.legend(
    title="Category",
    loc='upper right',
    frameon=True
)

# Increase legend font size
legend.get_title().set_fontsize(18)  # Title font size
for text in legend.get_texts():
    text.set_fontsize(16)  # Label font size
    
# Bottom left: Peak Summer (TCG vs TCW)
sns.scatterplot(
    data=data,
    x='TCG_PS',
    y='TCW_PS',
    hue='cat',
    palette=palette_dict,
    alpha=0.7,
    s=40,
    edgecolor='k',
    ax=ax10,
    legend=False
)
ax10.set_xlabel("Greenness (TCG)", fontsize=20)
ax10.set_ylabel("Wetness (TCW)", fontsize=20)
ax10.set_xlim(x_bot_lim)
ax10.set_ylim(y_bot_lim)

# Bottom right: Late Summer (TCG vs TCW)
sns.scatterplot(
    data=data,
    x='TCG_LS',
    y='TCW_LS',
    hue='cat',
    palette=palette_dict,
    alpha=0.7,
    s=40,
    edgecolor='k',
    ax=ax11,
    legend=False
)
ax11.set_xlabel("Greenness (TCG)", fontsize=20)
ax11.set_ylabel("")
ax11.set_xlim(x_bot_lim)
ax11.set_ylim(y_bot_lim)

for ax in axes.flatten():
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.tick_params(axis='both', which='major', labelsize=18)

plt.savefig("results/figures/TC.svg", format="svg", bbox_inches="tight")
plt.tight_layout()
plt.show()




######### BOXPLOT

# Melt the data into long format for seaborn
plot_data = pd.melt(
    data,
    id_vars='cat',
    value_vars=['TCB_PS', 'TCG_PS', 'TCW_PS'],
    var_name='Index',
    value_name='Value'
)



# Rename index labels for clarity
plot_data['Index'] = plot_data['Index'].str.replace('_PS', '').str.replace('TC', 'TC ')

# Create the grouped boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=plot_data, x='cat', y='Value', hue='Index')

plt.title("Tasseled Cap Indices by Category (Peak Summer)")
plt.xlabel("Category")
plt.ylabel("Index Value")
plt.xticks(rotation=45)
plt.legend(title="Tasseled Cap Index")
plt.tight_layout()
plt.show()