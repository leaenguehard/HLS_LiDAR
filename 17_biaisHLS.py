# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:19:05 2025

@author: leengu001
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("results//pixel_table/pixel_table_with_above.csv")
data2 = pd.read_csv("results//Regression_models/BlockCV_4models100m/predictions_csv/crown_cov_PS_predictions.csv")
data["crown_cov_PS_"] = data2["Predicted"]
data2["bias"] = data2["Predicted"] - data2["True"]
data["bias"] = data2["bias"]


#  Bias Plot
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

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=data,
    x="crown_cov",
    y="bias",
    hue="cat",                     
    palette=palette_dict,         
    alpha=0.6,
    edgecolor=None
)
plt.axhline(0, color="black", linestyle="--")
plt.title("Bias")
plt.xlabel("HLS forest structure crown cover (%)")
plt.ylabel("Bias (Predicted - True HLS Forest structure)")
plt.legend(title="Category", loc="best")  
plt.tight_layout()
plt.show()


# Map bias HLS forest structure – MEDIAN BIAS PER SITE
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm      # NEW
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

site_df  = pd.read_csv("data/site/sites_dup.csv")

treeline = (gpd.read_file("data/site/treeline_la.shp")
              .to_crs(epsg=4326))

data["site_id"] = data["UAV_tiff"].str.replace("_final.tif", "", regex=False)

# Median bias per site
site_bias = (
    data.groupby("site_id")
         .agg(median_bias=("bias", "median"))
         .reset_index()
)

# Merge with site coordinates
site_summary = pd.merge(site_df, site_bias,
                        left_on="plot", right_on="site_id", how="inner")

#color ramp
colors = ["blue", "white", "red"]
cmap   = LinearSegmentedColormap.from_list("custom_bwr", colors, N=256)

# Fixed range to match the second plot (−25 to +35) and centre at 0
norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=35)

# plot
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(
    1, 1, 1,
    projection=ccrs.NorthPolarStereo(central_longitude=-150)
)
ax.set_extent([-168, -121, 55, 75], crs=ccrs.PlateCarree())

# Base map
ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=":")

# Treeline
ax.add_geometries(
    treeline.geometry,
    crs=ccrs.PlateCarree(),
    edgecolor="darkgreen",
    facecolor="none",
    linewidth=0.5,
    zorder=1
)

# Gridlines
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.3,
    color="gray",
    alpha=0.35,
    x_inline=False,
    y_inline=False
)
gl.top_labels, gl.right_labels = False, False
gl.bottom_labels, gl.left_labels = True, True
gl.xlocator = mticker.FixedLocator([-160, -144, -128])
gl.ylocator = mticker.FixedLocator([60, 70])
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabel_style = dict(size=12, color="black",
                       rotation=20, ha="center", rotation_mode="anchor")
gl.ylabel_style = dict(size=12, color="black",
                       rotation=20, va="center")

sc = ax.scatter(
    site_summary["lon"],
    site_summary["lat"],
    c=site_summary["median_bias"],
    cmap=cmap,
    norm=norm,               
    s=60,
    edgecolors="k",
    alpha=0.9,
    transform=ccrs.PlateCarree()
)

# Colour-bar
cb = plt.colorbar(sc, ax=ax,
                  orientation="vertical", shrink=0.6, pad=0.02)
cb.set_label("Median Bias (Predicted – True HLS Forest structure)", fontsize=12)
cb.ax.set_yticks([-25, -12.5, 0, 12.5, 25, 35])   # match second plot ticks

treeline_handle = Line2D([0], [0], color="darkgreen",
                         linewidth=0.5, label="Treeline")
# ax.legend(handles=[treeline_handle], loc="upper right")

# Country labels
ax.text(-158, 67.5, "Alaska",
        transform=ccrs.PlateCarree(),
        fontsize=14, fontstyle="italic", fontweight="bold")
ax.text(-132, 66.5, "Canada",
        transform=ccrs.PlateCarree(),
        fontsize=14, fontstyle="italic", fontweight="bold")

# Scale bar
fontprops = fm.FontProperties(size=12, weight="bold")
scalebar  = AnchoredSizeBar(
    ax.transData, size=400_000, label="400 km", loc="lower left",
    pad=0.4, borderpad=0.5, sep=5, frameon=False,
    size_vertical=20, fontproperties=fontprops
)
ax.add_artist(scalebar)

plt.savefig("results/figures/HLS_median_bias_100m.svg",
            format="svg", bbox_inches="tight")
plt.tight_layout()
plt.show()
