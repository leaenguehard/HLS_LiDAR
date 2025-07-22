# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:53:27 2025

@author: leengu001
"""
##################ecoregions
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:53:27 2025

@author: leengu001
"""
################## ecoregions
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.lines import Line2D
from shapely.geometry import box
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# Load site data and treeline
df = pd.read_csv("data/site/sites_dup.csv")
treeline = gpd.read_file("data/site/treeline_la.shp")
treeline = treeline.to_crs(epsg=4326)

# Load and clip ecoregion shapefile
ecoreg = gpd.read_file("data/na_cec_eco_l2/NA_CEC_Eco_Level2.shp")
ecoreg = ecoreg.to_crs(epsg=4326)

# Clip to study area
study_box = box(-168, 55, -121, 75)
ecoreg_clipped = ecoreg[ecoreg.intersects(study_box)].copy()

# Generate unique colors for each ecoregion name
unique_ecos = sorted(ecoreg_clipped["NA_L2NAME"].unique())
colormap = plt.cm.get_cmap("tab20", len(unique_ecos))
eco_colors = {eco: colormap(i) for i, eco in enumerate(unique_ecos)}
ecoreg_clipped["color"] = ecoreg_clipped["NA_L2NAME"].map(eco_colors)

# Setup figure and projection
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.NorthPolarStereo(central_longitude=-150))
ax.set_extent([-168, -121, 55, 75], crs=ccrs.PlateCarree())

# Plot clipped ecoregions with color
for _, row in ecoreg_clipped.iterrows():
    ax.add_geometries(
        [row.geometry],
        crs=ccrs.PlateCarree(),
        facecolor=row["color"],
        edgecolor="black",
        linewidth=0.2,
        alpha=0.75,
        zorder=0
    )

# Add geographic features
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=":")
ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="none")

# Add treeline
ax.add_geometries(
    treeline.geometry,
    crs=ccrs.PlateCarree(),
    edgecolor="darkgreen",
    facecolor="none",
    linewidth=0.9,
    zorder=1
)

# Gridlines
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.3, color="gray", alpha=0.65,
    x_inline=False, y_inline=False
)
gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.xlocator = mticker.FixedLocator([-160, -144, -128])
gl.ylocator = mticker.FixedLocator([60, 70])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {"size": 12, "color": "black", "rotation": 20, "ha": "center", "rotation_mode": "anchor"}
gl.ylabel_style = {"size": 12, "color": "black", "rotation": 20, "va": "center"}

# Unique years and color map
years = sorted(df["year"].unique())
point_colors = ["#5B33AA", "#E5E327", "#51A68A"]

# Plot UAV-LiDAR points
for i, yr in enumerate(years):
    sel = df["year"] == yr
    ax.scatter(
        df.loc[sel, "lon"],
        df.loc[sel, "lat"],
        s=50,
        color=point_colors[i],
        edgecolor="k",
        alpha=1,
        transform=ccrs.PlateCarree(),
        label=str(yr)
    )

# Legend: UAV years
point_handles = [
    Line2D([], [], marker="o", color=point_colors[i], markeredgecolor="k", linestyle="None", markersize=6)
    for i in range(len(years))
]
treeline_handle = Line2D([0], [0], color="darkgreen", linewidth=0.9)
handles = point_handles + [treeline_handle]
labels = [str(yr) for yr in years] + ["Treeline"]

# Legend: Ecoregions
eco_patches = [
    mpatches.Patch(color=eco_colors[name], label=name)
    for name in unique_ecos
]

# Combine legends (fixed to show year labels)
first_legend = ax.legend(handles=handles, labels=labels, title="Year of UAV-LiDAR acquisition", title_fontsize=11, loc="upper right", fontsize=11)
ax.add_artist(first_legend)
second_legend = ax.legend(handles=eco_patches, title="Ecoregions (L2)", title_fontsize=9.5, loc="lower right", fontsize=9.5, frameon=True, ncol=1)
ax.add_artist(second_legend)

# Labels for regions
ax.text(-158, 67.5, "Alaska",
        transform=ccrs.PlateCarree(),
        fontsize=16, fontstyle="italic", fontweight="bold")
ax.text(-132, 66.5, "Canada",
        transform=ccrs.PlateCarree(),
        fontsize=16, fontstyle="italic", fontweight="bold")

# Scale bar
fontprops = fm.FontProperties(size=14, weight="bold")
scalebar = AnchoredSizeBar(
    ax.transData,
    size=400_000,      # 400 km
    label="400 km",
    loc="lower left",
    pad=0.4,
    borderpad=0.5,
    sep=5,
    frameon=False,
    size_vertical=20,
    fontproperties=fontprops
)
ax.add_artist(scalebar)

# Layout & Save
plt.tight_layout()
#plt.savefig("results/figures/Study_sites_with_ecoregions.svg", format="svg", bbox_inches="tight")
plt.show()
