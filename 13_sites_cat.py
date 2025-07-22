# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:40:49 2025
Pie chart by category per plot
@author: leengu001
"""
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Load data
data = pd.read_csv("results//pixel_table/pixel_table_final_filt.csv")
df = pd.read_csv("data/site/sites_dup.csv")
treeline = gpd.read_file("data/site/treeline_la.shp").to_crs(epsg=4326)

# Extract plot ID from UAV_tiff filename
data["plot"] = data["UAV_tiff"].str.extract(r"(EN\d+[^_]*)")[0]

# Custom category palette
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

# Prepare pie data
grouped = data.groupby("plot")["cat"].value_counts(normalize=True).unstack(fill_value=0)
loc_df = df[["plot", "lon", "lat"]].drop_duplicates()
grouped = grouped.merge(loc_df, on="plot", how="left")

# Set up the figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=-150))
ax.set_extent([-170, -120, 55, 75], crs=ccrs.PlateCarree())  # Adjusted extent

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
    linewidth=0.3, color="gray", alpha=0.35,
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

# Add pie charts at each plot location
for _, row in grouped.iterrows():
    lon, lat = row["lon"], row["lat"]

    # Skip invalid coordinates or out-of-map points
    if (
        pd.isnull(lon) or pd.isnull(lat)
        or not (-180 <= lon <= 180)
        or not (-90 <= lat <= 90)
        or not (-170 <= lon <= -120)
        or not (55 <= lat <= 75)
    ):
        continue

    sizes = [row.get(cat, 0) for cat in categories]
    if sum(sizes) == 0:
        continue

    # Enlarged pie chart size
    axins = inset_axes(ax,
                       width=0.3,
                       height=0.3,
                       loc='center',
                       bbox_to_anchor=(lon, lat),
                       bbox_transform=ccrs.PlateCarree()._as_mpl_transform(ax),
                       borderpad=0)

    axins.pie(
        sizes,
        colors=[palette_dict[cat] for cat in categories],
        wedgeprops={'linewidth': 0.2, 'edgecolor': 'k'}
    )
    axins.set_aspect("equal")
    axins.axis("off")

# Add country labels
ax.text(-158, 67.5, "Alaska",
        transform=ccrs.PlateCarree(),
        fontsize=14, fontstyle="italic", fontweight="bold")
ax.text(-132, 66.5, "Canada",
        transform=ccrs.PlateCarree(),
        fontsize=14, fontstyle="italic", fontweight="bold")

# Scale bar
fontprops = fm.FontProperties(size=12, weight="bold")
scalebar = AnchoredSizeBar(
    ax.transData,
    size=400_000,      # 400 km in metres
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

# Legend for categories
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette_dict[cat],
           markersize=10, label=cat) for cat in categories
]
legend_elements.append(
    Line2D([0], [0], color="darkgreen", linewidth=1, label="Treeline")
)

ax.legend(handles=legend_elements, title="Category", loc="upper right")

#plt.savefig("results/figures/Pies_study_sites.svg", format="svg", bbox_inches="tight")
plt.tight_layout()
plt.show()


