# -*- coding: utf-8 -*-
"""
Created on Mon May 19 11:28:01 2025

@author: leengu001
"""
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.coords import BoundingBox
from shapely.geometry import Point, box
import os
from tqdm import tqdm

csv_path = "results/pixel_table/pixel_table_final_filt.csv"
above_dir = "data/above/clip"
output_csv_path = "results/pixel_table/pixel_table_with_above.csv"

# Load pixel table
df = pd.read_csv(csv_path)

# Load ABoVE rasters
above_rasters = [os.path.join(above_dir, f) for f in os.listdir(above_dir) if f.endswith(".tif")]

df["ABoVE_tcc"] = pd.NA

above_index = []
for path in above_rasters:
    with rasterio.open(path) as src:
        bounds = box(*src.bounds)
        above_index.append((path, bounds))

# Loop over each row to extract ABoVE band value
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Adding ABoVE band"):
    point = Point(row["longitude"], row["latitude"])
    
    for path, bounds in above_index:
        if bounds.contains(point):
            try:
                with rasterio.open(path) as src:
                    row_idx, col_idx = rowcol(src.transform, row["longitude"], row["latitude"])
                    
                    # Check if within bounds
                    if (0 <= row_idx < src.height) and (0 <= col_idx < src.width):
                        value = src.read(1)[row_idx, col_idx]
                        if src.nodata is not None and value == src.nodata:
                            value = pd.NA
                        df.at[idx, "ABoVE_tcc"] = value
                        break
            except Exception as e:
                print(f" Error reading {path} for row {idx}: {e}")
                continue

# Save updated table
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ Updated pixel table saved: {output_csv_path}")
