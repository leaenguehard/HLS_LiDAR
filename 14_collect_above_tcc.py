# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:29:57 2025

@author: leengu001
"""
import os
import re
import requests
from urllib.parse import urljoin
from urllib.request import urlretrieve

######### Download data 

# Base URL for the dataset
base_url = "https://daac.ornl.gov/orders/d341af8213610a8561a12dae15b6affd/Boreal_CanopyCover_StandAge/data/" ###Valid for 30 days

# Local folder to save files
local_folder = r"\\dmawi\potsdam\data\bioing\user\lenguehard\project3\data\above"
os.makedirs(local_folder, exist_ok=True)

# Coordinate boundaries
min_lon, max_lon = -165, -121
min_lat, max_lat = 58, 68

# Get list of all files in the directory
response = requests.get(base_url)
file_links = re.findall(r'href="([^"]+\.tif)"', response.text)

for filename in file_links:
    match = re.match(r'(\d{3})([EW])_(\d{2})([NS])_.*\.tif', filename)
    if match:
        lon_deg = int(match.group(1))
        lon_dir = match.group(2)
        lat_deg = int(match.group(3))
        lat_dir = match.group(4)

        # Convert to signed coordinates
        lon = lon_deg if lon_dir == 'E' else -lon_deg
        lat = lat_deg if lat_dir == 'N' else -lat_deg

        # Check if file is within specified bounds
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            file_url = urljoin(base_url, filename)
            local_path = os.path.join(local_folder, filename)

            if not os.path.exists(local_path):  # Avoid re-downloading
                print(f"Downloading {filename}...")
                urlretrieve(file_url, local_path)

print("Download complete.")


#### Directly identify tiles and crop above
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import os
from glob import glob

your_rasters = glob(r"results\labeled_imagery\aggregated-final\*.tif")
above_tiles = glob(r"data\above\*.tif")
clip_output_dir = r"data\above\clip"
os.makedirs(clip_output_dir, exist_ok=True)

print(f"Found {len(your_rasters)} site rasters and {len(above_tiles)} ABoVE tiles.")

for your_file in your_rasters:
    with rasterio.open(your_file) as your_src:
        print(f"\n Processing site: {os.path.basename(your_file)}")
        your_bounds = your_src.bounds
        your_crs = your_src.crs
        print(f"   Bounds: {your_bounds}, CRS: {your_crs}")
        
        your_geom_shape = box(*your_bounds)
        your_geom = [your_geom_shape]

        found_overlap = False

        for above_file in above_tiles:
            with rasterio.open(above_file) as above_src:
                above_crs = above_src.crs

                if above_crs != your_crs:
                    print(f"    Skipping {os.path.basename(above_file)} due to CRS mismatch")
                    continue

                above_geom = box(*above_src.bounds)

                if above_geom.intersects(your_geom_shape):
                    found_overlap = True
                    try:
                        out_image, out_transform = mask(above_src, your_geom, crop=True)

                        # Prepare metadata and output path
                        out_meta = above_src.meta.copy()
                        out_meta.update({
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })

                        output_filename = f"{os.path.basename(your_file).replace('.tif', '')}_clipped_{os.path.basename(above_file)}"
                        output_path = os.path.join(clip_output_dir, output_filename)

                        with rasterio.open(output_path, "w", **out_meta) as dest:
                            dest.write(out_image)

                        print(f"    Clipped and saved: {output_filename}")
                    except Exception as e:
                        print(f"    Failed to clip {os.path.basename(above_file)}: {e}")

        if not found_overlap:
            print("    No overlapping ABoVE tiles found for this site.")

            
            
#### Export table and all HLS forest structre pixels with above data

