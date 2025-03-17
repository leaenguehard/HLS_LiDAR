# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:52:19 2025

@author: leengu001
"""

# C:\Users\leengu001\Documents\project3\python>venv\Scripts\activate   if necessary
import ee
import geemap
import os

ee.Authenticate()
ee.Initialize(project='phd-project-435908')

shapefile_dir = r"\\dmawi\potsdam\data\bioing\user\lenguehard\project3\R\data\shp"

# Get a list of all shapefiles ending with "_CHM_extent.shp"
shapefile_list = [os.path.join(shapefile_dir, f) for f in os.listdir(shapefile_dir) if f.endswith("_CHM_extent.shp")]

# Define output directory for the exported images
export_base_dir = r"\\dmawi\potsdam\data\bioing\user\lenguehard\project3\R\data\satellite\peak_summer"


# Define band names for both collections
bands_s2 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # Sentinel-based (HLSS30)
bands_ls = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']  # Landsat-based (HLSL30)

# Rename Landsat bands to match Sentinel-2
rename_dict = {'B5': 'B8', 'B6': 'B11', 'B7': 'B12'}

# Function to rename Landsat bands
def rename_landsat(image):
    return image.select(bands_ls).rename([rename_dict.get(b, b) for b in bands_ls])

# Function to mask clouds using Fmask
def mask_hls_fmask(image):
    fmask = image.select('Fmask')
    cirrus_mask = fmask.bitwiseAnd(1 << 0).eq(0)  # Cirrus Bit 0
    cloud_mask = fmask.bitwiseAnd(1 << 1).eq(0)  # Cloud Bit 1
    snow_mask = fmask.bitwiseAnd(1 << 4).eq(0)  # Snow Bit 4
    water_mask = fmask.bitwiseAnd(1 << 5).eq(0)  # Water Bit 5
    return image.updateMask(cirrus_mask.And(cloud_mask).And(snow_mask).And(water_mask))

for shapefile_path in shapefile_list:
    # Load shapefile as an Earth Engine FeatureCollection
    shapefile_fc = geemap.shp_to_ee(shapefile_path)

    # Extract individual feature (assuming each shapefile has one feature)
    feature = shapefile_fc.first()
    roi = feature.geometry()  # Extract geometry (Polygon)


    # Define HLS image collections, filtering to this shapefile area
    hlss30 = (
        ee.ImageCollection("NASA/HLS/HLSS30/v002")
        .filterBounds(roi)  
        .filterDate('2022-06-20', '2024-08-10').filter(ee.Filter.dayOfYear(171, 222))        
        .filter(ee.Filter.lt('CLOUD_COVERAGE', 30))
        .map(mask_hls_fmask)
        .select(bands_s2) 
    )

    hlsl30 = (
        ee.ImageCollection("NASA/HLS/HLSL30/v002")
        .filterBounds(roi)  
        .filterDate('2022-06-20', '2024-08-10').filter(ee.Filter.dayOfYear(171, 222))        
        .filter(ee.Filter.lt('CLOUD_COVERAGE', 30))
        .map(mask_hls_fmask)
        .map(rename_landsat)
        .select(bands_s2)  # Now they have the same band names  
    )

    # Merge collections for this specific feature
    merged_hls = hlss30.merge(hlsl30)
    
    shapefile_name = os.path.basename(shapefile_path).replace("_CHM_extent.shp", "")
    
    export_dir = os.path.join(export_base_dir, shapefile_name)
    os.makedirs(export_dir, exist_ok=True)  

    collection = merged_hls.select(bands_s2)
    print(collection.aggregate_array("system:index").getInfo())
    
    geemap.ee_export_image_collection(
        collection,
        out_dir=export_dir,
        scale=30,
        region=roi.bounds(),
        file_per_band = False
    )

    print(f"Exported: {export_dir}")

print("All tiles have been exported successfully!")



























