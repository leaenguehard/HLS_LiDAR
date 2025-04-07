# -*- coding: utf-8 -*-
"""
Updated on Mar 24, 2025 — per-site filtered median export
"""

import ee
import geemap
import os

ee.Authenticate()
ee.Initialize(project='phd-project-435908')

shapefile_dir = r"\\dmawi\potsdam\data\bioing\user\lenguehard\project3\data\shp"
export_base_dir = r"\\dmawi\potsdam\data\bioing\user\lenguehard\project3\data\satellite\peak_summer"
output_dir = r"\\dmawi\potsdam\data\bioing\user\lenguehard\project3\data\satellite\aggregated_PS"

shapefile_list = [os.path.join(shapefile_dir, f) for f in os.listdir(shapefile_dir) if f.endswith("_CHM_extent.shp")]

bands_s2 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
bands_ls = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
rename_dict = {'B5': 'B8', 'B6': 'B11', 'B7': 'B12'}

def rename_landsat(image):
    return image.select(bands_ls).rename([rename_dict.get(b, b) for b in bands_ls])

def mask_hls_fmask(image):
    fmask = image.select('Fmask')
    cirrus_mask = fmask.bitwiseAnd(1 << 0).eq(0)
    cloud_mask = fmask.bitwiseAnd(1 << 1).eq(0)
    snow_mask = fmask.bitwiseAnd(1 << 4).eq(0)
    water_mask = fmask.bitwiseAnd(1 << 5).eq(0)
    return image.updateMask(cirrus_mask.And(cloud_mask).And(snow_mask).And(water_mask))

# UPDATED: Now returns image with NDVI band kept
def mask_invalid_pixels(image):
    nir = image.select('B8')
    red = image.select('B4')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')

    valid_bands = image.gt(0).reduce(ee.Reducer.min())
    nir_valid = nir.gte(0.02).And(nir.lte(0.6))
    ndvi_valid = ndvi.gte(0.3)

    combined_mask = valid_bands.And(nir_valid).And(ndvi_valid)
    return image.addBands(ndvi).updateMask(combined_mask)

# Loop through shapefiles
for shapefile_path in shapefile_list:
    shapefile_fc = geemap.shp_to_ee(shapefile_path)
    feature = shapefile_fc.first()
    roi = feature.geometry()

    hlss30 = (
        ee.ImageCollection("NASA/HLS/HLSS30/v002")
        .filterBounds(roi)
        .filterDate('2022-06-20', '2024-08-10').filter(ee.Filter.dayOfYear(171, 222))
       # .filterDate('2022-08-19', '2024-09-30').filter(ee.Filter.dayOfYear(232, 273))   
        .filter(ee.Filter.lt('CLOUD_COVERAGE', 30))
        .map(mask_hls_fmask)
        .select(bands_s2)
    )

    hlsl30 = (
        ee.ImageCollection("NASA/HLS/HLSL30/v002")
        .filterBounds(roi)
        .filterDate('2022-06-20', '2024-08-10').filter(ee.Filter.dayOfYear(171, 222))
       # .filterDate('2022-08-19', '2024-09-30').filter(ee.Filter.dayOfYear(232, 273))        
        .filter(ee.Filter.lt('CLOUD_COVERAGE', 30))
        .map(mask_hls_fmask)
        .map(rename_landsat)
        .select(bands_s2)
    )

    merged_hls = hlss30.merge(hlsl30)

    # Apply filtering + add NDVI
    filtered = merged_hls.map(mask_invalid_pixels)

    # Use Reducer.median to preserve band names like "B8_median", "NDVI_median"    
    median_image = filtered.reduce(ee.Reducer.median())

    #recover and apply the proper mask to the median
    mask_reference = filtered.map(lambda img: img.mask().reduce(ee.Reducer.min()))
    combined_mask = mask_reference.reduce(ee.Reducer.max())
    median_image_masked = median_image.updateMask(combined_mask)

    shapefile_name = os.path.basename(shapefile_path).replace("_CHM_extent.shp", "")
    export_dir = os.path.join(output_dir, shapefile_name)
    os.makedirs(export_dir, exist_ok=True)

    #  Export image (includes median of NDVI)
    geemap.ee_export_image(
        median_image_masked,
        filename=os.path.join(export_dir, shapefile_name + "_median_filtered_PS.tif"),
        scale=30,
        region=roi.bounds()
    )

    print(f"✅ Exported: {shapefile_name}")

print("All median images with NDVI exported successfully!")
