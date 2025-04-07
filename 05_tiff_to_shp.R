##### Filtering raster #######
# 13/03/2024 lea enguehard

setwd("//dmawi/potsdam/data/bioing/user/lenguehard/project3/R")

library(terra)
library(sf)


raster_folder <- "data/CHM/"
output_folder <- "data/shp/"


raster_files <- list.files(raster_folder, pattern = "\\.tif$", full.names = TRUE)

# Loop through each raster file
for (raster_file in raster_files) {
  # Load the raster using terra
  r <- rast(raster_file)
  
  # Get the extent and convert to a polygon
  extent_poly <- as.polygons(ext(r))
  
  # Set CRS (Coordinate Reference System)
  crs(extent_poly) <- crs(r)
  
  # Convert to sf object for saving
  extent_sf <- st_as_sf(extent_poly)
  
  # Define output filename
  output_shapefile <- file.path(output_folder, paste0(tools::file_path_sans_ext(basename(raster_file)), "_extent.shp"))
  
  # Save as a shapefile
  st_write(extent_sf, output_shapefile, delete_layer = TRUE)
  
  # Print progress
  cat("Saved extent polygon:", output_shapefile, "\n")
}

cat("All extent polygons have been saved.\n")