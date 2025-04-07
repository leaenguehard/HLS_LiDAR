##############Crown cover##########
### Lea 06 03 2025

setwd("//dmawi/potsdam/data/bioing/user/lenguehard/project3/R")

library(lidR)
library(raster)

# Load the Canopy Height Model (CHM)
chm_path <- "data/CHM/EN23603_CHM.tif"
chm_raster <- raster(chm_path)

threshold <- 1 #threshold for tree in meters

# Create a binary raster based on the threshold
gap_fraction <- calc(chm_raster, fun = function(x) { ifelse(x >= threshold, 1, 0) })

# Save the new raster to a GeoTIFF file
output_path <- "results/crown_cov/EN23603_gap_fraction.tif"
# writeRaster(gap_fraction, output_path, format="GTiff", overwrite=TRUE)

plot(gap_fraction, col = c("white", "black"), legend = FALSE)


### Loop around all plots
chm_folder <- "data/CHM/"
output_folder <- "results/crown_cov"

threshold <- 1

chm_files <- list.files(chm_folder, pattern = ".tif", full.names = TRUE)

# Loop through each CHM file
for (chm_path in chm_files) {
  chm_raster <- raster(chm_path)
  
  gap_fraction <- calc(chm_raster, fun = function(x) { ifelse(x >= threshold, 1, 0) })
  
  file_name <- basename(chm_path) 
  output_name <- gsub("\\.tif$", "_gf.tif", file_name) 
  output_path <- file.path(output_folder, output_name)
  
  writeRaster(gap_fraction, output_path, format = "GTiff", overwrite = TRUE)
    print(paste("Processed:", file_name, "->", output_name))
}

# Print completion message
print("All CHM files have been processed successfully.")