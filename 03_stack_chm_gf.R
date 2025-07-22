##############Stack raster##########
### Lea 06 03 2025
setwd("//dmawi/potsdam/data/bioing/user/lenguehard/project3")

library(raster)
library(stringr)

# 
# chm <- raster("data/CHM/EN22002_CHM.tif")
# gf <- raster("data/crown_cov/EN22002_CHM_gf1.tif")
# 
# stack <- stack(chm, gf)
# 

chm_folder <- "data/CHM"
gf_folder <- "data/crown_cov"
stack_folder <- "data/stack"

chm_files <- list.files(chm_folder, pattern = "_CHM.tif$", full.names = TRUE)
gf_files <- list.files(gf_folder, pattern = "_CHM_gf1.tif$", full.names = TRUE)

# Function to extract the unique ID
extract_id <- function(filename, pattern) {
  basename(filename) |> 
    str_remove(pattern)  # Remove the pattern suffix (_CHM.tif or _CHM_gf1.tif)
}

# Extract unique IDs
chm_ids <- sapply(chm_files, extract_id, pattern = "_CHM.tif$")
gf_ids <- sapply(gf_files, extract_id, pattern = "_CHM_gf1.tif$")

# Match CHM and GF files
matched_ids <- intersect(chm_ids, gf_ids)

# Loop through matched IDs and process
for (uid in matched_ids) {
  
  # Get file paths
  chm_path <- chm_files[which(chm_ids == uid)]
  gf_path <- gf_files[which(gf_ids == uid)]
  
  # Load rasters
  chm_raster <- raster(chm_path)
  gf_raster <- raster(gf_path)
  
  # Check alignment before stacking
  if (compareRaster(chm_raster, gf_raster)) {
    # Stack rasters
    raster_stack <- stack(chm_raster, gf_raster)
    
    # Define output file name
    output_path <- file.path(stack_folder, paste0(uid, "_stack.tif"))
    
    # Save stacked raster
    writeRaster(raster_stack, output_path, format = "GTiff", overwrite = TRUE)
    
    print(paste("Saved:", output_path))
  } else {
    print(paste("Skipping", uid, "due to misalignment."))
  }
}

print("Processing complete.")
