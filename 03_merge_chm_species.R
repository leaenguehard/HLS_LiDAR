##### merge tree species and chm #######
# 07.03.2025 lea enguehard
setwd("//dmawi/potsdam/data/bioing/user/lenguehard/project3/R")

library(lidR)
library(raster)
library(sf)
library(stringr)


###########For one plot#####

las <- readLAS("resultS/tree_species/EN24110_treesonly.laz")
raster <- stack("data/stack/EN24110_stack.tif")

# Ensure the LAS file contains the Tree ID and Species classification
if (!("Tree" %in% colnames(las@data)) || !("Species" %in% colnames(las@data))) {
  stop("LAS file must contain 'Tree' and 'Species' attributes.")
}

las_sf <- st_as_sf(data.frame(x = las@data$X, 
                              y = las@data$Y, 
                              Species = las@data$Species), 
                   coords = c("x", "y"), crs = projection(raster))

# Rasterize the Species information to match the CHM
species_raster <- rasterize(las_sf, raster, field = "Species", fun = modal, na.rm = TRUE)

# Stack the CHM and Species rasters together
chm_species_stack <- stack(raster, species_raster)

# Save the new raster with 3 bands
output_path <- "data/stack_sp/EN24110_chm_gf_sp.tif"
writeRaster(chm_species_stack, output_path, format = "GTiff", overwrite = TRUE)

print(paste("Saved:", output_path))
nlayers(chm_species_stack)

##############################

## Loop around all plots

las_path <- "results/tree_species/"
stack_path <- "data/stack/"
out_folder <-"data/stack_sp/"

chm_files <- list.files(stack_path, pattern = "_stack.tif$", full.names = TRUE)
las_files <- list.files(las_path, pattern = "_treesonly.laz$", full.names = TRUE)

extract_id <- function(filename, pattern) {
  basename(filename) |> 
    str_remove(pattern)  # Remove the pattern suffix (_CHM.tif or _CHM_gf1.tif)
}

# Extract unique IDs
chm_ids <- sapply(chm_files, extract_id, pattern = "_stack.tif$")
las_ids <- sapply(las_files, extract_id, pattern = "_treesonly.laz$")

# Match CHM and GF files
matched_ids <- intersect(chm_ids, las_ids)

for (plot in matched_ids) {
  
  # Get file paths
  chm_path <- chm_files[which(chm_ids == plot)]
  las_path <- las_files[which(las_ids == plot)]
  
  # Load rasters and las
  raster <- stack(chm_path)
  las <- readLAS(las_path)
  
  
  # Ensure the LAS file contains the Tree ID and Species classification
  if (!("Tree" %in% colnames(las@data)) || !("Species" %in% colnames(las@data))) {
    stop("LAS file must contain 'Tree' and 'Species' attributes.")
  }
  
  
  las_sf <- st_as_sf(data.frame(x = las@data$X, 
                                y = las@data$Y, 
                                Species = las@data$Species), 
                     coords = c("x", "y"), crs = projection(raster))
  
  # Rasterize the Species information to match the CHM
  species_raster <- rasterize(las_sf, raster, field = "Species", fun = modal, na.rm = TRUE)
  
  # Stack the CHM and Species rasters together
  chm_species_stack <- stack(raster, species_raster)
  
   # Define output file name
   output_path <- file.path(out_folder, paste0(plot, "_chm_gf_sp.tif"))
    
    # Save stacked raster
    writeRaster(chm_species_stack, output_path, format = "GTiff", overwrite = TRUE)
    
    print(paste("Saved:", output_path))

}


##Plot species
library(RColorBrewer)

# Extract Band 3 (Species)
species_raster <- chm_species_stack[[3]]  # Get the third band

# Generate a color palette with enough unique colors
num_classes <- length(unique(values(species_raster)))  # Number of unique species
num_classes <- num_classes[!is.na(num_classes)]  # Remove NA values

# Use a color palette (Set3 has up to 12 colors, extend if needed)
palette_colors <- colorRampPalette(brewer.pal(min(12, num_classes), "Set3"))(num_classes)

# Plot only Band 3 with distinct colors
plot(species_raster, col = palette_colors, main = "Species Classification (Band 3)")











