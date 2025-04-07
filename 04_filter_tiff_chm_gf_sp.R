##### Filtering raster #######
# 11/03/2024 lea enguehard


setwd("//dmawi/potsdam/data/bioing/user/lenguehard/project3/R")

library(lidR)
library(raster)
library(sf)
library(stringr)

rast_path <- "data/stack_sp/"
out_folder <- "data/raster_final/"

raster_files <- list.files(rast_path, pattern = "_chm_gf_sp.tif$", full.names = TRUE)
extract_id <- function(filename, pattern) {
  basename(filename) |> 
    str_remove(pattern)  # Remove the pattern suffix (_CHM.tif or _CHM_gf1.tif)
}

rast_ids <- sapply(raster_files, extract_id, pattern = "_chm_gf_sp.tif$")

#nlayers(raster_stack)

for (i in seq_along(rast_ids)){
  
  raster_stack <- stack(raster_files[i])
  
  band1 <- raster_stack[[1]]
  band2 <- raster_stack[[2]]
  band3 <- raster_stack[[3]]
  
  band1[band1 < 0] <- NA # Set negative values in CHM to nodata
  band3[band1 < 1] <- NA# Filter tree species to nodata where CHM < 1m
  raster_stack <- stack(band1, band2, band3)
  
  
  output_path <- file.path(out_folder, paste0(rast_ids[i], "_final.tif"))
  
  writeRaster(raster_stack, output_path, format = "GTiff", overwrite = TRUE)
  
  print(paste("Saved:", output_path))
  
  
}

     
     






########### Plot species #############
library(RColorBrewer)

# Extract Band 3 (Species)
species_raster <- raster_stack[[3]]  # Get the third band

# Generate a color palette with enough unique colors
num_classes <- length(unique(values(species_raster)))  # Number of unique species
num_classes <- num_classes[!is.na(num_classes)]  # Remove NA values

# Use a color palette (Set3 has up to 12 colors, extend if needed)
palette_colors <- colorRampPalette(brewer.pal(min(12, num_classes), "Set3"))(num_classes)

# Plot only Band 3 with distinct colors
plot(species_raster, col = palette_colors, main = "Species Classification (Band 3)")


# Plot negative values
negative_values <- band1
negative_values[negative_values >= 0] <- NA  # Keep only negative values

# Convert to data frame for ggplot
df <- as.data.frame(rasterToPoints(negative_values), stringsAsFactors = FALSE)
colnames(df) <- c("x", "y", "value")  # Rename columns

# Plot the negative values
ggplot(df, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "blue", high = "red", na.value = "white") +
  labs(title = "Negative CHM Values", x = "Longitude", y = "Latitude", fill = "CHM Value") +
  theme_minimal()


