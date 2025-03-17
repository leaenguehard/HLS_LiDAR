##### Label imagery and raster, loop arounf all files#######
# 14/03/2025 - Lea Enguehard
library(raster)
library(dplyr)
library(sf)
setwd("//dmawi/potsdam/data/bioing/user/lenguehard/project3/R")

satellite_folder <- "data/satellite/late_summer/"
uav_folder <- "data/raster_final/"
output_raster_folder <- "results/labeled_imagery/late_summer/"
output_table_path <- "results/pixel_table/late_summer_pixels.csv"  
uav_ids <- list.dirs(satellite_folder, recursive = FALSE, full.names = FALSE)

all_pixel_data <- data.frame()

percentage_ones <- function(x, ...) {
  if (all(is.na(x))) return(NA)
  return(sum(x == 1, na.rm = TRUE) / length(x[!is.na(x)]) * 100)
}
get_mode <- function(x, ...) {
  if (all(is.na(x))) return(NA)
  tabulated <- table(x)
  return(as.numeric(names(which.max(tabulated))))
}
get_second_mode <- function(x, ...) {  
  if (all(is.na(x))) return(NA)  
  tabulated <- table(x)
  if (length(tabulated) < 2) return(NA)
  sorted_values <- sort(tabulated, decreasing = TRUE)
  return(as.numeric(names(sorted_values[2])))
}

# Loop through each UAV ID
for (uav_id in uav_ids) {
  tryCatch({
    
    # Get list of all satellite rasters in this UAV folder
    sat_files <- list.files(path = file.path(satellite_folder, uav_id), 
                            pattern = "\\.tif$", full.names = TRUE)
    uav_path <- file.path(uav_folder, paste0(uav_id, "_final.tif"))
    
    # Check if UAV file exists
    if (!file.exists(uav_path)) {
      message(paste("Skipping", uav_id, "- UAV file not found."))
      next
    }
    
    # Load UAV raster (same for all satellite files of this UAV ID)
    uav <- stack(uav_path)
    
    # Loop through each satellite raster for the current UAV ID
    for (sat_path in sat_files) {
      tryCatch({
        
        # Load the satellite raster
        sat <- stack(sat_path)
        
        # Extract UAV bands
        uav_band1 <- uav[[1]]  # Continuous
        uav_band2 <- uav[[2]]  # Binary (0 or 1)
        uav_band3 <- uav[[3]]  # Categorical
        
        # Compute aggregation factor based on resolution
        fact_x <- res(sat)[1] / res(uav_band1)[1]
        fact_y <- res(sat)[2] / res(uav_band1)[2]
        
        # Aggregate UAV data
        uav1_median <- aggregate(uav_band1, fact = c(fact_x, fact_y), fun = median, na.rm = TRUE)
        uav1_std <- aggregate(uav_band1, fact = c(fact_x, fact_y), fun = sd, na.rm = TRUE)
        uav2_percentage <- aggregate(uav_band2, fact = c(fact_x, fact_y), fun = percentage_ones)
        uav3_mode <- aggregate(uav_band3, fact = c(fact_x, fact_y), fun = get_mode)
        uav3_second_mode <- aggregate(uav_band3, fact = c(fact_x, fact_y), fun = get_second_mode)
        
        # Set extent to match satellite raster
        extent(uav1_median) <- extent(sat)
        extent(uav1_std) <- extent(sat)
        extent(uav2_percentage) <- extent(sat)
        extent(uav3_mode) <- extent(sat)
        extent(uav3_second_mode) <- extent(sat)
        
        # Resample to match the extent and resolution of sat
        uav1_median_resampled <- resample(uav1_median, sat, method = "bilinear")
        uav1_std_resampled <- resample(uav1_std, sat, method = "bilinear")
        uav2_percentage_resampled <- resample(uav2_percentage, sat, method = "bilinear")
        uav3_mode_resampled <- resample(uav3_mode, sat, method = "ngb")  # Nearest neighbor for categorical data
        uav3_second_mode_resampled <- resample(uav3_second_mode, sat, method = "ngb")
        
        # Add new UAV-derived bands to satellite stack
        sat <- addLayer(sat, uav1_median_resampled, uav1_std_resampled, 
                        uav2_percentage_resampled, uav3_mode_resampled, uav3_second_mode_resampled)
        
        # Remove pixels where bands 7-11 are all NA
        new_bands <- sat[[7:11]]
        na_mask <- calc(new_bands, function(x) ifelse(all(is.na(x)), NA, 1))  # Mask where all bands are NA
        sat_masked <- mask(sat, na_mask)
        
        # Rename bands
        new_band_names <- c("blue", "green", "red", "NIR", "SWIR1", "SWIR2", 
                            "med_CHM", "sd_CHM", "crown_cov", "dom1_sp", "dom2_sp")
        names(sat_masked) <- new_band_names
        
        # Save the processed raster with UAV ID and Satellite filename
        sat_filename <- tools::file_path_sans_ext(basename(sat_path))
        raster_output_path <- file.path(output_raster_folder, paste0(sat_filename, "_labeled.tif"))
        writeRaster(sat_masked, raster_output_path, format = "GTiff", overwrite = TRUE)
        
        # Create pixel table
        df <- as.data.frame(sat_masked, xy = TRUE, na.rm = TRUE)
        colnames(df)[1:2] <- c("longitude", "latitude")
        
        # Extract UTM Zone
        utm_zone <- gsub(".*zone=([0-9]+).*", "\\1", as.character(crs(sat_masked)))
        
        # Add metadata columns
        df$UAV_tiff <- paste0(uav_id, "_final.tif")
        df$sat_tiff <- basename(sat_path)  # Keep full satellite TIF filename
        df$UTM_zone <- utm_zone
        
        # Reorder columns
        df <- df %>% select(longitude, latitude, UTM_zone, all_of(new_band_names), UAV_tiff, sat_tiff)
        
        # Append to master dataframe
        all_pixel_data <- bind_rows(all_pixel_data, df)
        
        # Print status
        message(paste("Processed:", sat_filename))
        
      }, error = function(e) {
        message(paste("Error processing:", sat_path, "- Skipping."))
      })
    }
    
  }, error = function(e) {
    message(paste("Error processing UAV ID:", uav_id, "- Skipping."))
  })
}
write.csv(all_pixel_data, output_table_path, row.names = FALSE)
message("Final aggregated pixel table saved.")
