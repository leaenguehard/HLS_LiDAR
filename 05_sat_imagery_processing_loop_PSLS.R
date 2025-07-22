## March 24

library(raster)
library(dplyr)
library(sf)
library(tools)

setwd("//dmawi/potsdam/data/bioing/user/lenguehard/project3")

# Directories
satellite_folder <- "data/satellite/stacked/"
uav_folder <- "data/raster_final/"
output_raster_folder <- "results/labeled_imagery/aggregated/"
output_table_path <- "results/pixel_table/pixel_table.csv"
dir.create(output_raster_folder, showWarnings = FALSE, recursive = TRUE)

# Get all stacked raster files
sat_files <- list.files(satellite_folder, pattern = "\\.tif$", full.names = TRUE)

all_pixel_data <- data.frame()

# Helper functions
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

# Loop through each stacked satellite raster
for (sat_path in sat_files) {
  tryCatch({
    sat_filename <- file_path_sans_ext(basename(sat_path))
    uav_id <- sub("_stacked_PS_LS$", "", sat_filename)
    uav_path <- file.path(uav_folder, paste0(uav_id, "_final.tif"))
    
    if (!file.exists(uav_path)) {
      message(paste(" Skipping", uav_id, "- UAV file not found."))
      next
    }
    
    # Load rasters
    sat <- stack(sat_path)
    uav <- stack(uav_path)
    
    # Reproject each UAV band to match satellite CRS
    uav_band1 <- projectRaster(uav[[1]], crs = crs(sat), method = "bilinear")  # CHM
    uav_band1[uav_band1 < 1] <- NA  # Mask low CHM
    uav_band2 <- projectRaster(uav[[2]], crs = crs(sat), method = "ngb")       # Crown cover
    uav_band3 <- projectRaster(uav[[3]], crs = crs(sat), method = "ngb")       # Species
    
    # Check resolution for aggregation
    res_sat <- res(sat)
    res_uav <- res(uav_band1)
    fact_x <- res_sat[1] / res_uav[1]
    fact_y <- res_sat[2] / res_uav[2]
    
    aggregate_uav <- (fact_x >= 1 && fact_y >= 1)
    
    if (aggregate_uav) {
      uav1_median <- aggregate(uav_band1, fact = c(fact_x, fact_y), fun = "median", na.rm = TRUE)
      uav1_std    <- aggregate(uav_band1, fact = c(fact_x, fact_y), fun = "sd", na.rm = TRUE)
      uav1_min    <- aggregate(uav_band1, fact = c(fact_x, fact_y), fun = "min", na.rm = TRUE)
      uav1_max    <- aggregate(uav_band1, fact = c(fact_x, fact_y), fun = "max", na.rm = TRUE)
      uav2_percentage <- aggregate(uav_band2, fact = c(fact_x, fact_y), fun = percentage_ones)
      uav3_mode   <- aggregate(uav_band3, fact = c(fact_x, fact_y), fun = get_mode)
      uav3_second_mode <- aggregate(uav_band3, fact = c(fact_x, fact_y), fun = get_second_mode)
    } else {
      message(paste("Skipping aggregation for", uav_id, "- Using direct resampling."))
      uav1_median <- uav_band1
      uav1_std    <- uav_band1
      uav1_min    <- uav_band1
      uav1_max    <- uav_band1
      uav2_percentage <- uav_band2
      uav3_mode   <- uav_band3
      uav3_second_mode <- uav_band3
    }
    
    # Resample UAV layers to satellite grid
    uav1_median_resampled <- resample(uav1_median, sat, method = "bilinear")
    uav1_std_resampled <- resample(uav1_std, sat, method = "ngb")
    uav1_min_resampled <- resample(uav1_min, sat, method = "ngb")
    uav1_max_resampled <- resample(uav1_max, sat, method = "ngb")
    uav2_percentage_resampled <- resample(uav2_percentage, sat, method = "bilinear")
    uav3_mode_resampled <- resample(uav3_mode, sat, method = "ngb")
    uav3_second_mode_resampled <- resample(uav3_second_mode, sat, method = "ngb")
    
    # extent(uav1_median_resampled) <- extent(sat)
    # extent(uav1_std_resampled) <- extent(sat)
    # extent(uav1_min_resampled) <- extent(sat)
    # extent(uav1_max_resampled) <- extent(sat)
    # extent(uav2_percentage_resampled) <- extent(sat)
    # extent(uav3_mode_resampled) <- extent(sat)
    # extent(uav3_second_mode_resampled) <- extent(sat)

    #extent(uav1_median_resampled) <- extent(sat)
    # Stack all bands
    sat <- addLayer(sat,
                    uav1_median_resampled, uav1_std_resampled,
                    uav1_min_resampled, uav1_max_resampled,
                    uav2_percentage_resampled, uav3_mode_resampled,
                    uav3_second_mode_resampled)
    
    # Create NA mask: keep pixels where at least one satellite band is valid
    bands_ps_ls <- sat[[1:14]]
    na_mask <- calc(bands_ps_ls, function(x) {
      if (all(is.na(x))) return(NA)
      return(1)
    })
    sat_masked <- mask(sat, na_mask)
    
    # Rename bands
    new_band_names <- c(
      paste0(c("blue", "green", "red", "NIR", "SWIR1", "SWIR2", "NDVI"), "_PS"),
      paste0(c("blue", "green", "red", "NIR", "SWIR1", "SWIR2", "NDVI"), "_LS"),
      "med_CHM", "sd_CHM", "min_CHM", "max_CHM", "crown_cov", "dom1_sp", "dom2_sp"
    )
    names(sat_masked) <- new_band_names
    
    # Save labeled raster
    raster_output_path <- file.path(output_raster_folder, paste0(sat_filename, "_labeled.tif"))
 #   writeRaster(sat_masked, raster_output_path, format = "GTiff", overwrite = TRUE)
    
    # Create pixel table
    df <- as.data.frame(sat_masked, xy = TRUE) #, na.rm = TRUE)
    colnames(df)[1:2] <- c("longitude", "latitude")
    
    # Extract UTM zone if available
    utm_zone <- tryCatch({
      gsub(".*zone=([0-9]+).*", "\\1", as.character(crs(sat_masked)))
    }, error = function(e) { NA })
    
    df$UAV_tiff <- paste0(uav_id, "_final.tif")
    df$sat_tiff <- basename(sat_path)
    df$UTM_zone <- utm_zone
    
    df <- df %>% select(longitude, latitude, UTM_zone, all_of(new_band_names), UAV_tiff, sat_tiff)
    
    # Append to master table
    all_pixel_data <- bind_rows(all_pixel_data, df)
    
    message(paste("PRocessed:", sat_filename))
    
  }, error = function(e) {
    message(paste("Error processing:", sat_path, "-", e$message))
  })
}

# Save pixel table
write.csv(all_pixel_data, output_table_path, row.names = FALSE)
message("Final aggregated pixel table saved.")
