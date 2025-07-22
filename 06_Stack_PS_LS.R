library(raster)
library(tools)

setwd("//dmawi/potsdam/data/bioing/user/lenguehard/project3")

ps_dir <- "data/satellite/aggregated_PS/"
ls_dir <- "data/satellite/aggregated_LS/"
output_dir <- "data/satellite/stacked/"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Get list of PS files 
ps_files <- list.files(ps_dir, pattern = "_median_filtered_PS\\.tif$", full.names = TRUE, recursive = TRUE)

for (ps_path in ps_files) {
  site_name <- file_path_sans_ext(basename(ps_path))
  site_base <- sub("_median_filtered_PS", "", site_name)
    ls_files <- list.files(ls_dir, pattern = paste0(site_base, "_median_filtered_LS\\.tif$"),
                         full.names = TRUE, recursive = TRUE)
  
  if (length(ls_files) == 0) {
    cat(" Skipping", site_base, "- LS composite not found.\n")
    next
  }
  
  ls_path <- ls_files[1]  
  
  ps <- stack(ps_path)
  ls <- stack(ls_path)
  
  if (!compareRaster(ps, ls, extent=TRUE, rowcol=TRUE, crs=TRUE, res=TRUE, stopiffalse=FALSE)) {
    cat(" Skipping", site_base, "- raster dimensions don't match.\n")
    next
  }
  
  # Create combined mask: valid where all bands > 0
  ps_mask <- calc(ps, fun=function(x) all(x > 0))
  ls_mask <- calc(ls, fun=function(x) all(x > 0))
  combined_mask <- ps_mask * ls_mask
  
  # Apply mask
  ps_masked <- mask(ps, combined_mask, maskvalue=0)
  ls_masked <- mask(ls, combined_mask, maskvalue=0)
  
  # Rename bands
  names(ps_masked) <- paste0(names(ps_masked), "_PS")
  names(ls_masked) <- paste0(names(ls_masked), "_LS")
  
  # Stack and save
  stacked <- stack(ps_masked, ls_masked)
  out_path <- file.path(output_dir, paste0(site_base, "_stacked_PS_LS.tif"))
  writeRaster(stacked, filename = out_path, format = "GTiff", overwrite = TRUE, options = c("COMPRESS=LZW"))
  
  cat("stacked  saved for", site_base, "\n")
}

cat(" All stacking complete!\n")
