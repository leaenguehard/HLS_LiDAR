# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:25:58 2025
Script to explore Satellite pixedata
@author: leengu001
"""
# import sys
# print(sys.executable)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("results//pixel_table/pixel_table.csv")

data.describe()
data.info()
data.head(5)

data["UAV_tiff"].unique()
data["UAV_tiff"].nunique() ## There are 89 plots before removing NAs => makes sense

max(data["max_CHM"])
max(data)
print(data.isna().sum())

## Apply filters to data
data_f = data[(data["crown_cov"] >= 0) & (data["crown_cov"] <= 100) & (data["med_CHM"] >= 0)] #Remove crown cover outliers
data_f = data_f.drop(columns='UTM_zone')
data_f = data_f.dropna(subset=['sd_CHM', 'min_CHM', 'max_CHM'], how='any') # filter NA values 
print(data_f.isna().sum())
max(data_f["NDVI_PS"]) # check
min(data_f["med_CHM"]) # check
data_f["UAV_tiff"].nunique() ## There are 86 plots after removing NAs => makes sense

# Save the filtered table to a new CSV file
data_f.to_csv("results/pixel_table/pixel_table_final.csv", index=False)
##
sp_bands_PS = ["blue_PS", "green_PS", "red_PS", "NIR_PS", "SWIR1_PS", "SWIR2_PS"]
sp_bands_LS = ["blue_LS", "green_LS", "red_LS", "NIR_LS", "SWIR1_LS", "SWIR2_LS"]

# Visually explore data
sns.histplot(data=data_f, x= "NDVI_PS", hue = "med_CHM", kde = True).set_title("Peak Summer") #Nice histogram with density !
sns.scatterplot(data = data_f, x = "med_CHM", y = "NDVI_LS", hue ="crown_cov").set_title("Distribution of CHM and NDVI LS")#, style = "UTM_zone")
sns.boxplot(data = data_f, y = "NDVI_PS" ) #change y for 

# sns.pairplot(data = data, hue= "NIR")

# distribution
a = sns.displot(data_f, x="med_CHM", kind="kde")   #, hue = "UTM_zone")
a.ax.set_title("Median CHM per pixel")
a.set_axis_labedata("med", "Bill length (mm)")

a = sns.displot(data_f, x="crown_cov", kind="kde")   #, hue = "UTM_zone")
a.ax.set_title("Median crown_cover")



# Select relevant columns
data_f_PS_melted= data_f.melt(value_vars=sp_bands_PS, var_name="Band", value_name="Value")
data_f_LS_melted= data_f.melt(value_vars=sp_bands_LS, var_name="Band", value_name="Value")

# Plot Late summer
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_f_PS_melted, x="Band", y="Value").set_title("Distribution of Spectral Bands for PS pixels")
plt.xlabel("Spectral Band")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# Plot Peak summer
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_f_LS_melted, x="Band", y="Value")
plt.title("Distribution of Spectral Bands for LS pixels")
plt.xlabel("Spectral Band")
plt.ylabel("Value")
plt.grid(True)



###Side by side

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Late Summer plot
sns.boxplot(data=data_f_LS_melted, x="Band", y="Value", ax=axes[0])
axes[0].set_title("Late Summer")
axes[0].set_xlabel("Spectral Band")
axes[0].set_ylabel("Reflectance Value")
axes[0].grid(True)

# Peak Summer plot
sns.boxplot(data=data_f_PS_melted, x="Band", y="Value", ax=axes[1])
axes[1].set_title("Peak Summer")
axes[1].set_xlabel("Spectral Band")
axes[1].grid(True)

# Main title and layout adjustments
fig.suptitle("Distribution of Spectral bands", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()









