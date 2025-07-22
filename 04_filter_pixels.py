# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:18:40 2025
Script to filter data
@author: leengu001
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("results//pixel_table/pixel_table.csv")

data.describe()
data.info()
data.head(5)

data["UAV_tiff"].unique()
data["UAV_tiff"].nunique() 

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
