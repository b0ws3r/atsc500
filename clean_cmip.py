import xarray as xr
import intake
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
import re

data_path =  Path.home() / "Documents/UBC/ATSC500/atsc500/" / 'data/unzipped'
models_with_seb_variables_file = f'{data_path}/models_with_seb_variables.pickle'

# # start here to reuse preprocessed data
with open(models_with_seb_variables_file, 'rb') as f:
    models_with_seb_variables = pickle.load(f)

# get each model subset to see if it has what we need (greenland) 
experiment_keys = list()
for model in enumerate(models_with_seb_variables):
    experiment_keys.extend([*models_with_seb_variables[model[1]]])

print(experiment_keys)

# summit station 72.6°N, 38.5°W; 3,211 m
# 72.5, 321.5 --> (72,73), (-321, -322)
# data QA-ed for 2013-2014
lon_slice = slice(321, 323)
lat_slice = slice(72, 73)
time_slice = slice("2013-07-01", "2014-6-30")

for model in enumerate(models_with_seb_variables):
    keys = list(models_with_seb_variables[model[1]].keys())
    for experiment in enumerate(keys):
        dataset = (models_with_seb_variables[model[1]])[experiment[1]]
        if 'hfss' in dataset:
            subsetted_data = dataset.sel(lon = lon_slice, lat = lat_slice)
            subsetted_data['datetime'] = subsetted_data.indexes['time'].to_datetimeindex()
            subsetted_data = subsetted_data.sel(datetime=time_slice)
            # print(subsetted_data['rlus'].values)
            # ds = subsetted_data.to_netcdf(f"{data_path}/{experiment[1]}.nc", mode='w')
            with open(f"{data_path}/{experiment[1]}.pickle", 'wb') as handle:
                pickle.dump(subsetted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open(f"{data_path}/CMIP.NOAA-GFDL.GFDL-CM4.historical.3hr.gr1.pickle", 'rb') as f:
    gfdl = pickle.load(f)



##########################################################################
##########################################################################

                        #### resampling

##########################################################################
##########################################################################



# Even though this is the 3hr table, there appear to be 90-min values
hfls = gfdl[['datetime','hfls']]
hfss =  gfdl[['datetime','hfss']]
rlds = gfdl[['datetime','rlds']]
rlus = gfdl[['datetime','rlus']]
rsds = gfdl[['datetime','rsds']]
rsus = gfdl[['datetime','rsus']]
tslsi = gfdl[['datetime','tslsi']]

# gfdl['datetime'] = gfdl.indexes['time'].to_datetimeindex()

hfls_3hr_avg = hfls.resample(datetime='3H').mean(dim='datetime')
hfss_3hr_avg = hfss.resample(datetime='3H').mean(dim='datetime')
rlds_3hr_avg = rlds.resample(datetime='3H').mean(dim='datetime')
rlus_3hr_avg = rlus.resample(datetime='3H').mean(dim='datetime')
rsds_3hr_avg = rsds.resample(datetime='3H').mean(dim='datetime')
rsus_3hr_avg = rsus.resample(datetime='3H').mean(dim='datetime')
tslsi_3hr_avg = tslsi.resample(datetime='3H').mean(dim='datetime')


gfdl_3hr = xr.merge([hfls_3hr_avg,hfss_3hr_avg, rlds_3hr_avg, rlus_3hr_avg, rsds_3hr_avg, rsus_3hr_avg, tslsi_3hr_avg])# save off processed data
# gfdl_3hr = gfdl_3hr.assign_coords(time=gfdl['datetime'] )
gfdl_3hr_path = f"{data_path}/subsetted_gfdl_3hr.pickle"

# with open(gfdl_3hr_path, 'wb') as handle:
#     pickle.dump(gfdl_3hr, handle, protocol=pickle.HIGHEST_PROTOCOL)
gfdl_3hr.to_netcdf(f"{data_path}/subsetted_gfdl_3hr.nc", mode='w')

