---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Getting 3hrly data for surface temp, and radiation variables
Need to get radiation variables and compare to the observations at summit station



### Questions: 
- Which experiment has the lowest surface temp bias?
- Why?
- In each model, which terms contribute the most to forcing?

```{code-cell} ipython3
# Import statements
import xarray as xr
xr.set_options(display_style='html')
import intake
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
import zarr
import re
import cartopy

data_path =  Path.home() / "repos/a500/" /'final/data/'
```

+++ {"jp-MarkdownHeadingCollapsed": true}

## Get CMIP models

+++

Pull the data from the sit itself and put into pandas df in order to be able to visualize it.

```{code-cell} ipython3
cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(cat_url)
```

```{code-cell} ipython3
dataframe = col.df
threeHourlyModels = dataframe[col.df['table_id'].str.contains("3hr")]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
threeHourlyModels
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Look at the unique keys for all the entries in order to find the models we want

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
models = threeHourlyModels['source_id'].unique()
models
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Model / Variable Info

+++ {"editable": true, "slideshow": {"slide_type": ""}, "jp-MarkdownHeadingCollapsed": true}

### Monthly ice sheet SEB variables

| Variable id | Variable name|
|-------------|--------------|
|rsdsls|Ice Sheet Surface Downwelling Shortwave Radiation [W m-2]
|rsusIs|Ice Sheet Surface Upwelling Shortwave Radiation [W m-2]
|rldsIs|Ice Sheet Surface Downwelling Longwave Radiation [W m-2]
|rlusIs|Ice Sheet Surface Upwelling Longwave Radiation [W m-2]
|albsn|Snow Albedo [1]
|acabfIs|Ice Sheet Surface Mass Balance Flux [kg m-2 s-1]
|hfssIs|Ice Sheet Surface Upward Sensible Heat Flux [W m-2]
|hflsIs|Ice Sheet Surface Upward Latent Heat Flux [W m-2]

### 3hrly SEB variables

| Variable id | Variable name|
|-------------|--------------|
|rld|Downwelling Longwave Radiation [W m-2]
|rlds|Surface Downwelling Longwave Radiation [W m-2]
|rlus|Surface Upwelling Longwave Radiation [W m-2]
|rlu|Upwelling Longwave Radiation [W m-2]
|rsd|Downwelling Shortwave Radiation [W m-2]
|rsu|Upwelling Shortwave Radiation [W m-2]
|rsds|Surface Downwelling Shortwave Radiation [W m-2]
|rsus|Surface Upwelling Shortwave Radiation [W m-2]
|hfls|Surface Upward Latent Heat Flux [W m-2]
|hfss|Surface Upward Sensible Heat Flux [W m-2]

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Collect and subset data for each model

- Summit Station 72.6°N, 38.5°W; 3,211 m
- Data QA-ed for 2013-2014, so we will use that time period

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# track exceptions in case we care about them later
exceptions = list()

# a dictionary of dictionaries (each key is a model, and each value the pangeo dataset dictionary)
models_with_seb_variables = dict()
variables=['rld','rlds','rlus','rlu','rsd','rsu','rsds','rsus','hfls','hfss', 'ts', 'tslsi', 'tas', 'tsIs']
for model in enumerate(models):
    model_dict = dict()
    model_subset = col.search(table_id="3hr", variable_id = variables,# 'rsus', 
                                source_id = model, 
                                experiment_id = 'historical')
    try:
        model_dict = model_subset.to_dataset_dict(zarr_kwargs={'consolidated':True})
        if len(model_dict) > 0:
            #The keys in the returned dictionary of datasets are constructed as follows:
        	#'activity_id.institution_id.source_id.experiment_id.table_id.grid_label'
            models_with_seb_variables[model[1]] = model_dict
            print(f"Model '{model}' contains 3hr variables")
    except Exception as e:
        # pass
        exceptions.append(e)
    
```

```{code-cell} ipython3
models_with_seb_variables
```

### Chosen model 
Only `CMIP.NOAA-GFDL.GFDL-CM4.historical.3hr` has sensible/latent heat flux variables!

```{code-cell} ipython3
# Save off the model information we collected
models_with_seb_variables_file = f'{data_path}/models_with_seb_variables.pickle'
```

```{code-cell} ipython3
with open(f'{data_path}/models_with_seb_variables.pickle', 'wb') as handle:
    pickle.dump(models_with_seb_variables, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```{code-cell} ipython3
# start here to reuse preprocessed data
with open(models_with_seb_variables_file, 'rb') as f:
    models_with_seb_variables = pickle.load(f)
```

```{code-cell} ipython3
# get each model subset to see if it has what we need (greenland) 
experiment_keys = list()
for model in enumerate(models_with_seb_variables):
    experiment_keys.extend([*models_with_seb_variables[model[1]]])

experiment_keys
```

```{code-cell} ipython3
# summit station 72.6°N, 38.5°W; 3,211 m
# 72.5, 321.5 --> (72,73), (-321, -322)
# data QA-ed for 2013-2014
lon_slice = slice(321, 323)
lat_slice = slice(72, 73)
time_slice = slice('2013', '2014')

for model in enumerate(models_with_seb_variables):
    keys = list(models_with_seb_variables[model[1]].keys())
    for experiment in enumerate(keys):
        dataset = (models_with_seb_variables[model[1]])[experiment[1]]
        if 'hfss' in dataset:
            subsetted_data = dataset.sel(lon = lon_slice, lat = lat_slice, time = time_slice)
            # print(subsetted_data['rlus'].values)
            # ds = subsetted_data.to_netcdf(f"{data_path}/{experiment[1]}.nc", mode='w')
            with open(f"{data_path}/{experiment[1]}.pickle", 'wb') as handle:
                pickle.dump(subsetted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```{code-cell} ipython3
subsetted_data
```

### Clean and subset gfdl dataset

```{code-cell} ipython3
# subsetted_data.load().to_zarr(f"{data_path}/{experiment[1]}.zarr")

with open(f"{data_path}/CMIP.NOAA-GFDL.GFDL-CM4.historical.3hr.gr1.pickle", 'rb') as f:
    gfdl = pickle.load(f)
```

```{code-cell} ipython3
gfdl
```

```{code-cell} ipython3


gfdl['datetime'] = gfdl.indexes['time'].to_datetimeindex()
gfdl['datetime']
# gfdl.to_netcdf(f"{data_path}/subsetted_gfdl_3hr.nc", mode='w')
```

```{code-cell} ipython3
# variables=['rld','rlds','rlus','rlu','rsd','rsu','rsds','rsus','hfls','hfss', 'ts', 'tslsi', 'tas', 'tsIs']

# 30 min observations
hfls = gfdl[['datetime','hfls']]
hfss =  gfdl[['datetime','hfss']]
rlds = gfdl[['datetime','rlds']]
rlus = gfdl[['datetime','rlus']]
rsds = gfdl[['datetime','rsds']]
rsus = gfdl[['datetime','rsus']]
tas  = gfdl[['datetime','tas']]
tslsi = gfdl[['datetime','tslsi']]

thirtyMinOffset = pd.Timedelta(unit='minutes', value=0)

hfls_3hr_avg = hfls.resample(datetime='3H', offset = thirtyMinOffset).mean()
hfss_3hr_avg = hfss.resample(datetime='3H', offset = thirtyMinOffset).mean()
rlds_3hr_avg = rlds.resample(datetime='3H', offset = thirtyMinOffset).mean()
rlus_3hr_avg = rlus.resample(datetime='3H', offset = thirtyMinOffset).mean()
rsds_3hr_avg = rsds.resample(datetime='3H', offset = thirtyMinOffset).mean()
rsus_3hr_avg = rsus.resample(datetime='3H', offset = thirtyMinOffset).mean()
tas_3hr_avg = tas.resample(datetime='3H', offset = thirtyMinOffset).mean()
tslsi_3hr_avg = tslsi.resample(datetime='3H', offset = thirtyMinOffset).mean()

gfdl_3hr = xr.merge([hfls_3hr_avg,hfss_3hr_avg, rlds_3hr_avg, rlus_3hr_avg, rsds_3hr_avg, rsus_3hr_avg, tas_3hr_avg, tslsi_3hr_avg])
```

```{code-cell} ipython3
hfls_3hr_avg
```

```{code-cell} ipython3
gfdl_3hr = gfdl_3hr.assign_coords(time=gfdl['datetime'] )
ds_dropped = gfdl_3hr.drop_dims('time')


gfdl_3hr
```

```{code-cell} ipython3
# save off processed data
gfdl_3hr_path = f"{data_path}/subsetted_gfdl_3hr.pickle"
# with open(gfdl_3hr_path, 'wb') as handle:
#     pickle.dump(gfdl_3hr, handle, protocol=pickle.HIGHEST_PROTOCOL)
gfdl_3hr.to_netcdf(f"{data_path}/subsetted_gfdl_3hr.nc", mode='w')
```

## Get observations

```{code-cell} ipython3
# netcdf get 2013-2014 observations
# summit_30min_jan2011tojun2014_seb_20160926.cdf
# with netCDF4.Dataset("summit_30min_jan2011tojun2014_seb_20160926.cdf", "r", auto_complex=True) as nc:
the_file = Path.home() / "repos/a500/" /'final/data/summit_30min_jan2011tojun2014_seb_20160926.cdf'
# obs_all = netcdf_file(the_file, 'r')
ds1 = xr.open_dataset(the_file)
# ds1['time'].data
# ds1.loc[2013:2014, 'yyyy']
year_array = ds1['yyyy'].values
time_slice = np.where(year_array >= 2013)[0]
start = time_slice[0]
stop = time_slice[-1]

time_subset = ds1.sel(time=slice(start,stop))
# time_subset.variables
ds1.info()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
time_subset['dd'].values

zeros = np.where(time_subset['dd'].values == 0)[0]
# the minutes are in nanoseconds, which is fun
print(len(time_subset['nn'].values / (1000000000 * 60)))
```

```{code-cell} ipython3
# Let's clean up the integer time format, because it's annoying

# Create a datetime object for each time point
datetimes = pd.to_datetime({
    'year':  map(int, time_subset['yyyy'].values),
    'month': map(int, time_subset['mm'].values),
    'day': map(int, time_subset['dd'].values),
    'hour': map(int, time_subset['hh'].values),
    'minute': map(int, time_subset['nn'].values / (1000000000 * 60))
})


# Replace the integer time dimension with the new datetime dimension
time_dimension_ds = time_subset.copy()
time_dimension_ds['datetime'] = datetimes
```

```{code-cell} ipython3
time_dimension_ds['datetime']
```

```{code-cell} ipython3
# lwdn - lwup + swdn - swup - sh_bulk(or sh_cv if avail.) - lh_grad + cond_flux + storage_flux
seb_obs_vars = ['lwdn','lwup'  'swdn' , 'swup' , 'sh_bulk', 'lh_grad', 'cond_flux', 'storage_flux', 'Tsurf', 'lwp']

# 30 min observations
lwdn = time_dimension_ds[['datetime','lwdn']]
lwup =  time_dimension_ds[['datetime','lwup']]
swdn = time_dimension_ds[['datetime','swdn']]
swup = time_dimension_ds[['datetime','swup']]
sh_bulk = time_dimension_ds[['datetime','sh_bulk']]
lh_grad = time_dimension_ds[['datetime','lh_grad']]
cond_flux = time_dimension_ds[['datetime','cond_flux']]
storage_flux = time_dimension_ds[['datetime','storage_flux']]
surface_temp = time_dimension_ds[['datetime','Tsurf']]
lwp = time_dimension_ds[['datetime','lwp']]

thirtyMinOffset = pd.Timedelta(unit='minutes', value=0)

lwdn_3hr_avg = lwdn.resample(datetime='3H', offset = thirtyMinOffset).mean()
lwup_3hr_avg = lwup.resample(datetime='3H', offset = thirtyMinOffset).mean()
swdn_3hr_avg = swdn.resample(datetime='3H', offset = thirtyMinOffset).mean()
swup_3hr_avg = swup.resample(datetime='3H', offset = thirtyMinOffset).mean()
sh_bulk_3hr_avg = sh_bulk.resample(datetime='3H', offset = thirtyMinOffset).mean()
lh_grad_3hr_avg = lh_grad.resample(datetime='3H', offset = thirtyMinOffset).mean()
cond_flux_3hr_avg = cond_flux.resample(datetime='3H', offset = thirtyMinOffset).mean()
storage_flux_3hr_avg = storage_flux.resample(datetime='3H', offset = thirtyMinOffset).mean()
surface_temp_3hr_avg = surface_temp.resample(datetime='3H', offset = thirtyMinOffset).mean()
lwp_3hr_avg = lwp.resample(datetime='3H', offset = thirtyMinOffset).mean()
```

```{code-cell} ipython3
lwdn_3hr_avg['lwdn']
print(lwdn.shape())
```

```{code-cell} ipython3
obs_3hr = xr.merge([lwdn_3hr_avg, lwup_3hr_avg, swdn_3hr_avg, swup_3hr_avg, sh_bulk_3hr_avg, lh_grad_3hr_avg, cond_flux_3hr_avg, storage_flux_3hr_avg, surface_temp_3hr_avg, lwp_3hr_avg])
```

```{code-cell} ipython3
obs_3hr
```

```{code-cell} ipython3
# save off processed data
obs_3hr_path = f"{data_path}/subsetted_obs_3hr.pickle"
with open(obs_3hr_path, 'wb') as handle:
    pickle.dump(obs_3hr, handle, protocol=pickle.HIGHEST_PROTOCOL)
# obs = obs_3hr.to_netcdf(f"{data_path}/subsetted_obs_3hr.nc", mode='w')
```

## Forcing vs Response in each model

Example SEB calculation = lwdn - lwup + swdn - swup - sh_bulk(or sh_cv if avail.) - lh_grad + cond_flux + storage_flux

Response variables:
- lh + sh + g - lwup

```{code-cell} ipython3
# Save off the SEB and the response terms
# ['rld','rlds','rlus','rlu','rsd','rsu','rsds','rsus','hfls','hfss']
g = gfdl['rlds'] - gfdl['rlus'] + gfdl['rsds'] -  gfdl['rsus'] - gfdl['hfls'] - gfdl['hfss']
gfdl['g'] = g

model_response_terms = - gfdl['rlus'] +  gfdl['g'] + gfdl['hfls'] + gfdl['hfss']
gfdl['model_response_terms'] = model_response_terms
```

```{code-cell} ipython3
gfdl
```

```{code-cell} ipython3
seb = lwdn_3hr_avg['lwdn'] - lwup_3hr_avg['lwup'] + swdn_3hr_avg['swdn'] - swup_3hr_avg['swup'] - sh_bulk_3hr_avg['sh_bulk'] - lh_grad_3hr_avg['lh_grad'] + cond_flux_3hr_avg['cond_flux'] + storage_flux_3hr_avg['storage_flux']
response_terms = sh_bulk_3hr_avg['sh_bulk'] + lh_grad_3hr_avg['lh_grad'] + cond_flux_3hr_avg['cond_flux'] + storage_flux_3hr_avg['storage_flux'] - lwup_3hr_avg['lwup
```

```{code-cell} ipython3
seb
# response_terms
```

## Temperature plots

```{code-cell} ipython3
obs_3hr_path = f"{data_path}/subsetted_obs_3hr.pickle"

# with open(obs_3hr_path, 'rb') as handle:
#     obs_3hr = pickle.load(obs_3hr_path)

T_obs = obs_3hr['Tsurf']
# T_model = gfdl['tslsi'].values

# T_model
```

```{code-cell} ipython3
obs_3hr
```

```{code-cell} ipython3
# tslsi_day_avg = gfdl.resample(datetime='1m').mean()
tsurf_day_avg = obs_3hr[['datetime', 'Tsurf']].resample(datetime='1m').mean()
```

```{code-cell} ipython3
tsurf_day_avg['Tsurf']
```

```{code-cell} ipython3
fig, ax = plt.subplots()
# ax.plot(T_model['datetime'].values, T_model, label='GFDL-CM4')
ax.plot(
    # tsurf_day_avg['datetime'].values, 
    tsurf_day_avg['Tsurf'].values, label='Obs')
ax.set_xlabel('X-axis label')
ax.set_ylabel('Y-axis label')
plt.show()
```

+++ {"jp-MarkdownHeadingCollapsed": true}

# old

```{code-cell} ipython3
bc_dset = xr.open_dataset('can_bc_dset.zarr')
mean_precip = bc_dset.groupby('time.year').mean('time').mean(['lon', 'lat'])*86400*365
plt.figure()
mean_precip.mean('member_id').pr.plot()
plt.title("Averaged Yearly precipitation for the CanESM5 GCM")
plt.xlabel('Year')
plt.ylabel('Precipitation total (mm)')
```

```{code-cell} ipython3
var_precip = bc_dset.groupby('time.year').mean('time').mean(['lon', 'lat'])*86400*365
var_precip.std('member_id').pr.plot()
plt.title("Standard deviation of precipitation for the CanESM5 members")
plt.xlabel('Year')
plt.ylabel('Precipitation total (mm)')
```

```{code-cell} ipython3
hist_data = bc_dset.groupby('time.year').mean('time').mean(['lon', 'lat'])*86400*365
hist_data = hist_data.sel(year=2010)
hist_data.pr.plot.hist()
plt.title("2010 Precipitation Average distribution across the CanESM5 members")
plt.xlabel('Precipitation total (mm)')
plt.ylabel('Number')
```

```{code-cell} ipython3
## Try plotting on a map for 2010

data2010 = bc_dset.sel(time='2010')
precip_data2010 = data2010.groupby('time.year').mean('time')*86400*365
precip_data2010 = precip_data2010.mean('member_id')

fig = plt.figure(1, figsize=[30,13])

ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1)
ax.set_extent([-140, -110, 44, 60])

resol = '50m'

provinc_bodr = cartopy.feature.NaturalEarthFeature(category='cultural', 
    name='admin_1_states_provinces_lines', scale=resol, facecolor='none', edgecolor='k')
ax.add_feature(provinc_bodr, linestyle='--', linewidth=0.6, edgecolor="k", zorder=10)



precip_data2010.pr.plot(ax=ax,cmap='coolwarm')
ax.title.set_text("Precipitation total for 2010")
```

+++ {"jp-MarkdownHeadingCollapsed": true}

# Collect

```{code-cell} ipython3
hadGEM = False
if hadGEM:
    had_subset = col.search(table_id="Amon", variable_id = "pr", source_id = "HadGEM3-GC31-MM", experiment_id = 'historical')
    dset_dict = had_subset.to_dataset_dict(zarr_kwargs={'consolidated':True})
    had_dset = dset_dict['CMIP.MOHC.HadGEM3-GC31-MM.historical.Amon.gn']
    had_bc_dset = had_dset.sel(lon = slice(225.4, 239.6), lat = slice(48.835241, 59.99702), time = slice('1960', '2010'))
    had_bc_dset.load().to_zarr('had_bc_dset.zarr')
    print('done')
```

```{code-cell} ipython3
had_bc_dset = xr.open_dataset('had_bc_dset.zarr')
mean_precip_had = had_bc_dset.groupby('time.year').mean('time').mean(['lon', 'lat'])*86400*365
plt.figure()
mean_precip_had.mean('member_id').pr.plot()
plt.title("Averaged Yearly precipitation for the HadGEM3 GCM")
plt.xlabel('Year')
plt.ylabel('Precipitation total (mm)')
```

```{code-cell} ipython3
had_std_precip = had_bc_dset.groupby('time.year').mean('time').mean(['lon', 'lat'])*86400*365
had_std_precip.std('member_id').pr.plot()
plt.title("Standard deviation of precipitation for the HadGEM members")
plt.xlabel('Year')
plt.ylabel('Precipitation total (mm)')
```

```{code-cell} ipython3
had_hist_data = had_bc_dset.groupby('time.year').mean('time').mean(['lon', 'lat'])*86400*365
had_hist_data = had_hist_data.sel(year=2010)
had_hist_data.pr.plot.hist()
plt.title("2010 Precipitation Average distribution across the HadGEM members")
plt.xlabel('Precipitation total (mm)')
plt.ylabel('Number')
```

```{code-cell} ipython3
had_data2010 = had_bc_dset.sel(time='2005')
had_precip_data2010 = had_data2010.groupby('time.year').mean('time')*86400*365
had_precip_data2010 = had_precip_data2010.mean('member_id')

fig = plt.figure(1, figsize=[30,13])

ax2 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1)
ax2.set_extent([-140, -110, 40, 60])

resol = '50m'

provinc_bodr = cartopy.feature.NaturalEarthFeature(category='cultural', 
    name='admin_1_states_provinces_lines', scale=resol, facecolor='none', edgecolor='k')
ax2.add_feature(provinc_bodr, linestyle='--', linewidth=0.6, edgecolor="k", zorder=10)



had_precip_data2010.pr.plot(ax=ax2,cmap='coolwarm')
ax.title.set_text("Precipitation total for 2010")
```

+++ {"jp-MarkdownHeadingCollapsed": true}

# GISS

```{code-cell} ipython3
GISS = False
if GISS:
    gis_subset = col.search(table_id="Amon", variable_id = "pr", source_id = "GISS-E2-1-H", experiment_id = 'historical')
    dset_dict = gis_subset.to_dataset_dict(zarr_kwargs={'consolidated':True})
    gis_dset = dset_dict['CMIP.NASA-GISS.GISS-E2-1-H.historical.Amon.gn']
    gis_bc_dset = gis_dset.sel(lon = slice(226.25, 238.75), lat = slice(48.835241, 59.99702), time = slice('1960', '2010'))
    gis_bc_dset.load().to_zarr('gis_bc_dset.zarr')
```

```{code-cell} ipython3
gis_bc_dset = xr.open_dataset('gis_bc_dset.zarr')
mean_precip_gis = gis_bc_dset.groupby('time.year').mean('time').mean(['lon', 'lat'])*86400*365
plt.figure()
mean_precip_gis.mean('member_id').pr.plot()
plt.title("Averaged Yearly precipitation for the GISS GCM")
plt.xlabel('Year')
plt.ylabel('Precipitation total (mm)')
```

```{code-cell} ipython3
gis_std_precip = gis_bc_dset.groupby('time.year').mean('time').mean(['lon', 'lat'])*86400*365
gis_std_precip.std('member_id').pr.plot()
plt.title("Standard deviation of precipitation for the HadGEM members")
plt.xlabel('Year')
plt.ylabel('Precipitation total (mm)')
```

```{code-cell} ipython3
gis_hist_data = gis_bc_dset.groupby('time.year').mean('time').mean(['lon', 'lat'])*86400*365
gis_hist_data = gis_hist_data.sel(year=2010)
gis_hist_data.pr.plot.hist()
plt.title("2010 Precipitation Average distribution across the GISS members")
plt.xlabel('Precipitation total (mm)')
plt.ylabel('Number')
```

```{code-cell} ipython3
gis_data1990 = gis_bc_dset.sel(time='2010')
gis_precip_data1990 = gis_data1990.groupby('time.year').mean('time')*86400*365
gis_precip_data1990 = gis_precip_data1990.mean('member_id')

fig = plt.figure(1, figsize=[30,13])

ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1)
ax.set_extent([-140, -110, 40, 60])

resol = '50m'

provinc_bodr = cartopy.feature.NaturalEarthFeature(category='cultural', 
    name='admin_1_states_provinces_lines', scale=resol, facecolor='none', edgecolor='k')
ax.add_feature(provinc_bodr, linestyle='--', linewidth=0.6, edgecolor="k", zorder=10)



gis_precip_data1990.pr.plot(ax=ax,cmap='coolwarm')
```

### Creating plots with all 3 models

```{code-cell} ipython3
time = mean_precip_gis.year
fig, axs = plt.subplots(1, 1, figsize=(15, 8))
axs.plot(time,mean_precip_gis.mean('member_id').pr,label='gis')
axs.plot(time, mean_precip_had.mean('member_id').pr,label='hadley')
axs.plot(time, mean_precip.mean('member_id').pr,label='cccma')
axs.legend();
```

```{code-cell} ipython3

```
