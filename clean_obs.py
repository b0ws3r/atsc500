import xarray as xr
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

data_path =  Path.home() / "Documents/UBC/ATSC500/atsc500/" / 'data/unzipped'

# summit_30min_jan2011tojun2014_seb_20160926.cdf
# with netCDF4.Dataset("summit_30min_jan2011tojun2014_seb_20160926.cdf", "r", auto_complex=True) as nc:
the_file = f"{data_path}/summit_30min_jan2011tojun2014_seb_20160926.cdf"
ds1 = xr.open_dataset(the_file)
year_array = ds1['yyyy'].values
time_slice = np.where(year_array >= 2013)[0]
start = time_slice[0]
stop = time_slice[-1]
time_subset = ds1.sel(time=slice(start,stop))

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


# Create a datetime object for each time point
datetimes = pd.to_datetime({
    'year':  map(int, time_subset['yyyy'].values),
    'month': map(int, time_subset['mm'].values),
    'day': map(int, time_subset['dd'].values),
    'hour': map(int, time_subset['hh'].values),
    'minute': map(int, time_subset['nn'].values / (1000000000 * 60))
})


# Replace the integer time dimension with the new datetime dimension
# time_dimension_ds = time_subset.copy()
seb_obs_vars = ['lwdn','lwup', 'swdn' , 'swup' , 'sh_bulk', 'lh_grad', 'cond_flux', 'storage_flux', 'Tsurf', 'lwp']
data_vars = {
    'lwdn': ('datetime', time_subset['lwdn'].values),
    'lwup': ('datetime',  time_subset['lwup'].values),
    'swdn': ('datetime', time_subset['swdn'].values),
    'swup': ('datetime', time_subset['swup'].values),
    'sh_bulk': ('datetime',  time_subset['sh_bulk'].values),
    'lh_grad': ('datetime',  time_subset['lh_grad'].values),
    'cond_flux': ('datetime',  time_subset['cond_flux'].values),
    'storage_flux': ('datetime', time_subset['storage_flux'].values),
    'Tsurf': ('datetime', time_subset['Tsurf'].values),
    'lwp': ('datetime', time_subset['lwp'].values),
}
time_dimension_ds = xr.Dataset(
    data_vars = data_vars,
    coords = {'datetime': datetimes })

seb_obs_vars = ['lwdn','lwup', 'swdn' , 'swup' , 'sh_bulk', 'lh_grad', 'cond_flux', 'storage_flux', 'Tsurf', 'lwp']

# Remove NaN's
time_dimension_ds = time_dimension_ds.where(time_dimension_ds != -999.0, np.nan)


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

lwdn_3hr_avg = lwdn.resample(datetime='3H', offset = thirtyMinOffset).mean(dim='datetime')
lwup_3hr_avg = lwup.resample(datetime='3H', offset = thirtyMinOffset).mean(dim='datetime')
swdn_3hr_avg = swdn.resample(datetime='3H', offset = thirtyMinOffset).mean(dim='datetime')
swup_3hr_avg = swup.resample(datetime='3H', offset = thirtyMinOffset).mean(dim='datetime')
sh_bulk_3hr_avg = sh_bulk.resample(datetime='3H', offset = thirtyMinOffset).mean(dim='datetime')
lh_grad_3hr_avg = lh_grad.resample(datetime='3H', offset = thirtyMinOffset).mean(dim='datetime')
cond_flux_3hr_avg = cond_flux.resample(datetime='3H', offset = thirtyMinOffset).mean(dim='datetime')
storage_flux_3hr_avg = storage_flux.resample(datetime='3H', offset = thirtyMinOffset).mean(dim='datetime')
surface_temp_3hr_avg = surface_temp.resample(datetime='3H', offset = thirtyMinOffset).mean(dim='datetime')
lwp_3hr_avg = lwp.resample(datetime='3H', offset = thirtyMinOffset).mean(dim='datetime')

obs_3hr = xr.merge([lwdn_3hr_avg, lwup_3hr_avg, swdn_3hr_avg, swup_3hr_avg, sh_bulk_3hr_avg, lh_grad_3hr_avg, cond_flux_3hr_avg, storage_flux_3hr_avg, surface_temp_3hr_avg, lwp_3hr_avg])

# save off processed data
obs_3hr_path = f"{data_path}/subsetted_obs_3hr.pickle"
with open(obs_3hr_path, 'wb') as handle:
    pickle.dump(obs_3hr, handle, protocol=pickle.HIGHEST_PROTOCOL)
# obs = obs_3hr.to_netcdf(f"{data_path}/subsetted_obs_3hr.nc", mode='w')
