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
import cartopy

data_path =  Path.home() / "Documents/UBC/ATSC500/atsc500/" / 'data/unzipped'
cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(cat_url)
dataframe = col.df
threeHourlyModels = dataframe[col.df['table_id'].str.contains("3hr")]
models = threeHourlyModels['source_id'].unique()
# track exceptions in case we care about them later
exceptions = list()

# a dictionary of dictionaries (each key is a model, and each value the pangeo dataset dictionary)
models_with_seb_variables = dict()
variables=['rlds','rlus','rsds','rsus','hfls','hfss', 'tslsi']
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
        exceptions.append(e)

models_with_seb_variables_file = f'{data_path}/models_with_seb_variables.pickle'
with open(f'{data_path}/models_with_seb_variables.pickle', 'wb') as handle:
    pickle.dump(models_with_seb_variables, handle, protocol=pickle.HIGHEST_PROTOCOL)
