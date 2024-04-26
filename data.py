from pathlib import Path
import xarray as xr
from pathlib import Path
import pandas as pd
import pickle 

class Data:
    def __init__(self):
        self.data_path = Path.home() / "Documents/UBC/ATSC500/atsc500/" / 'data/unzipped'
        self.gfdl_name = "subsetted_gfdl_3hr.pickle"
        self.obs_name = "subsetted_obs_3hr.pickle"

    def load_xarray_data(self, path):
        data = xr.open_dataset(path)
        return data

    def load_pickle_data(self, path):
        with open(self.data_path / path, 'rb') as f:
            data = pickle.load(f)
        return  data

    def get_data(self, ):
        obs = self.load_pickle_data(self.obs_name)
        gfdl = self.load_pickle_data(self.gfdl_name)
        return obs, gfdl