from plots import Plotter
import xarray as xr
from pathlib import Path
import pandas as pd
import pickle 
from data import Data


(obs, gfdl) = Data().get_data()
Plotter.plot_temp_data(gfdl, obs)

