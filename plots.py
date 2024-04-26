import matplotlib as plt
import xarray as xr
from pathlib import Path


class Plotter:
    def __init__(self): 
        pass

    def savefig(self, fname, fig=None, verbose=True):
        path = Path("..", "figs", fname)
        (plt if fig is None else fig).savefig(path, bbox_inches="tight", pad_inches=0)
        if verbose:
            print(f"Figure saved as '{path}'")


    def plot_temp_data(self, model, obs):
        fig, ax = plt.subplots()
        ax.plot(model['tlsi'], label='GFDL-CM4')
        ax.plot(obs['Tsurf'], label='Obs')
        ax.set_xlabel('X-axis label')
        ax.set_ylabel('Y-axis label')
        plt.show()


    def plot_box_whisker_by_month(self, data):
        fig, ax = plt.subplots()
        data.groupby('time.month').boxplot(column='value', ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Value')
        plt.show()