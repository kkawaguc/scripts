import xarray as xr
from SST_patch_analysis_functions import *

experiment_names = ['Greens_function_control', 'wp_p2k', 'wp_m2k', 'ep_p2k', 'ep_m2k',
                    'np_p2k', 'np_m2k', 'so_p2k', 'so_m2k']

for exp in experiment_names:
    interpolate_to_pressure_levels(exp)
    print(exp)
