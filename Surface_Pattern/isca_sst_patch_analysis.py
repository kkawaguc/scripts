#Script to analyse the SST patch experiments run on ISCA

# %%

import xarray as xr
import matplotlib.pyplot as plt
import cftime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from metpy.interpolate import log_interpolate_1d
from SST_patch_analysis_functions import *

# %%

experiment_list = ['Greens_function_control', 'wp_p2k', 'wp_m2k', 'ep_p2k', 'ep_m2k', 'np_p2k', 'np_m2k', 'so_p2k', 'so_m2k']

for exp in experiment_list:
    interpolate_to_pressure_levels(exp)
    print(exp)


# %%

SO_warm = extract_response_function(patch_name='so', pert_type='warming')
plot_TOA_fluxes(SO_warm, title='SO_warming_TOA')

# %%

SO_cool = extract_response_function(patch_name='so', pert_type='cooling')
plot_TOA_fluxes(SO_cool, title='SO_cooling_TOA')

# %%

WP_resp = extract_response_function(patch_name='wp', pert_type='symmetric')

plot_TOA_fluxes(WP_resp, title='WP_symm_TOA')
plot_SFC_fluxes(WP_resp, title='WP_symm_SFC')

# %%

EP_resp = extract_response_function(patch_name='ep', pert_type='symmetric')
plot_TOA_fluxes(EP_resp, title='EP_symm_TOA')
plot_SFC_fluxes(EP_resp, title='EP_symm_SFC')

# %%

NP_resp = extract_response_function(patch_name='np', pert_type='symmetric')
plot_TOA_fluxes(NP_resp, title='NP_symm_TOA')
plot_SFC_fluxes(NP_resp, title='NP_symm_SFC')
# %%

SO_resp = extract_response_function(patch_name='so', pert_type='symmetric')
plot_TOA_fluxes(SO_resp, title='SO_symm_TOA')
plot_SFC_fluxes(SO_resp, title='SO_symm_SFC')