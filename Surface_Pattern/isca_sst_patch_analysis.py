#Script to analyse the SST patch experiments run on ISCA

# %%

import xarray as xr
import matplotlib.pyplot as plt
import cftime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from metpy.interpolate import log_interpolate_1d
from SST_patch_analysis_functions import *
from metpy.units import units

# %%
plot_turbulent_flux_decomposition(patch_name='wp', pert_type='warming')
# %%

WP_resp, ctrl, WP_warm = extract_pressure_level_response(patch='wp', pert_type='warming')

# %%
plot_zonal_mean_responses(WP_resp, ctrl, WP_warm)

# %%

WP_resp = extract_response_function(patch_name='wp', pert_type='warming')

plot_TOA_fluxes(WP_resp, title='WP_warming_TOA')
plot_SFC_fluxes(WP_resp, title='WP_warming_SFC')


# %%

SO_warm = extract_response_function(patch_name='so', pert_type='warming')
plot_TOA_fluxes(SO_warm, title='SO_warming_TOA')

# %%
SO_plev_resp, ctrl, SO_plev_warm = extract_pressure_level_response(patch='so', pert_type='warming')

plot_zonal_mean_responses(SO_plev_resp, ctrl, SO_plev_warm)

# %%

SO_cool = extract_response_function(patch_name='so', pert_type='cooling')
plot_TOA_fluxes(SO_cool, title='SO_cooling_TOA')

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