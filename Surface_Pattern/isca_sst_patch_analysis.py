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
patches = ['wp', 'ep', 'np', 'so50']

for patch in patches:
    plev_resp, plev_ctrl, plev_pert = extract_pressure_level_response(patch=patch, pert_type='warming')
    plot_turbulent_flux_decomposition(patch_name=patch, pert_type='warming')
    plot_zonal_mean_responses(plev_resp, plev_ctrl, plev_pert, title=patch+'_warming')
    resp = extract_response_function(patch_name=patch, pert_type='warming')
    plot_TOA_fluxes(resp, title=patch+'_warming')
    plot_SFC_fluxes(resp, title=patch+'_warming')

# %%
