# %%
import xarray as xr
import numpy as np
import os
import pandas as pd
import cartopy.crs as ccrs

# %%

p4K_data = xr.open_dataset('p4K_data.nc')
p4K_data = p4K_data.rename({'lat':'lat_in', 'lon':'lon_in'})

#Import and assign correct coordinates to the SW/LW coefficients

rsdt = xr.open_dataset('/gws/ssde/j25a/csgap/ceppi/data/cmip6/5x5/rsdt/rsdt_Amon_HadGEM3-GC31-LL_piClim-control_r1i1p1f3_gn.nc')
rsdt = rsdt['rsdt'].mean('time').rename({'lat':'lat_in', 'lon':'lon_in'})

lw_coeffs = xr.open_dataset('/gws/ssde/j25a/csgap/public/ceppi/pnas_2021/coeffs/coeffs_lw.nc')
sw_coeffs = xr.open_dataset('/gws/ssde/j25a/csgap/public/ceppi/pnas_2021/coeffs/coeffs_sw.nc')

lw_coeffs = lw_coeffs.isel({'model_nr':slice(4, None)})
sw_coeffs = sw_coeffs.isel({'model_nr':slice(4, None)}) * rsdt

model_names = ['ACCESS-CM2','ACCESS-ESM1-5','ACCESS1-0','ACCESS1-3','BCC-CSM2-MR','BCC-ESM1','BNU-ESM','CCSM4',
               'CESM2-WACCM','CESM2','CNRM-CM5','CNRM-CM6-1','CNRM-ESM2-1','CSIRO-Mk3-6-0','CanESM2','CanESM5',
               'EC-Earth3-Veg','FGOALS-f3-L','FGOALS-g3','GFDL-CM3','GFDL-CM4','GFDL-ESM2G','GFDL-ESM2M',
               'GISS-E2-1-G','GISS-E2-H','GISS-E2-R','HadGEM2-ES','HadGEM3-GC31-LL','INM-CM4-8','INM-CM5-0',
               'IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM5B-LR','IPSL-CM6A-LR','MIROC-ES2L','MIROC-ESM','MIROC5',
               'MIROC6','MPI-ESM-LR','MPI-ESM-MR','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-CGCM3','MRI-ESM2-0',
               'NESM3','NorESM1-M','NorESM2-LM','NorESM2-MM','UKESM1-0-LL','bcc-csm1-1-m','bcc-csm1-1','inmcm4']

var_names = ['ts', 'eislts', 'hur700', 'wap500', 'utrh']

lat_coords = np.linspace(-87.5, 87.5, 36)
lon_coords = np.linspace(-177.5, 177.5, 72)

lw_coeffs = lw_coeffs.assign_coords({'model_nr':model_names, 'var':var_names, 'lat_in':lat_coords, 'lon_in':lon_coords, 
                                     'lat_out':lat_coords, 'lon_out':lon_coords})
sw_coeffs = sw_coeffs.assign_coords({'model_nr':model_names, 'var':var_names, 'lat_in':lat_coords, 'lon_in':lon_coords, 
                                     'lat_out':lat_coords, 'lon_out':lon_coords})

lw_coeffs = lw_coeffs.rename({'model_nr':'model'})
sw_coeffs = sw_coeffs.rename({'model_nr':'model'})

# Import model SDs
cmip5_std_dir = os.listdir('/gws/ssde/j25a/csgap/ceppi/data/cmip5/5x5/sd/')

cmip5_std_list = []
cmip5_std_filenames = []

for item in cmip5_std_dir:
    if 'historical-rcp45' in item and 'anom' not in item:
        cmip5_std_list.append(item[:-20])
        cmip5_std_filenames.append('/gws/ssde/j25a/csgap/ceppi/data/cmip5/5x5/sd/'+item)

CMIP5_CCF_std = xr.open_mfdataset(cmip5_std_filenames, combine = 'nested', 
                                  concat_dim=[pd.Index(cmip5_std_list, name="atmos_mod")], coords="minimal", 
                                  compat="override",combine_attrs = "drop")

cmip6_std_dir = os.listdir('/gws/ssde/j25a/csgap/ceppi/data/cmip6/5x5/sd/')

cmip6_std_list = []
cmip6_std_filenames = []

for item in cmip6_std_dir:
    if 'historical-ssp' in item and 'anom' not in item:
        cmip6_std_list.append(item[:-18])
        cmip6_std_filenames.append('/gws/ssde/j25a/csgap/ceppi/data/cmip6/5x5/sd/'+item)

CMIP6_CCF_std = xr.open_mfdataset(cmip6_std_filenames, combine = 'nested', 
                                  concat_dim=[pd.Index(cmip6_std_list, name="atmos_mod")], coords="minimal", 
                                  compat="override",combine_attrs = "drop")

CCF_std = xr.concat([CMIP5_CCF_std, CMIP6_CCF_std], dim='atmos_mod')
CCF_std = CCF_std[['ts', 'eislts', 'hur700', 'wap500', 'utrh']].to_dataarray('var').rename({'lat':'lat_in', 'lon':'lon_in'})
CCF_std = CCF_std.rename({'atmos_mod':'model'})

# %%
#Subset to the 7 models we are examining

lw_coeffs = lw_coeffs.sel({'model':['CCSM4', 'CESM2', 'CNRM-CM6-1', 'CanESM5', 
                                       'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MRI-ESM2-0']})
sw_coeffs = sw_coeffs.sel({'model':['CCSM4', 'CESM2', 'CNRM-CM6-1', 'CanESM5', 
                                       'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MRI-ESM2-0']})

CCF_std = CCF_std.sel({'model':['CCSM4', 'CESM2', 'CNRM-CM6-1', 'CanESM5', 
                                   'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MRI-ESM2-0']})

SW_coeffs_physical=(-sw_coeffs['coeffs']/CCF_std)
LW_coeffs_physical=(lw_coeffs['coeffs']/CCF_std)

# %%

local_theta_test = (SW_coeffs_physical+LW_coeffs_physical).sel({'var':'ts'}).sum(('lat_in', 'lon_in'))
fig = local_theta_test.plot(col='model', col_wrap = 3, robust=True,transform=ccrs.PlateCarree(),subplot_kws={'projection':ccrs.Robinson(central_longitude=210)}, aspect=1.6)
for ax in fig.axs.flat:
    ax.coastlines()

theta_weights = np.cos(np.radians(local_theta_test.lat_out))
p4K_weights = np.cos(np.radians(p4K_data.lat_in))

ocean = xr.where(p4K_data['eislts'] == p4K_data['eis'], 1, np.nan)
Sm = (ocean * p4K_data['eis'].sel({'lat_in':slice(-50, 50)})).weighted(p4K_weights).mean(('lat_in', 'lon_in'))

sigma_i_spatial = ((SW_coeffs_physical+LW_coeffs_physical).sel({'var':'eislts'}) * p4K_data['eislts']).sum(('lat_in', 'lon_in'))

sigma_i = sigma_i_spatial.weighted(theta_weights).mean(('lat_out', 'lon_out'))/Sm

tau_i = local_theta_test.weighted(theta_weights).mean(('lat_out', 'lon_out'))

print(Sm)
print(sigma_i.compute())
print(tau_i.compute())

# %%

ts = ((SW_coeffs_physical+LW_coeffs_physical).sel({'var':'ts'}) * p4K_data['ts']).sum(('lat_in', 'lon_in'))
print(ts.weighted(theta_weights).mean(('lat_out', 'lon_out')).compute())

eis = ((SW_coeffs_physical+LW_coeffs_physical).sel({'var':'eislts'}) * p4K_data['eislts']).sum(('lat_in', 'lon_in'))
print(eis.weighted(theta_weights).mean(('lat_out', 'lon_out')).compute())

# %%
