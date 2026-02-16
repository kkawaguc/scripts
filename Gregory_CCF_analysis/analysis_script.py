# %%

import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy as sp

# %%
# Plotting routine

def global_mean_plot(SW_full, LW_full, SW_sub, LW_sub, var=['ts', 'eislts'], name = ''):
    fig, ax = plt.subplots(nrows=1, ncols=3, layout='constrained', figsize=(9,3))

    for i in range(3):
        if i == 0:
            estimate_data_full = SW_full + LW_full
            estimate_data_sub = SW_sub + LW_sub
            model_fdbk_data = model_fdbks['lwcld'] +model_fdbks['swcld']
            ax[0].set_title('Net')
            ax[0].set_ylabel('Predicted Cloud Feedback ($\\mathrm{W/m^2/K}$)')
        if i == 1:
            estimate_data_full = SW_full 
            estimate_data_sub = SW_sub
            model_fdbk_data = model_fdbks['swcld'] 
            ax[1].set_title('SW')
            ax[1].set_yticks([])
        if i == 2:
            estimate_data_full = LW_full 
            estimate_data_sub = LW_sub
            model_fdbk_data = model_fdbks['lwcld']
            ax[2].set_title('LW')
            ax[2].set_yticks([])
        #Calculate global_means
        if type(var) == list:
            estimate_data_full = glob_mean(estimate_data_full).sel({'var':var}).sum('var')
            estimate_data_sub = glob_mean(estimate_data_sub).sel({'var':var}).sum('var')
        else:
            estimate_data_full = glob_mean(estimate_data_full).sel({'var':var})
            estimate_data_sub = glob_mean(estimate_data_sub).sel({'var':var})
        model_fdbk_data = glob_mean(model_fdbk_data)
        #Scatterplot the data
        ax[i].scatter(model_fdbk_data.sel({'atmos_mod':model_names}), estimate_data_full, c='k')
        ax[i].scatter(model_fdbk_data.sel({'atmos_mod':pattern_models}), estimate_data_sub, c='r')
        
        #Calculate the linear regressions
        full_net = sp.stats.linregress(model_fdbk_data.sel({'atmos_mod':model_names}), estimate_data_full)
        sub_net = sp.stats.linregress(model_fdbk_data.sel({'atmos_mod':pattern_models}), estimate_data_sub)

        x_vals = np.linspace(-1, 2, 10)
        ax[i].plot(x_vals, full_net.slope * x_vals + full_net.intercept, 'k')
        ax[i].plot(x_vals, sub_net.slope * x_vals + sub_net.intercept, 'r')
        ax[i].plot((-1, 2), (-1, 2), color='k', ls=':')
        ax[i].set_ylim([-1, 1.5])
        ax[i].set_xlim([-1, 1.5])

        ax[i].text(-0.8, 1.3, 'r='+str(np.round(full_net.rvalue, 2))+', m='+str(np.round(full_net.slope, 2)))
        ax[i].text(-0.8, 1.1, 'r='+str(np.round(sub_net.rvalue, 2))+', m='+str(np.round(sub_net.slope, 2)), color='r')

        ax[i].set_xlabel('True Cloud Feedback ($\\mathrm{W/m^2/K}$)')
    fig.suptitle(name)
    fig.savefig(name+'_global_mean.png')
    return None

def spatial_plot(SW_full, LW_full, name=''):
    model_fdbk_data = glob_mean(model_fdbks['lwcld'] +model_fdbks['swcld']).sel({'atmos_mod':model_names})
    estimate_data = (SW_full + LW_full).pad({'lon_out':2}, mode='wrap').assign_coords({'lon_out':np.linspace(-187.5, 187.5, 76)})
    if 'model_nr' in estimate_data.dims:
        estimate_data = estimate_data.assign_coords({'fdbk':model_fdbk_data.rename({'atmos_mod':'model_nr'})})
    else:
        estimate_data = estimate_data.assign_coords({'fdbk':model_fdbk_data})
    print(estimate_data.coords)
    polyfit = estimate_data.polyfit('fdbk', 1)

    slopes = polyfit['polyfit_coefficients'].sel({'degree':1})
    print(slopes.coords)
    cbar_max = float(np.round(max(slopes.max(), abs(slopes.min())), 1)) + 0.1

    if 'model_nr' in estimate_data.dims:
        corr = xr.corr(model_fdbk_data, estimate_data.sum('var').rename({'model_nr':'atmos_mod'}), dim='atmos_mod')
    else:
        corr = xr.corr(model_fdbk_data, estimate_data.sum('var'), dim='atmos_mod')
    
    fig, ax = plt.subplots(nrows = 2, ncols = 3, layout='constrained', figsize=(15, 6), subplot_kw={'projection':ccrs.Robinson(central_longitude=210)})
    corr.plot.contourf(ax=ax[0, 0], transform=ccrs.PlateCarree(), vmin=-1, vmax=1, 
    levels=[-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8], cmap='PuOr',cbar_kwargs={'location':'left'})
    for i in range(5):
        cbar = slopes.isel({'var':i}).plot.contourf(ax=ax[(i+1)//3, (i+1) % 3], transform=ccrs.PlateCarree(), vmin=-cbar_max, vmax=cbar_max,
        levels=12, extend='both', cmap='bwr', add_colorbar=False)
    fig.colorbar(cbar, ax=ax[:,:], orientation = 'horizontal')

    ax[0,0].set_title('Fdbk Correlation')
    ax[0,1].set_title('ts')
    ax[0,2].set_title('eis')
    ax[1,0].set_title('RH700')
    ax[1,1].set_title('$\\omega$500')
    ax[1,2].set_title('UTRH')

    for i in range(6):
        ax[i//3, i%3].coastlines()
    fig.savefig(name+'_spatial.png')
    return None



# %%

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

sw_coeffs['coeffs'].sum(('lat_in', 'lon_in')).mean('model_nr').plot(col='var')

# %%

#Import the model feedbacks from CMIP5
cmip5_basedir = os.listdir('/gws/ssde/j25a/csgap/ceppi/data/cmip5/5x5/feedbacks/')

cmip5_mod_list = []
cmip5_mod_filenames = []

for item in cmip5_basedir:
    if item[-8:] == '_1-20.nc' and item[:6]!='abrupt':
        cmip5_mod_list.append(item[:-8])
        cmip5_mod_filenames.append('/gws/ssde/j25a/csgap/ceppi/data/cmip5/5x5/feedbacks/'+item)

CMIP5_model_feedbacks = xr.open_mfdataset(cmip5_mod_filenames, combine='nested', concat_dim=[pd.Index(cmip5_mod_list, name="atmos_mod")], 
                                          coords="minimal", compat="override",combine_attrs = "drop")


cmip6_basedir = os.listdir('/gws/ssde/j25a/csgap/ceppi/data/cmip6/5x5/feedbacks/')

cmip6_mod_list = []
cmip6_mod_filenames = []

for item in cmip6_basedir:
    if item[-8:] == '_1-20.nc' and item[:6]!='abrupt':
        cmip6_mod_list.append(item[:-8])
        cmip6_mod_filenames.append('/gws/ssde/j25a/csgap/ceppi/data/cmip6/5x5/feedbacks/'+item)

CMIP6_model_feedbacks = xr.open_mfdataset(cmip6_mod_filenames, combine='nested', concat_dim=[pd.Index(cmip6_mod_list, name="atmos_mod")], 
                                          coords="minimal", compat="override",combine_attrs = "drop")

model_feedbacks = xr.concat([CMIP5_model_feedbacks, CMIP6_model_feedbacks], dim='atmos_mod')

# %%
#Import the model standard deviations
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
# %%

# Concatenate the model feedbacks and the standard deviations
model_fdbks = xr.concat([CMIP5_model_feedbacks, CMIP6_model_feedbacks], dim='atmos_mod')
model_fdbks = model_fdbks.sel({'atmos_mod':model_names})
model_fdbks_CCF = model_fdbks[['ts', 'eislts', 'hur700', 'wap500', 'utrh']].to_dataarray('var')

CCF_std = xr.concat([CMIP5_CCF_std, CMIP6_CCF_std], dim='atmos_mod')
CCF_std = CCF_std[['ts', 'eislts', 'hur700', 'wap500', 'utrh']].to_dataarray('var').rename({'lat':'lat_in', 'lon':'lon_in'})
CCF_std = CCF_std.rename({'atmos_mod':'model_nr'})
# %%

# Calculate the sensitivities in physical units:

SW_coeffs_physical=(-sw_coeffs['coeffs']/CCF_std)
LW_coeffs_physical=(lw_coeffs['coeffs']/CCF_std)

pattern_models = ['CanESM5', 'CESM2', 'CNRM-ESM2-1', 'GFDL-CM4', 
                  'HadGEM3-GC31-LL', 'MIROC6', 'NorESM2-LM']

# %%
def glob_mean(data):
    if 'lat_out' in data.dims:
        lat_weights = np.cos(np.radians(data.lat_out))
        return data.weighted(lat_weights).mean(('lat_out', 'lon_out'))
    else:
        lat_weights = np.cos(np.radians(data.lat))
        return data.weighted(lat_weights).mean(('lat', 'lon'))

# %%

# Initial testing: 

SW_CN21_fdbk = (SW_coeffs_physical * model_fdbks_CCF.rename({'atmos_mod':'model_nr', 'lat':'lat_in', 'lon':'lon_in'})).sum(('lat_in', 'lon_in'))
LW_CN21_fdbk = (LW_coeffs_physical * model_fdbks_CCF.rename({'atmos_mod':'model_nr', 'lat':'lat_in', 'lon':'lon_in'})).sum(('lat_in', 'lon_in'))

SW_CN21_fdbk_subset = SW_CN21_fdbk.sel({'model_nr':pattern_models})
LW_CN21_fdbk_subset = LW_CN21_fdbk.sel({'model_nr':pattern_models})

global_mean_plot(SW_CN21_fdbk, LW_CN21_fdbk, SW_CN21_fdbk_subset, LW_CN21_fdbk_subset, var='ts', name='CN21_replication_ts')
global_mean_plot(SW_CN21_fdbk, LW_CN21_fdbk, SW_CN21_fdbk_subset, LW_CN21_fdbk_subset, var=['ts', 'eislts'], name='CN21_replication')        
global_mean_plot(SW_CN21_fdbk, LW_CN21_fdbk, SW_CN21_fdbk_subset, LW_CN21_fdbk_subset, var=['ts', 'eislts', 'hur700', 'utrh', 'wap500'], name='CN21_replication_all_vars') 

spatial_plot(SW_CN21_fdbk, LW_CN21_fdbk, name='dtheta_dCCF')

# %%

# Calculate assuming mean theta with dCCF

SW_dCCF_fdbk = (SW_coeffs_physical.mean('model_nr') * model_fdbks_CCF.rename({'lat':'lat_in', 'lon':'lon_in'})).sum(('lat_in', 'lon_in'))
LW_dCCF_fdbk = (LW_coeffs_physical.mean('model_nr') * model_fdbks_CCF.rename({'lat':'lat_in', 'lon':'lon_in'})).sum(('lat_in', 'lon_in'))

SW_dCCF_fdbk_subset = SW_dCCF_fdbk.sel({'atmos_mod':pattern_models})
LW_dCCF_fdbk_subset = LW_dCCF_fdbk.sel({'atmos_mod':pattern_models})

global_mean_plot(SW_dCCF_fdbk, LW_dCCF_fdbk, SW_dCCF_fdbk_subset, LW_dCCF_fdbk_subset, var = 'ts', name='dCCF_ts')
global_mean_plot(SW_dCCF_fdbk, LW_dCCF_fdbk, SW_dCCF_fdbk_subset, LW_dCCF_fdbk_subset, var = ['ts', 'eislts'], name='dCCF_ts_eis')
global_mean_plot(SW_dCCF_fdbk, LW_dCCF_fdbk, SW_dCCF_fdbk_subset, LW_dCCF_fdbk_subset, var=['ts', 'eislts', 'hur700', 'utrh', 'wap500'], name='dCCF_all_vars')

spatial_plot(SW_dCCF_fdbk, LW_dCCF_fdbk, name='dCCF')

# %%
# Calculate assuming dtheta and mean CCF  

SW_dtheta_fdbk = (SW_coeffs_physical * model_fdbks_CCF.mean('atmos_mod').rename({'lat':'lat_in', 'lon':'lon_in'})).sum(('lat_in', 'lon_in'))
LW_dtheta_fdbk = (LW_coeffs_physical * model_fdbks_CCF.mean('atmos_mod').rename({'lat':'lat_in', 'lon':'lon_in'})).sum(('lat_in', 'lon_in'))

SW_dtheta_fdbk_subset = SW_dtheta_fdbk.sel({'model_nr':pattern_models})
LW_dtheta_fdbk_subset = LW_dtheta_fdbk.sel({'model_nr':pattern_models})

global_mean_plot(SW_dtheta_fdbk, LW_dtheta_fdbk, SW_dtheta_fdbk_subset, LW_dtheta_fdbk_subset, var = 'ts', name='dtheta_ts')
global_mean_plot(SW_dtheta_fdbk, LW_dtheta_fdbk, SW_dtheta_fdbk_subset, LW_dtheta_fdbk_subset, var = ['ts', 'eislts'], name='dtheta_ts_eis')
global_mean_plot(SW_dtheta_fdbk, LW_dtheta_fdbk, SW_dtheta_fdbk_subset, LW_dtheta_fdbk_subset, var = ['ts', 'eislts', 'hur700', 'utrh', 'wap500'], name='dtheta_all_vars')

spatial_plot(SW_dtheta_fdbk, LW_dtheta_fdbk, name='dtheta')
# %%
