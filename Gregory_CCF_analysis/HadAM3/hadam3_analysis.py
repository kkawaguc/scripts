# %%

import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

dir_list = os.listdir('/gws/ssde/j25a/csgap/kkawaguchi/JG_HadAM3')
HadAM3_model_list = [item[21:-3] for item in dir_list if 'EIS' in item]

# %%
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ts_data = xr.open_mfdataset('/gws/ssde/j25a/csgap/kkawaguchi/JG_HadAM3/ts*.nc', concat_dim=[pd.Index(HadAM3_model_list, name='model')],combine='nested', decode_times=time_coder)
ts_data = ts_data['data']
EIS_data = xr.open_mfdataset('/gws/ssde/j25a/csgap/kkawaguchi/JG_HadAM3/EIS*.nc', concat_dim=[pd.Index(HadAM3_model_list, name='model')],combine='nested', decode_times=time_coder)
EIS_data = EIS_data['data']

HadAM3_data = xr.concat([ts_data, EIS_data], dim=pd.Index(['ts', 'eislts'], name='var')).rename({'latitude':'lat_in', 'longitude':'lon_in'})
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

combined_model_list = [mod_name for mod_name in HadAM3_model_list if mod_name in model_names]

var_names = ['ts', 'eislts', 'hur700', 'wap500', 'utrh']

lat_coords = np.linspace(-87.5, 87.5, 36)
lon_coords = np.linspace(-177.5, 177.5, 72)

lw_coeffs = lw_coeffs['coeffs'].assign_coords({'model_nr':model_names, 'var':var_names, 'lat_in':lat_coords, 'lon_in':lon_coords, 
                                     'lat_out':lat_coords, 'lon_out':lon_coords})
sw_coeffs = sw_coeffs['coeffs'].assign_coords({'model_nr':model_names, 'var':var_names, 'lat_in':lat_coords, 'lon_in':lon_coords, 
                                     'lat_out':lat_coords, 'lon_out':lon_coords})

lw_coeffs = lw_coeffs.rename({'model_nr':'model'})
sw_coeffs = sw_coeffs.rename({'model_nr':'model'})
# %%

#Import the model feedbacks from CMIP5
cmip5_basedir = os.listdir('/gws/ssde/j25a/csgap/ceppi/data/cmip5/5x5/feedbacks/')

cmip5_mod_list = []
cmip5_mod_filenames = []

for item in cmip5_basedir:
    if item[-8:] == '_1-20.nc' and item[:6]!='abrupt':
        cmip5_mod_list.append(item[:-8])
        cmip5_mod_filenames.append('/gws/ssde/j25a/csgap/ceppi/data/cmip5/5x5/feedbacks/'+item)

CMIP5_model_feedbacks = xr.open_mfdataset(cmip5_mod_filenames, combine='nested', concat_dim=[pd.Index(cmip5_mod_list, name='model')], 
                                          coords="minimal", compat="override",combine_attrs = "drop")


cmip6_basedir = os.listdir('/gws/ssde/j25a/csgap/ceppi/data/cmip6/5x5/feedbacks/')

cmip6_mod_list = []
cmip6_mod_filenames = []

for item in cmip6_basedir:
    if item[-8:] == '_1-20.nc' and item[:6]!='abrupt':
        cmip6_mod_list.append(item[:-8])
        cmip6_mod_filenames.append('/gws/ssde/j25a/csgap/ceppi/data/cmip6/5x5/feedbacks/'+item)

CMIP6_model_feedbacks = xr.open_mfdataset(cmip6_mod_filenames, combine='nested', concat_dim=[pd.Index(cmip6_mod_list, name='model')], 
                                          coords="minimal", compat="override",combine_attrs = "drop")

model_feedbacks = xr.concat([CMIP5_model_feedbacks, CMIP6_model_feedbacks], dim='model')
model_fdbks = xr.concat([CMIP5_model_feedbacks, CMIP6_model_feedbacks], dim='model')
model_fdbks = model_fdbks.sel({'model':combined_model_list})

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
                                  concat_dim=[pd.Index(cmip5_std_list, name='model')], coords="minimal", 
                                  compat="override",combine_attrs = "drop")


cmip6_std_dir = os.listdir('/gws/ssde/j25a/csgap/ceppi/data/cmip6/5x5/sd/')

cmip6_std_list = []
cmip6_std_filenames = []

for item in cmip6_std_dir:
    if 'historical-ssp' in item and 'anom' not in item:
        cmip6_std_list.append(item[:-18])
        cmip6_std_filenames.append('/gws/ssde/j25a/csgap/ceppi/data/cmip6/5x5/sd/'+item)

CMIP6_CCF_std = xr.open_mfdataset(cmip6_std_filenames, combine = 'nested', 
                                  concat_dim=[pd.Index(cmip6_std_list, name="model")], coords="minimal", 
                                  compat="override",combine_attrs = "drop")
# %%

CCF_std = xr.concat([CMIP5_CCF_std, CMIP6_CCF_std], dim='model')
CCF_std = CCF_std[['ts', 'eislts', 'hur700', 'wap500', 'utrh']].to_dataarray('var').rename({'lat':'lat_in', 'lon':'lon_in'})

def glob_mean(data, bnds=90):
    if 'lat_out' in data.dims:
        lat_weights = np.cos(np.radians(data.lat_out))
        return data.sel({'lat':slice(-bnds, bnds)}).weighted(lat_weights).mean(('lat_out', 'lon_out'))
    else:
        lat_weights = np.cos(np.radians(data.lat))
        return data.sel({'lat':slice(-bnds, bnds)}).weighted(lat_weights).mean(('lat', 'lon'))

# Calculate the sensitivities in physical units:

SW_coeffs_physical=(-sw_coeffs/CCF_std).sel({'model':combined_model_list})
LW_coeffs_physical=(lw_coeffs/CCF_std).sel({'model':combined_model_list})

SW_coeffs_physical_MIROC_ES2L = (-sw_coeffs/CCF_std).sel({'model':'MIROC-ES2L'})
LW_coeffs_physical_MIROC_ES2L = (lw_coeffs/CCF_std).sel({'model':'MIROC-ES2L'})
# %%
HadAM3_data = HadAM3_data.sel({'model':combined_model_list})

SW_full_reconstruction = (HadAM3_data * SW_coeffs_physical).sum(('lat_in', 'lon_in'))
LW_full_reconstruction = (HadAM3_data * LW_coeffs_physical).sum(('lat_in', 'lon_in'))

SW_dCCF = (HadAM3_data * SW_coeffs_physical.mean('model')).sum(('lat_in', 'lon_in'))
LW_dCCF = (HadAM3_data * LW_coeffs_physical.mean('model')).sum(('lat_in', 'lon_in'))

SW_dtheta = (HadAM3_data.mean('model') * SW_coeffs_physical).sum(('lat_in', 'lon_in'))
LW_dtheta = (HadAM3_data.mean('model') * LW_coeffs_physical).sum(('lat_in', 'lon_in'))

def global_mean_plot(SW_full, LW_full, var=['ts', 'eislts'], name = ''):
    fig, ax = plt.subplots(nrows=1, ncols=3, layout='constrained', figsize=(9,3))

    for i in range(3):
        if i == 0:
            estimate_data_full = SW_full + LW_full
            model_fdbk_data = model_fdbks['lwcld'] +model_fdbks['swcld']
            ax[0].set_title('Net')
            ax[0].set_ylabel('Predicted Cloud Feedback ($\\mathrm{W/m^2/K}$)')
        if i == 1:
            estimate_data_full = SW_full 
            model_fdbk_data = model_fdbks['swcld'] 
            ax[1].set_title('SW')
            ax[1].set_yticks([])
        if i == 2:
            estimate_data_full = LW_full 
            model_fdbk_data = model_fdbks['lwcld']
            ax[2].set_title('LW')
            ax[2].set_yticks([])
        #Calculate global_means
        if type(var) == list:
            estimate_data_full = glob_mean(estimate_data_full).sel({'var':var}).sum('var')
        else:
            estimate_data_full = glob_mean(estimate_data_full).sel({'var':var})
        model_fdbk_data = glob_mean(model_fdbk_data)
        #Scatterplot the data
        ax[i].scatter(model_fdbk_data.sel({'model':combined_model_list}), estimate_data_full, c='k')
        
        #Calculate the linear regressions
        full_net = sp.stats.linregress(model_fdbk_data.sel({'model':combined_model_list}), estimate_data_full)

        x_vals = np.linspace(-1, 2, 10)
        ax[i].plot(x_vals, full_net.slope * x_vals + full_net.intercept, 'k')
        ax[i].plot((-1, 2), (-1, 2), color='k', ls=':')
        ax[i].set_ylim([-1, 1.5])
        ax[i].set_xlim([-1, 1.5])

        ax[i].text(-0.8, 1.3, 'r='+str(np.round(full_net.rvalue, 2))+', m='+str(np.round(full_net.slope, 2)))

        ax[i].set_xlabel('True Cloud Feedback ($\\mathrm{W/m^2/K}$)')
    fig.suptitle(name)
    fig.savefig(name+'_global_mean.png')
    return None

global_mean_plot(SW_full_reconstruction, LW_full_reconstruction, name='HadAM3_full_reconstruction')
global_mean_plot(SW_dCCF, LW_dCCF, name='HadAM3_dCCF')
global_mean_plot(SW_dtheta, LW_dtheta, name='HadAM3_dTheta')

SW_dCCF_NorESM2 = (HadAM3_data * SW_coeffs_physical.sel({'model':'NorESM2-LM'})).sum(('lat_in', 'lon_in'))
LW_dCCF_NorESM2 = (HadAM3_data * LW_coeffs_physical.sel({'model':'NorESM2-LM'})).sum(('lat_in', 'lon_in'))
global_mean_plot(SW_dCCF_NorESM2, LW_dCCF_NorESM2, name='HadAM3_dCCF_NorESM2')

SW_dCCF_MRI = (HadAM3_data * SW_coeffs_physical_MIROC_ES2L).sum(('lat_in', 'lon_in'))
LW_dCCF_MRI = (HadAM3_data * LW_coeffs_physical_MIROC_ES2L).sum(('lat_in', 'lon_in'))
global_mean_plot(SW_dCCF_MRI, LW_dCCF_MRI, name='HadAM3_dCCF_MIROC_ES2L')


