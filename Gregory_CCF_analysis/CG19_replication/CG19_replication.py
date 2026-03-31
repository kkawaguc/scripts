# %%
import xarray as xr
import numpy as np
from functions import *
import matplotlib.pyplot as plt
import pandas as pd

def glob_mean(data):
    lat_weight = np.cos(np.radians(data.lat))
    return data.weighted(lat_weight).mean(('lat', 'lon'))

# %%

z = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/era5_geopotential.nc')
z = z['z'].mean('valid_time').rename({'latitude':'lat', 'longitude':'lon'})
regrid_z = regrid(z)

mask = xr.where(regrid_z > 15, True, False)
# %%
#Import CAM4

CAM4_basedir = "/badc/cmip5/data/cmip5/output1/NCAR/CCSM4/"

experiments = ['amip', 'amip4K']
variable_list = ['tas', 'ts', 'ta', 'ps']

filelist = []

for exp in experiments:
    for var in variable_list:
        if exp == 'amip':
            filelist.append(CAM4_basedir+exp+'/mon/atmos/Amon/r1i1p1/latest/'+var+'/'+var+'_Amon_CCSM4_'+exp+'_r1i1p1_197901-201012.nc')
        else:
            filelist.append(CAM4_basedir+exp+'/mon/atmos/Amon/r1i1p1/latest/'+var+'/'+var+'_Amon_CCSM4_'+exp+'_r1i1p1_197901-200512.nc')
    globals()[f'CAM4_{exp}_data'] = xr.open_mfdataset(filelist, preprocess=regrid)
    filelist = []

CAM4_amip_data = CAM4_amip_data.sel({'plev':70000}).isel({'time':slice(None, 324)})
CAM4_amip4K_data = CAM4_amip4K_data.sel({'plev':70000})

CAM4_amip_data['eis'] = estimated_inversion_strength(CAM4_amip_data['ts'], CAM4_amip_data['ta'], CAM4_amip_data['ps']/100)
CAM4_amip4K_data['eis'] = estimated_inversion_strength(CAM4_amip4K_data['ts'], CAM4_amip4K_data['ta'], CAM4_amip4K_data['ps']/100)

CAM4_amip_data['lts'] = lower_tropospheric_stability(CAM4_amip_data['ts'], CAM4_amip_data['ta'], CAM4_amip_data['ps']/100)
CAM4_amip4K_data['lts'] = lower_tropospheric_stability(CAM4_amip4K_data['ts'], CAM4_amip4K_data['ta'], CAM4_amip4K_data['ps']/100)

CAM4_amip_data['eislts'] = xr.where(mask, CAM4_amip_data['lts'], CAM4_amip_data['eis'])
CAM4_amip4K_data['eislts'] = xr.where(mask, CAM4_amip4K_data['lts'], CAM4_amip4K_data['eis'])

CAM4_data = (CAM4_amip4K_data - CAM4_amip_data).mean('time')/(glob_mean(CAM4_amip4K_data['tas']) - glob_mean(CAM4_amip_data['tas'])).mean('time')
# %%

#Import CMIP6 experiments
variable_list = ['tas', 'ts', 'ta', 'ps']
model_list = {'CESM2':['r1i1p1f1', 'gn.nc'], 
              'CNRM-CM6-1':['r1i1p1f2', 'gr.nc'], 
              'CanESM5':['r1i1p2f1', 'gn.nc'], 
              'HadGEM3-GC31-LL':['r5i1p1f3', 'gn.nc'], 
              'IPSL-CM6A-LR': ['r1i1p1f1', 'gr.nc'], 
              'MRI-ESM2-0':['r1i1p1f1', 'gn.nc']}
experiments = ['amip', 'amip-p4K']

filelist = []

for model in model_list:
    for exp in experiments:
        for var in variable_list:
            filelist.append('/gws/ssde/j25a/csgap/shared/cmip6/'+var+'/'+var+'_Amon_'+model+'_'+exp+'_'+model_list[model][0]+'_'+model_list[model][1])
        temp = xr.open_mfdataset(filelist, preprocess=regrid)
        temp = temp.sel({'plev':70000, 'time':slice('1979-01-01', '2014-01-01')})
        temp['eis'] = estimated_inversion_strength(temp['ts'], temp['ta'], temp['ps']/100)
        temp['lts'] = lower_tropospheric_stability(temp['ts'], temp['ta'], temp['ps']/100)
        temp['eislts'] = xr.where(mask, temp['lts'], temp['eis'])
        globals()[f'{exp.replace('-', '_')}_data'] = temp
        filelist = []
    globals()[f'{model.replace('-', '_')}_data'] = (amip_p4K_data - amip_data).mean('time')/(glob_mean(amip_p4K_data['tas']) - glob_mean(amip_data['tas'])).mean('time')

# %%

full_data = xr.concat([CAM4_data, CESM2_data, CNRM_CM6_1_data, CanESM5_data, HadGEM3_GC31_LL_data, IPSL_CM6A_LR_data, MRI_ESM2_0_data], 
                      dim=pd.Index(['CCSM4', 'CESM2', 'CNRM-CM6-1', 'CanESM5', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MRI-ESM2-0'], name='model'), coords='minimal', compat='override')

full_data.to_netcdf('p4K_data.nc')