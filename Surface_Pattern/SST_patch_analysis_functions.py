import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from metpy.interpolate import log_interpolate_1d
import numpy as np
import metpy.units as units

def extract_response_function(patch_name='wp', pert_type='symmetric'):
    '''Imports the response to a given patch location
    and whether the perturbation is warming, cooling or symmetric response.

    Removes the spin-up year, then calculates the time-mean response.
    Inputs:
        patch_name: must be one of wp, ep, np or so
        pert_type: must be one of symmetric, warming or cooling
    Returns:
        resp: the response function'''
    if pert_type == 'symmetric':
        p2K = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+patch_name+'_p2k/run*/atmos_monthly.nc',
                         use_cftime=True, engine='netcdf4')
        m2K = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+patch_name+'_m2k/run*/atmos_monthly.nc',
                         use_cftime=True, engine='netcdf4')
        p2K = p2K.isel({'time':slice(12, None)}).mean('time')
        m2K = m2K.isel({'time':slice(12, None)}).mean('time')
        resp = (p2K - m2K)/2
    elif pert_type == 'cooling':
        ctrl = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/Greens_function_control/run*/atmos_monthly.nc',
                         use_cftime=True, engine='netcdf4')
        m2K = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+patch_name+'_m2k/run*/atmos_monthly.nc',
                         use_cftime=True, engine='netcdf4')
        ctrl = ctrl.isel({'time':slice(12, None)}).mean('time')
        m2K = m2K.isel({'time':slice(12, None)}).mean('time')
        resp = ctrl - m2K
    elif pert_type == 'warming':
        p2K = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+patch_name+'_p2k/run*/atmos_monthly.nc',
                         use_cftime=True, engine='netcdf4')
        ctrl = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/Greens_function_control/run*/atmos_monthly.nc',
                         use_cftime=True, engine='netcdf4')
        p2K = p2K.isel({'time':slice(12, None)}).mean('time')
        ctrl = ctrl.isel({'time':slice(12, None)}).mean('time')
        resp = p2K - ctrl
    else:
        print('Error: perturbation type must be warming, cooling or symmetric')
        return None
    return resp

def plot_TOA_fluxes(resp, title=''):
    '''Plots the TOA response to the patch perturbation'''
    if title=='':
        print('Error: Please enter plot title')
        return None

    fig, ax = plt.subplots(3, 1, layout='constrained', subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    (resp['temp'].isel({'pfull':-1})).plot(ax=ax[0], vmin=-2, vmax=2, cmap='RdBu_r', transform=ccrs.PlateCarree(), cbar_kwargs={'label':'$\\Delta T$ (Lowest model level)'})
    (resp['soc_toa_sw'] - resp['soc_olr']).plot(ax=ax[1], vmin=-20, vmax=20, cmap='RdBu_r', transform=ccrs.PlateCarree(), cbar_kwargs={'label':'TOA Resp ($\\mathrm{W/m^2}$)'})
    (resp['low_cld_amt']).plot(ax=ax[2], vmin=-10, vmax=10, cmap='RdBu_r', transform=ccrs.PlateCarree(),cbar_kwargs={'label':'LCC (%)'})

    for i in range(3):
        ax[i].coastlines()
    ax[0].set_title(title)
    fig.savefig('Plots/Patch_experiments/TOA_'+title+'.png')
    plt.show()
    return None

def plot_SFC_fluxes(resp, title=''):
    '''Plots the surface flux response to the patch perturbation'''
    if title=='':
        print('Error: Please enter plot title')
        return None
    fig, ax = plt.subplots(3, 2, figsize=(12, 8), layout='constrained', subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    (resp['soc_surf_flux_lw'] + resp['soc_surf_flux_sw'] - 
     resp['flux_lhe'] - resp['flux_t']).plot(ax=ax[0,0], transform=ccrs.PlateCarree(), vmin=-25, vmax=25, cmap='RdBu_r', cbar_kwargs={'label':'Net ($\\mathrm{W/m^2}$)'})

    (resp['soc_surf_flux_lw_clr'] + resp['soc_surf_flux_sw_clr']).plot(ax=ax[1,0], transform=ccrs.PlateCarree(), vmin=-25, 
                                                                       vmax=25, cmap='RdBu_r', cbar_kwargs={'label':'Rad Clear ($\\mathrm{W/m^2}$)'})

    (resp['soc_surf_flux_lw'] - resp['soc_surf_flux_lw_clr'] 
     + resp['soc_surf_flux_sw'] - resp['soc_surf_flux_sw_clr']).plot(ax=ax[2,0], transform=ccrs.PlateCarree(), vmin=-25, 
                                                                     vmax=25, cmap='RdBu_r', cbar_kwargs={'label':'CRE ($\\mathrm{W/m^2}$)'})

    (-resp['flux_lhe']).plot(ax=ax[0,1], transform=ccrs.PlateCarree(), vmin=-25, vmax=25, cmap='RdBu_r',cbar_kwargs={'label':'LH ($\\mathrm{W/m^2}$)'})
    (-resp['flux_t']).plot(ax=ax[1,1], transform=ccrs.PlateCarree(), vmin=-25, vmax=25, cmap='RdBu_r', cbar_kwargs={'label':'SH ($\\mathrm{W/m^2}$)'})
    (-resp['flux_lhe'] - resp['flux_t']).plot(ax=ax[2,1], transform=ccrs.PlateCarree(), vmin=-25, vmax=25, cmap='RdBu_r', cbar_kwargs={'label':'LH + SH ($\\mathrm{W/m^2}$)'})

    for i in range(6):
        ax[i//2, i%2].coastlines()

    fig.suptitle(title, fontsize=18)
    fig.savefig('Plots/Patch_experiments/SFC_'+title+'.png')
    plt.show()
    return None

def sel_pressure_data(dataset):
    return dataset[['ucomp', 'vcomp', 'omega', 'temp', 'ps']]

def interpolate_to_pressure_levels(experiment_name='Greens_function_control'):
    '''Interpolate data to standard pressure levels using metpy. 
    Saves as one large netcdf file.'''
    
    raw_data = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+experiment_name+'/run*/atmos_monthly.nc',
                                 use_cftime=True, preprocess=sel_pressure_data, engine='netcdf4')
    sigma_pressure_locations = (raw_data.pfull / 1000) * raw_data['ps']
    sigma_pressure_locations.attrs['units'] = 'Pa'
    sigma_pressure_locations = sigma_pressure_locations.metpy.quantify()
    sigma_pressure_locations = sigma_pressure_locations.transpose('time', 'pfull', 'lat', 'lon')

    interp_pressure_levels = interp_pressure_levels = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 97500, 100000] * units.Pa
    raw_data['temp'].attrs['units'] = 'K'

    temp_interp = xr.apply_ufunc(log_interpolate_1d, interp_pressure_levels, sigma_pressure_locations, raw_data['temp'].metpy.quantify(),
                                 input_core_dims=[['plev'], ['pfull'], ['pfull']],output_core_dims=[['plev']], vectorize=True, dask='parallelized', 
                                 output_dtypes=[float])
    uwind_interp = xr.apply_ufunc(log_interpolate_1d, interp_pressure_levels, sigma_pressure_locations, raw_data['ucomp'].metpy.quantify(),
                                  input_core_dims=[['plev'], ['pfull'], ['pfull']],output_core_dims=[['plev']], vectorize=True, dask='parallelized', 
                                  output_dtypes=[float])
    vwind_interp = xr.apply_ufunc(log_interpolate_1d, interp_pressure_levels, sigma_pressure_locations, raw_data['vcomp'].metpy.quantify(),
                                  input_core_dims=[['plev'], ['pfull'], ['pfull']],output_core_dims=[['plev']], vectorize=True, dask='parallelized', 
                                  output_dtypes=[float])
    omega_interp = xr.apply_ufunc(log_interpolate_1d, interp_pressure_levels, sigma_pressure_locations, raw_data['omega'].metpy.quantify(),
                                  input_core_dims=[['plev'], ['pfull'], ['pfull']],output_core_dims=[['plev']], vectorize=True, dask='parallelized', 
                                  output_dtypes=[float])
    output_dataset = xr.Dataset()
    output_dataset['temp'] = temp_interp
    output_dataset['uwind'] = uwind_interp
    output_dataset['vwind'] = vwind_interp
    output_dataset['omega'] = omega_interp
    output_dataset.assign_coords({'plev':interp_pressure_levels})
    output_dataset.to_netcdf('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+experiment_name+'_plev.nc')
    return output_dataset