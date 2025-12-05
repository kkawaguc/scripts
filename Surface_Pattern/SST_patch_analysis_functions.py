import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from metpy.interpolate import log_interpolate_1d
import numpy as np
from metpy.units import units
import math
import easyclimate.core.utility as utility
import easyclimate.field.boundary_layer.aerobulk as aerobulk

def subset_variables(dataset):
    '''Only extract relevant variables that are needed.
    Manually add variables here as necessary.'''
    list_of_vars = ['temp','ucomp','vcomp','sphum','ps',
                    't_surf', 'ice_conc', 'flux_lhe', 'flux_t',
                    'soc_olr', 'soc_olr_clr',
                    'soc_toa_sw', 'soc_toa_sw_clr', 'soc_toa_sw_up', 'soc_toa_sw_up_clr', 'soc_toa_sw_down',  
                    'soc_surf_flux_lw', 'soc_surf_flux_lw_down', 'soc_surf_flux_lw_clr', 'soc_surf_flux_lw_down_clr',
                    'soc_surf_flux_sw', 'soc_surf_flux_sw_down', 'soc_surf_flux_sw_clr', 'soc_surf_flux_sw_down_clr', 
                    'tot_cld_amt','high_cld_amt', 'mid_cld_amt', 'low_cld_amt']
    return dataset[list_of_vars].isel({'pfull':-1})


def extract_response_function(patch_name='wp', pert_type='warming'):
    '''Imports the response to a given patch location
    and whether the perturbation is warming, cooling or symmetric response.

    Removes the spin-up year, then calculates the time-mean response.
    Inputs:
        patch_name: must be one of wp, ep, np or so
        pert_type: must be one of symmetric, warming or cooling
    Returns:
        resp: the response function'''
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    if pert_type == 'cooling':
        ctrl = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/Greens_function_control/run*/atmos_monthly.nc',
                                 decode_times=time_coder, decode_timedelta=True, engine='netcdf4', preprocess=subset_variables, chunks=None)
        m2K = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+patch_name+'_m2k/run*/atmos_monthly.nc',
                                 decode_times=time_coder, decode_timedelta=True, engine='netcdf4', preprocess=subset_variables, chunks=None)
        ctrl = ctrl.isel({'time':slice(12, None)}).mean('time')
        m2K = m2K.isel({'time':slice(12, None)}).mean('time')
        resp = ctrl - m2K
    elif pert_type == 'warming':
        p2K = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+patch_name+'_p2k/run*/atmos_monthly.nc',
                                decode_times=time_coder, decode_timedelta=True, engine='netcdf4', preprocess=subset_variables, chunks=None)
        ctrl = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/Greens_function_control/run*/atmos_monthly.nc',
                                decode_times=time_coder, decode_timedelta=True, engine='netcdf4', preprocess=subset_variables, chunks=None)
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

    (resp['temp']).plot(ax=ax[0], vmin=-2, vmax=2, cmap='RdBu_r', transform=ccrs.PlateCarree(), cbar_kwargs={'label':'$\\Delta T$ (Lowest model level)'})
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

def plot_turbulent_flux_decomposition(patch_name='wp', pert_type='warming'):
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ctrl = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/Greens_function_control/run*/atmos_monthly.nc',
                             decode_times=time_coder, decode_timedelta=True, engine='netcdf4', preprocess=subset_variables, 
                             chunks={'time':12, 'lat':64, 'lon':128})
    ctrl = ctrl.isel({'time':slice(12, None)}).groupby('time.month').mean('time').compute()
    HF_ctrl = aerobulk.calc_turbulent_fluxes_without_skin_correction(ctrl["t_surf"], "degK", ctrl["temp"], "degK", 
                                                                     ctrl["sphum"], "kg/kg", ctrl["ucomp"],
                                                                     ctrl["vcomp"], ctrl["ps"], "Pa")

    if pert_type == 'cooling':
        m2K = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+patch_name+'_m2k/run*/atmos_monthly.nc',
                                decode_times=time_coder, decode_timedelta=True, engine='netcdf4', preprocess=subset_variables,
                                chunks={'time':12, 'lat':64, 'lon':128})
        m2K = m2K.isel({'time':slice(12, None)}).groupby('time.month').mean('time').compute()
        #HF_m2K = aerobulk.calc_turbulent_fluxes_without_skin_correction(m2K["t_surf"], "degK", m2K["temp"], "degK", 
        #                                                                m2K["sphum"], "kg/kg", m2K["ucomp"],
        #                                                                m2K["vcomp"], m2K["ps"], "Pa")
        HF_m2K_temp = aerobulk.calc_turbulent_fluxes_without_skin_correction(m2K["t_surf"], "degK", m2K["temp"], "degK", 
                                                                             m2K["sphum"], "kg/kg", ctrl["ucomp"],
                                                                             ctrl["vcomp"], m2K["ps"], "Pa")
        HF_m2K_wind = aerobulk.calc_turbulent_fluxes_without_skin_correction(ctrl["t_surf"], "degK", ctrl["temp"], "degK", 
                                                                             ctrl["sphum"], "kg/kg", m2K["ucomp"],
                                                                             m2K["vcomp"], m2K["ps"], "Pa")
        real_dHF = (ctrl['flux_lhe'] + ctrl['flux_t']) - (m2K['flux_lhe'] - m2K['flux_t'])
        dHF_temp = (HF_ctrl['ql'] + HF_ctrl['qh']) - (HF_m2K_temp['ql'] + HF_m2K_temp['qh']) 
        dHF_wind = (HF_ctrl['ql'] + HF_ctrl['qh']) - (HF_m2K_wind['ql'] + HF_m2K_wind['qh'])
        delta_wind = (np.sqrt(ctrl['ucomp'] ** 2 + ctrl['vcomp'] ** 2) - 
                        np.sqrt(m2K['ucomp'] ** 2 + m2K['vcomp'] ** 2))
    elif pert_type == 'warming':
        p2K = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+patch_name+'_p2k/run*/atmos_monthly.nc',
                                decode_times=time_coder, decode_timedelta=True, engine='netcdf4', preprocess=subset_variables,
                                chunks={'time':12, 'lat':64, 'lon':128})
        p2K = p2K.isel({'time':slice(12, None)}).groupby('time.month').mean('time').compute()
        #HF_p2K = aerobulk.calc_turbulent_fluxes_without_skin_correction(p2K["t_surf"], "degK", p2K["temp"], "degK", 
        #                                                                p2K["sphum"], "kg/kg", p2K["ucomp"],
        #                                                                p2K["vcomp"], p2K["ps"], "Pa")
        HF_p2K_temp = aerobulk.calc_turbulent_fluxes_without_skin_correction(p2K["t_surf"], "degK", p2K["temp"], "degK", 
                                                                             p2K["sphum"], "kg/kg", ctrl["ucomp"],
                                                                             ctrl["vcomp"], p2K["ps"], "Pa")
        HF_p2K_wind = aerobulk.calc_turbulent_fluxes_without_skin_correction(ctrl["t_surf"], "degK", ctrl["temp"], "degK", 
                                                                             ctrl["sphum"], "kg/kg", p2K["ucomp"],
                                                                             p2K["vcomp"], p2K["ps"], "Pa")
        real_dHF = (p2K['flux_lhe'] + p2K['flux_t']) - (ctrl['flux_lhe'] + ctrl['flux_t'])
        dHF_temp = (HF_p2K_temp['ql'] + HF_p2K_temp['qh']) - (HF_ctrl['ql'] + HF_ctrl['qh'])
        dHF_wind = (HF_p2K_wind['ql'] + HF_p2K_wind['qh']) - (HF_ctrl['ql'] + HF_ctrl['qh'])
        delta_wind = (np.sqrt(p2K['ucomp']** 2 + p2K['vcomp'] ** 2) - 
                        np.sqrt(ctrl['ucomp'] ** 2 + ctrl['vcomp']** 2))
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
    (-real_dHF).mean('month').plot(ax = ax[0, 0], vmin=-25, vmax=25,cmap='RdBu_r',transform=ccrs.PlateCarree())
    dHF_temp.mean('month').plot(ax = ax[0, 1], vmin=-25, vmax=25,cmap='RdBu_r',transform=ccrs.PlateCarree())
    dHF_wind.mean('month').plot(ax = ax[1, 1], vmin=-25, vmax=25,cmap='RdBu_r',transform=ccrs.PlateCarree())
    #delta_wind.mean('month').plot(ax = ax[1, 0], vmin=-2, vmax=2, cmap='RdBu_r', transform=ccrs.PlateCarree())
    #ctrl.mean('month').isel({'lat':slice(None, None, 8), 'lon':slice(None, None, 8)}).plot.quiver(x='lon', y='lat', u='ucomp', v='vcomp', ax=ax[1,0])
    for i in range(4):
        ax[i // 2, i %2].coastlines()
    ax[0, 0].set_title('Change in Turbulent flux')
    ax[0, 1].set_title('Thermodynamic component')
    ax[1, 1].set_title('Dynamic component')
    plt.delaxes(ax[1, 0])
    plt.show()
    return None
    

def sel_pressure_data(dataset):
    return dataset[['ucomp', 'vcomp', 'omega', 'temp', 'ps']]

def sel_surf_pres(dataset):
    return dataset['ps']

def interpolate_to_pressure_levels(experiment_name='Greens_function_control'):
    '''Interpolate data to standard pressure levels using metpy. 
    Saves as one large netcdf file.'''
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    raw_data = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+experiment_name+'/run*/atmos_monthly.nc',
                                 decode_times=time_coder, decode_timedelta=True, preprocess=sel_pressure_data, engine='netcdf4')
    mean_surf_pressure = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/Greens_function_control/run*/atmos_monthly.nc',
                                 decode_times=time_coder, decode_timedelta=True, preprocess=sel_surf_pres, engine='netcdf4')
    mean_surf_pressure = mean_surf_pressure.isel({'time':slice(12, None)}).mean('time')
    sigma_pressure_locations = (raw_data.pfull / 1000) * mean_surf_pressure['ps']
    sigma_pressure_locations.attrs['units'] = 'Pa'
    sigma_pressure_locations = sigma_pressure_locations.metpy.quantify()
    sigma_pressure_locations = sigma_pressure_locations.transpose('pfull', 'lat', 'lon')

    interp_pressure_levels = [500, 1000,2000,3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 97500, 100000] * units.Pa
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
    output_dataset['ps'] = raw_data['ps']
    output_dataset = output_dataset.assign_coords({'plev':interp_pressure_levels})
    output_dataset.to_netcdf('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+experiment_name+'_plev.nc')
    return output_dataset

def extract_pressure_level_response(patch = 'wp', pert_type='warming'):
    '''Extracts the response and the control from the pressure level data'''
    p2K = xr.open_dataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+patch+'_p2k_plev.nc')
    m2K = xr.open_dataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/'+patch+'_m2k_plev.nc')
    p2K = p2K.isel({'time':slice(12, None)}).mean('time')
    m2K = m2K.isel({'time':slice(12, None)}).mean('time')
    control = xr.open_dataset('/gws/nopw/j04/csgap/kkawaguchi/SST_patch_data/results/Greens_function_control_plev.nc')
    control = control.isel({'time':slice(12, None)}).mean('time')

    if pert_type=='warming':
        response = p2K - control
        return response, control, p2K
    elif pert_type=='cooling':
        response = control - m2K  
        return response, control, m2K
    else:
        print('Error: pert_type must be warming or cooling')
        return None

def calc_streamfunction(data):
    '''Calculate the mass-weighted meridional streamfunction'''
    r = 6.371e6
    g = 9.81
    pres_bnds = np.array([0, 750, 1500, 2500, 4000, 6000, 8500, 12500, 17500, 22500, 27500, 32500, 37500, 42500, 47500, 
                          52500, 57500, 62500, 67500,72500,77500,82500,87500,92500,96250, 98750,102500])
    interp_pressure_levels = [500, 1000,2000,3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 
                              45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 97500, 100000] * units.Pa
    
    mass_weights = xr.DataArray(np.diff(pres_bnds, n=1), coords={'plev':interp_pressure_levels})
    streamfunction = (2 * math.pi * r * np.cos(np.radians(data.lat)) / g) * (data['vwind'].mean('lon') * mass_weights).cumulative('plev').sum()
    return streamfunction

def plot_zonal_mean_responses(response, control, pert_run):
    fig = plt.figure(layout='constrained', figsize=(16, 8))
    ax = np.empty((2, 2), dtype=object)

    # Regular subplots (no projection)
    ax[0,0] = fig.add_subplot(2, 2, 1)
    ax[0,1] = fig.add_subplot(2, 2, 2)
    ax[1,0] = fig.add_subplot(2, 2, 3)

    # Map subplot with PlateCarree projection
    ax[1,1] = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree(central_longitude=180))

    response['temp'].mean('lon').plot(ax=ax[0, 0], x='lat', yincrease=False, vmin=-0.8, vmax=0.8, cmap='RdBu_r', cbar_kwargs={'label':''})
    temp_contours = control['temp'].mean('lon').plot.contour(ax=ax[0, 0], x='lat', yincrease=False, add_colorbar=False, 
                                                             levels=[190, 200, 210, 220, 230, 240, 250, 260, 270, 280,290, 300], 
                                                             colors='black')
    ax[0,0].clabel(temp_contours, temp_contours.levels[::2], fontsize=10)
    ax[0,0].set_yticks([20000, 40000, 60000, 80000, 100000], labels=['200', '400', '600', '800', '1000'])
    ax[0,0].set_xlabel('')
    ax[0,0].set_title('a) Temperature (K)')

    response['uwind'].mean('lon').plot(ax=ax[0, 1], x='lat', yincrease=False, vmin=-4, vmax=4, cmap='RdBu_r', cbar_kwargs={'label':''})
    wind_contours = control['uwind'].mean('lon').plot.contour(ax=ax[0, 1], x='lat', yincrease=False, add_colorbar=False, 
                                                              levels=np.linspace(-40, 40, 9, dtype=int), colors='black')
    ax[0,1].clabel(wind_contours, wind_contours.levels[::2], fontsize=10)
    ax[0,1].set_yticks([20000, 40000, 60000, 80000, 100000], labels=['200', '400', '600', '800', '1000'])
    ax[0,1].set_xlabel('')
    ax[0,1].set_title('b) Zonal wind (m/s)')

    calc_streamfunction(response).plot(ax=ax[1, 0], x='lat', yincrease=False, robust=True, cmap='RdBu_r')
    stm_fn = calc_streamfunction(control).plot.contour(ax=ax[1, 0], x='lat', yincrease=False, add_colorbar=False, 
                                                       levels=[-2.5e11, -2e11, -1.5e11, -1e11, -6e10, -3e10, -1e10, 1e10, 3e10, 6e10, 1e11, 1.5e11, 2e11, 2.5e11], 
                                                       colors='black')
    ax[1,0].clabel(stm_fn, stm_fn.levels[::2], fontsize=10, fmt=lambda x: f"{x/1e10:.1f}")
    ax[1,0].set_yticks([20000, 40000, 60000, 80000, 100000], labels=['200', '400', '600', '800', '1000'])
    ax[1,0].set_xlabel('')
    ax[1,0].set_title('c) Meridional streamfunction (kg/s)')

    pert_run['omega'].sel({'plev':50000, 'lat':slice(-20, 20)}).plot(ax=ax[1, 1], vmin=-0.15, vmax=0.15, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    control['omega'].sel({'plev':50000, 'lat':slice(-20, 20)}).plot.contour(ax=ax[1, 1], add_colorbar=False, levels=[0], 
                                                      colors='black', linewidth=3,transform=ccrs.PlateCarree())
    ax[1,1].coastlines()
    ax[1,1].set_title('d) Tropical 500hPa vertical velocity (Pa/s)')
    plt.show()
    return None