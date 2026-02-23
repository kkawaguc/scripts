# %%

from CMIP6_analysis_functions import *
import xarray as xr
import numpy as np
import xskillscore as xs
import cartopy.crs as ccrs
import pandas as pd
from data_preprocessing_amip_piForcing import regrid_data

# %%

def calc_slp(ps, alt, temp):
    Rd = 287
    e_p = calc_vapor_pressure(temp)/ps
    virt_temp = temp / (1 - (1-0.622) * e_p)
    slp = ps * np.exp(9.81 * alt / (Rd * virt_temp))
    return slp

def Walker_strength(slp):
    '''calculates the Walker circulation strength using the sea level pressure
    differential following Vecchi et al. (2007)'''
    lat_weights = np.cos(np.radians(slp.lat))
    west_slp = slp.sel({'lat':slice(-5, 5), 'lon':slice(80, 160)}).weighted(lat_weights).mean(('lat', 'lon'))
    east_slp = slp.sel({'lat':slice(-5, 5), 'lon':slice(200, 280)}).weighted(lat_weights).mean(('lat', 'lon'))
    return east_slp - west_slp

monthly_data = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/amip_piForcing_new_monthly.nc', chunks={'time':-1, 'lat':24, 'lon':24, 'model':1}, drop_variables=['ta', 'hus'])

monthly_data['SFC'] = monthly_data['rsds'] + monthly_data['rlds'] - (monthly_data['rsus'] + monthly_data['rlus'] + monthly_data['hfss'] + monthly_data['hfls'])
monthly_data['SFC_rad'] = monthly_data['rsds'] + monthly_data['rlds'] - (monthly_data['rsus'] + monthly_data['rlus'])
monthly_data['SFC_rad_cs'] = monthly_data['rsdscs'] + monthly_data['rldscs'] - (monthly_data['rsuscs'] + monthly_data['rlus'])
monthly_data['SFC_CRE'] = monthly_data['SFC_rad'] - monthly_data['SFC_rad_cs']


land_mask = calc_land_mask(monthly_data)
ocean_data = xr.where(land_mask == True, monthly_data, np.nan)

lat_weighting = np.cos(np.radians(monthly_data.lat))

geopotential = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/era5_geopotential.nc')
altitude = geopotential / 9.81
alt_regridded = regrid_data(altitude.isel({'latitude':slice(None, None, -1)})).drop_vars(['lat_b', 'lon_b'])

monthly_data['SLP'] = calc_slp(monthly_data['ps'], alt_regridded['z'], monthly_data['tas'])

def calc_anomaly(x):
    return (x - x.mean(dim='time'))

# %%

#Calculate normalised SST#. 
#tropical_SST = ocean_data['ts'].sel({'lat':slice(-30, 30)})
#SST_sharp_threshold = tropical_SST.compute().weighted(lat_weighting).quantile(0.7, dim=('lat', 'lon'))
#SST_sharp_raw = (xr.where(tropical_SST > SST_sharp_threshold, tropical_SST, np.nan).weighted(lat_weighting).mean(('lat', 'lon'))
#                 - tropical_SST.weighted(lat_weighting).mean(('lat', 'lon')))

#SST_sharp_anomaly = SST_sharp_raw.groupby('time.month').map(calc_anomaly)
#normed_SST_sharp_anomaly = SST_sharp_anomaly/SST_sharp_anomaly.std('time')

#normed_SST_sharp_anomaly.to_netcdf('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/sst_sharp_idx.nc')
# %%

#Plot 1: Show the surface flux responses to the SST# anomaly

normed_SST_sharp_anomaly = xr.open_dataarray('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/sst_sharp_idx.nc')

def fig1_process(var_name):
    if var_name == 'ts':
        var_anomaly = ocean_data[var_name].groupby('time.month').map(calc_anomaly).chunk({'time':-1})
    else:
        var_anomaly = monthly_data[var_name].groupby('time.month').map(calc_anomaly).chunk({'time':-1})
    return xs.linslope(normed_SST_sharp_anomaly, var_anomaly, dim='time').mean('model')

fig1a = fig1_process('ts')
fig1b = fig1_process('SFC')
fig1c = fig1_process('SFC_rad_cs')
fig1d = fig1_process('SFC_CRE')
fig1e = fig1_process('hfss')
fig1f = fig1_process('hfls')

fig1, ax1 = plt.subplots(nrows=3, ncols=2, figsize=(10, 8), layout='constrained', subplot_kw={'projection':ccrs.Robinson(central_longitude=210)})
fig1a.plot(ax=ax1[0, 0], vmin=-0.3, vmax=0.3, cmap='bwr',transform=ccrs.PlateCarree())
ax1[0,0].set_title('Surface Temperature ($\\mathrm{K}/\sigma$)')
cbar = fig1b.plot(ax=ax1[0, 1], transform=ccrs.PlateCarree(),  vmin=-5, vmax=5, cmap='bwr', add_colorbar=False)
ax1[0,1].set_title('Net Flux')
fig1c.plot(ax=ax1[1, 0], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, cmap='bwr', add_colorbar=False)
ax1[1,0].set_title('Clear-sky radiative')
fig1d.plot(ax=ax1[1, 1], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, cmap='bwr', add_colorbar=False)
ax1[1,1].set_title('Cloud radiative')
(-fig1e).plot(ax=ax1[2, 0], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, cmap='bwr', add_colorbar=False)
ax1[2,0].set_title('Sensible heat')
(-fig1f).plot(ax=ax1[2, 1], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, cmap='bwr', add_colorbar=False)
ax1[2,1].set_title('Latent heat')

fig1.colorbar(cbar, ax=ax1[:,:], orientation='horizontal')

for i in range(6):
    ax1[i//2, i%2].coastlines()

fig1.savefig('Plots/Plot1.png')

# %%

#Plot 2a: Show the radiative kernel responses to the SST# anomaly using the fixed q method

toa_kern = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/TOA_kern_fix_q.nc')
sfc_kern = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/SFC_kern_fix_q.nc')

toa_kern['CLD_NET'] = toa_kern['CLD_SW'] + toa_kern['CLD_LW']
sfc_kern['CLD_NET'] = sfc_kern['CLD_SW'] + sfc_kern['CLD_LW']

toa_kern = toa_kern[['Planck', 'LR', 'WV', 'Albedo', 'CLD_SW', 'CLD_LW', 'CLD_NET']]
sfc_kern = sfc_kern[['Planck', 'LR', 'WV', 'Albedo', 'CLD_SW', 'CLD_LW', 'CLD_NET']]

toa_kern = toa_kern.to_dataarray(dim='component')
sfc_kern = sfc_kern.to_dataarray(dim='component')

def fig2_process():
    toa_anomaly = toa_kern.groupby('time.month').map(calc_anomaly).chunk({'time':-1})
    toa = xs.linslope(normed_SST_sharp_anomaly, toa_anomaly, dim='time').mean('model')
    sfc_anomaly = sfc_kern.groupby('time.month').map(calc_anomaly).chunk({'time':-1})
    sfc = xs.linslope(normed_SST_sharp_anomaly, sfc_anomaly, dim='time').mean('model')
    atm = toa - sfc
    return xr.concat([toa, atm, sfc], pd.Index(['TOA', 'ATM', 'SFC'], name='loc'))

fig2_data = fig2_process()

fig2 = fig2_data.plot(col='loc', row='component', cmap='bwr', aspect=1.6,
                      transform=ccrs.PlateCarree(), subplot_kws={'projection':ccrs.Robinson(central_longitude=210)})

for ax in fig2.axes.flat:
    ax.coastlines()

fig2.fig.savefig('Plots/Plot2a.png')

# %%
#Plot 2b: Show the radiative kernel responses to the SST# anomaly using the fixed RH method

toa_kern = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/TOA_kern_fix_RH.nc')
sfc_kern = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/SFC_kern_fix_RH.nc')

toa_kern['CLD_NET'] = toa_kern['CLD_SW'] + toa_kern['CLD_LW']
sfc_kern['CLD_NET'] = sfc_kern['CLD_SW'] + sfc_kern['CLD_LW']

toa_kern = toa_kern[['Planck', 'LR', 'RH', 'Albedo', 'CLD_SW', 'CLD_LW', 'CLD_NET']]
sfc_kern = sfc_kern[['Planck', 'LR', 'RH', 'Albedo', 'CLD_SW', 'CLD_LW', 'CLD_NET']]


toa_kern = toa_kern.to_dataarray(dim='component')
sfc_kern = sfc_kern.to_dataarray(dim='component')

fig2_data = fig2_process()

fig2 = fig2_data.plot(col='loc', row='component', cmap='bwr', aspect=1.6,
                      transform=ccrs.PlateCarree(), subplot_kws={'projection':ccrs.Robinson(central_longitude=210)})

for ax in fig2.axes.flat:
    ax.coastlines()

fig2.fig.savefig('Plots/Plot2b.png')
# %%

def fig2_std_process():
    toa_anomaly = toa_kern.groupby('time.month').map(calc_anomaly).chunk({'time':-1})
    toa = xs.linslope(normed_SST_sharp_anomaly, toa_anomaly, dim='time')
    sfc_anomaly = sfc_kern.groupby('time.month').map(calc_anomaly).chunk({'time':-1})
    sfc = xs.linslope(normed_SST_sharp_anomaly, sfc_anomaly, dim='time')
    atm = toa - sfc
    return xr.concat([toa, atm, sfc], pd.Index(['TOA', 'ATM', 'SFC'], name='loc')).std('model')

fig2_std_data = fig2_std_process()

fig2_std = fig2_std_data.plot(col='loc', row='component', aspect=1.6,
                      transform=ccrs.PlateCarree(), subplot_kws={'projection':ccrs.Robinson(central_longitude=210)})

for ax in fig2_std.axes.flat:
    ax.coastlines()

fig2_std.fig.savefig('Plots/Plot2_std.png')

# %%

wap = xr.open_dataset("/gws/ssde/j25a/csgap/kkawaguchi/surface_data/amip_piForcing_monthly_clouds.nc")

wap = wap['wap'].sel({'plev':50000})
wap_clim = wap.mean('time')
wap_anomaly = wap.groupby('time.month').map(calc_anomaly).chunk({'time':-1})
dwapdSST = xs.linslope(normed_SST_sharp_anomaly, wap_anomaly, dim='time')

fig3, ax3 = plt.subplots(1, 1, subplot_kw = {'projection':ccrs.Robinson(central_longitude=210)})
wap_clim.mean('model').sel({'lat':slice(-30, 30)}).plot.contour(colour='black',ax=ax3, levels=[0], transform=ccrs.PlateCarree())
dwapdSST.mean('model').sel({'lat':slice(-30, 30)}).plot(ax=ax3, transform=ccrs.PlateCarree())
ax3.coastlines()

fig3.savefig('Plots/vert_vel.png')


# %%

wap_clim_land = xr.where(land_mask == True, wap_clim, np.nan)
dwapdSST_land = xr.where(land_mask == True, dwapdSST, np.nan)

ascend_clim = xr.where(wap_clim_land < 0, 1, 0)
frac_ascend_clim = ascend_clim.mean('lon').weighted(lat_weighting).mean('lat')

ascend_delta = xr.where(wap_clim_land + dwapdSST_land < 0, 1, 0)
frac_ascend_delta = ascend_delta.mean('lon').weighted(lat_weighting).mean('lat')

fig4, ax4 = plt.subplots(1,2)
ax4[0].scatter(frac_ascend_clim, frac_ascend_delta-frac_ascend_clim)
ax4[0].set_xlabel('Climatological ascent fraction')
ax4[0].set_ylabel('Change in ascent fraction (1 std)')

slp_anomaly = monthly_data['SLP'].groupby('time.month').map(calc_anomaly).chunk({'time':-1})
slp_slope = xs.linslope(normed_SST_sharp_anomaly, slp_anomaly, dim='time')

clim_walker = Walker_strength(monthly_data['SLP'].mean('time'))
diff_walker = Walker_strength(slp_slope)

ax4[1].scatter(clim_walker, diff_walker)
ax4[1].set_xlabel('climatological Walker strength')
ax4[1].set_ylabel('change in walker strength')
fig4.savefig('Plots/mod_walker_ascent.png')