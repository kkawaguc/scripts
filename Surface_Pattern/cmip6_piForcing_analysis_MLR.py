# %%

from CMIP6_analysis_functions import *
import xarray as xr
import numpy as np
import xskillscore as xs
import cartopy.crs as ccrs
import pandas as pd
import xesmf as xe
from sklearn.linear_model import LinearRegression

# %%
def glob_mean(data):
    lat_weight = np.cos(np.radians(data.lat))
    return data.weighted(lat_weight).mean(('lat', 'lon'))

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

def regrid_data(model_data):
    '''Regrid model data conservatively to a common 3 degree grid
    Inputs:
        model_data: the model data to be regridded
    Outputs:
        regridded_data: data regridded conservatively at 3 degrees'''
    output_grid = xr.Dataset(
    {
        "lat": (["lat"], np.linspace(-88.5, 88.5, 60), {"units": "degrees_north"}),
        "lon": (["lon"], np.linspace(1.5, 358.5, 120), {"units": "degrees_east"}),
    }
    )
    # NEED TO ADD LON BOUNDS HERE
    if 'lon' not in model_data.dims:
        model_data = model_data.rename({'longitude':'lon', 'latitude':'lat'})
    model_lon = model_data.lon
    model_lat = model_data.lat
    dlon = np.diff(model_lon, 1)[0]
    model_data = model_data.assign_coords({'lon_b':np.linspace(float(model_lon.min()) - dlon/2, 
                                                 float(model_lon.max()) + dlon/2, 
                                                 len(model_lon)+1),
                                           'lat_b':np.linspace(-90,90,len(model_lat)+1)})
    regridder_conservative = xe.Regridder(model_data, output_grid, 'conservative', periodic=True)
    return regridder_conservative(model_data)

def calc_anomaly(x):
    return (x - x.mean(dim='time'))

def fit_lr(x, y):
    # x, y are 1D arrays along "time"
    maskx = np.isfinite(x) & np.isfinite(y)
    masky = np.isfinite(y)
    if masky.sum() < 2:
        return (np.nan, np.nan, np.nan)  # slope, intercept

    model = LinearRegression()
    model.fit(x[maskx].reshape(2, -1).T, 
              y[masky].reshape(-1))
    return (model.coef_[0].item(), model.coef_[1].item(),model.intercept_.item())

def SST_sharp_contribution(X, Y, return_val='SST_sharp', calc_anom=True):
    '''Calculates the contribution to a given quantity from SST sharp
    Inputs:
        X: the regressor variables (GMST and SST sharp in standard deviation units)
        Y: the regressand
    Outputs:
        SST_sharp_coef: contribution from one standard deviation of SST sharp'''
    if calc_anom == True:
        Y_anom = Y.groupby('time.month').map(calc_anomaly).chunk({'time':-1})
    else:
        Y_anom = Y
    #X_anom_rolling = X.rolling(dim={'time':7}, center=True).mean().dropna('time', how='all').chunk({'time':-1})
    #Y_anom_rolling = Y_anom.rolling(dim={'time':7}, center=True).mean().dropna('time', how='all').chunk({'time':-1})
    T_coef, SST_sharp_coef, intercept = xr.apply_ufunc(fit_lr, X, Y_anom, 
               input_core_dims=[['vars','time'], ['time']], output_core_dims=[[], [], []], 
               vectorize=True, dask='parallelized',
               output_dtypes=[float, float, float])
    fit = T_coef * X.sel({'vars':'T'}) + SST_sharp_coef * X.sel({'vars':'SST_sharp'}) + intercept
    if return_val == 'SST_sharp':
        return SST_sharp_coef
    elif return_val == 'T':
        return T_coef
    elif return_val == 'intercept':
        return intercept
    elif return_val == 'fit':
        return xr.corr(fit, Y_anom, dim='time')
    else:
        print('return_val must be one of SST_sharp, T, intercept or fit')
        return None

# %%

monthly_data = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/amip_piForcing_new_monthly.nc', chunks={'time':-1, 'lat':24, 'lon':24, 'model':1}, drop_variables=['ta', 'hus'])
monthly_data = monthly_data.drop('plev')

monthly_data['TOA'] = -(monthly_data['rsut'] + monthly_data['rlut']) 
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

# %%
GMST_anomaly = monthly_data['tas'].weighted(lat_weighting).mean(('lat', 'lon')).groupby('time.month').map(calc_anomaly)
normed_GMST_anomaly = GMST_anomaly/GMST_anomaly.std('time')
normed_SST_sharp_anomaly = xr.open_dataarray('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/sst_sharp_idx.nc')

#monthly_data.polyfit()
# %%

#Calculate normalised SST#. 
#tropical_SST = ocean_data['ts'].sel({'lat':slice(-30, 30)})
#SST_sharp_threshold = tropical_SST.compute().weighted(lat_weighting).quantile(0.7, dim=('lat', 'lon'))
#SST_sharp_raw = (xr.where(tropical_SST > SST_sharp_threshold, tropical_SST, np.nan).weighted(lat_weighting).mean(('lat', 'lon'))
#                 - tropical_SST.weighted(lat_weighting).mean(('lat', 'lon')))

#SST_sharp_anomaly = SST_sharp_raw.groupby('time.month').map(calc_anomaly)
#monthly_data = monthly_data.assign_coords({'T':normed_GMST_anomaly, 'SST_sharp':normed_SST_sharp_anomaly})

regressor_variables = xr.concat([normed_GMST_anomaly, normed_SST_sharp_anomaly], dim=pd.Index(['T', 'SST_sharp'], name='vars')).compute()

# %%

#Plot 1: Show the surface temperature and circulation responses to the change in SST#

#def fig1_process(var_name):
#    if var_name == 'ts':
#        var_anomaly = ocean_data[var_name].groupby('time.month').map(calc_anomaly).chunk({'time':-1})
#    else:
#        var_anomaly = monthly_data[var_name].groupby('time.month').map(calc_anomaly).chunk({'time':-1})
#    return xs.linslope(normed_SST_sharp_anomaly, var_anomaly, dim='time').mean('model')

sst_sharp_data = SST_sharp_contribution(regressor_variables, monthly_data)
check_fit_ts = SST_sharp_contribution(regressor_variables, monthly_data['ts'], return_val='fit')
check_fit_TOA = SST_sharp_contribution(regressor_variables, monthly_data['TOA'], return_val='fit')
check_fit_SFC = SST_sharp_contribution(regressor_variables, monthly_data['SFC'], return_val='fit')
T_mediated_data = SST_sharp_contribution(regressor_variables, monthly_data[['ts','TOA', 'SFC']], return_val='T')

check_fit_fig, check_fit_ax = plt.subplots(nrows=3, figsize=(5, 8), layout='constrained', subplot_kw={'projection':ccrs.Robinson(central_longitude=210)})
check_fit_ts.mean('model').plot(ax=check_fit_ax[0], transform=ccrs.PlateCarree())
check_fit_TOA.mean('model').plot(ax=check_fit_ax[1], transform=ccrs.PlateCarree())
check_fit_SFC.mean('model').plot(ax=check_fit_ax[2], transform=ccrs.PlateCarree())
for i in range(3):
    check_fit_ax[i].coastlines()
check_fit_ax[0].set_title('Surface Temperature')
check_fit_ax[1].set_title('TOA')
check_fit_ax[2].set_title('SFC')
check_fit_fig.suptitle('Model-mean correlation with predictors')

check_fit_fig.savefig('Plots/check_fit.png')

# %%

fig1, ax1 = plt.subplots(nrows=4, ncols=2, figsize=(10, 10), layout='constrained', subplot_kw={'projection':ccrs.Robinson(central_longitude=210)})
T_mediated_data['ts'].mean('model').plot(ax=ax1[0, 0], transform=ccrs.PlateCarree(), vmin=-1, vmax=1, extend='both', cmap='RdBu_r', add_colorbar=False)
T_mediated_data['TOA'].mean('model').plot(ax=ax1[1, 0], transform=ccrs.PlateCarree(), vmin=-8, vmax=8, extend='both',cmap='RdBu_r', add_colorbar=False)
(T_mediated_data['TOA'] - T_mediated_data['SFC']).mean('model').plot(ax=ax1[2, 0], transform=ccrs.PlateCarree(), vmin=-8, vmax=8, extend='both',cmap='RdBu_r', add_colorbar=False)
T_mediated_data['SFC'].mean('model').plot(ax=ax1[3, 0], transform=ccrs.PlateCarree(), vmin=-8, vmax=8, extend='both',cmap='RdBu_r', add_colorbar=False)
sst_sharp_data['ts'].mean('model').plot(ax=ax1[0, 1], transform=ccrs.PlateCarree(), vmin=-1, vmax=1, extend='both',cmap='RdBu_r', cbar_kwargs={'label':'ts ($\\mathrm{K/\\sigma}$)', 'pad':0.1})
sst_sharp_data['TOA'].mean('model').plot(ax=ax1[1, 1], transform=ccrs.PlateCarree(), vmin=-8, vmax=8, extend='both',cmap='RdBu_r', add_colorbar=False)# cbar_kwargs={'label':'TOA ($\\mathrm{W/m^2/\\sigma}$)'})
cbar = (sst_sharp_data['TOA'] - sst_sharp_data['SFC']).mean('model').plot(ax=ax1[2, 1], transform=ccrs.PlateCarree(), vmin=-8, vmax=8, extend='both',cmap='RdBu_r', add_colorbar=False)# cbar_kwargs={'label':'Energy Response ($\\mathrm{W/m^2/\\sigma}$)', 'shrink':1.5, 'aspect':30})
sst_sharp_data['SFC'].mean('model').plot(ax=ax1[3, 1], transform=ccrs.PlateCarree(), vmin=-8, vmax=8, extend='both',cmap='RdBu_r', add_colorbar=False) #cbar_kwargs={'label':'ts ($\\mathrm{W/m^2/\\sigma}$)'})

fig1.colorbar(cbar,ax=ax1[1:,:], label='Energy Response ($\\mathrm{W/m^2/\\sigma}$)', shrink=0.8, aspect=40)

for i in range(4):
    ax1[i, 0].coastlines()
    ax1[i,1].coastlines()

ax1[0, 0].set_title('T mediated')
ax1[0, 1].set_title('SST# mediated')
ax1[1,0].set_title('')
ax1[1,1].set_title('')
ax1[2,0].set_title('')
ax1[2,1].set_title('')
ax1[3,0].set_title('')
ax1[3,1].set_title('')

ax1[0,0].set_ylabel('Surface Temperature', size=13)
ax1[1,0].set_ylabel('TOA', size=13)
ax1[2,0].set_ylabel('ATM', size=13)
ax1[3,0].set_ylabel('SFC', size=13)

ax1[0,0].set_yticks([])
ax1[1,0].set_yticks([])
ax1[2,0].set_yticks([])
ax1[3,0].set_yticks([])

fig1.savefig('Plots/sst_sharp_T_mediated.png')

#wap = xr.open_dataset("/gws/ssde/j25a/csgap/kkawaguchi/surface_data/amip_piForcing_monthly_clouds.nc")

#wap = wap['wap'].sel({'plev':50000})
#wap_clim = wap.mean('time')

#dwapdSST = SST_sharp_contribution(regressor_variables, wap)

#wap_clim_land = xr.where(land_mask == True, wap_clim, np.nan).sel({'lat':slice(-30, 30)})
#dwapdSST_land = xr.where(land_mask == True, dwapdSST, np.nan).sel({'lat':slice(-30, 30)})

#ascend_clim = xr.where(wap_clim_land < 0, 1, 0)
#frac_ascend_clim = ascend_clim.mean('lon').weighted(lat_weighting).mean('lat')

#ascend_delta = xr.where(wap_clim_land + dwapdSST_land < 0, 1, 0)
#frac_ascend_delta = ascend_delta.mean('lon').weighted(lat_weighting).mean('lat')

#tropical_q250 = monthly_data['hus'].sel({'plev':25000, 'lat':slice(-30, 30)})
#tropical_T250 = monthly_data['ta'].sel({'plev':25000, 'lat':slice(-30, 30)})
#tropical_RH250 = calc_relhum(tropical_T250, tropical_q250, 25000)
#RH250_dSST = SST_sharp_contribution(regressor_variables, tropical_RH250)

#ta_dSST = sst_sharp_data['ta']
#%%


#fig1 = plt.figure(layout='constrained', figsize=(10,8))
#subfigs1 = fig1.subfigures(3, 1, height_ratios=[1.3, 1, 1.2])

#fig1a = sst_sharp_data['ts'].mean('model')
TOA_plot = sst_sharp_data['TOA'].mean('model')
#SFC_plot = sst_sharp_data['SFC'].mean('model')

print(glob_mean(sst_sharp_data['TOA']).mean('model').compute())
print(glob_mean(sst_sharp_data['TOA']).quantile([0, 1], dim='model').compute())
print(glob_mean(sst_sharp_data['SFC_rad']).mean('model').compute())
print(glob_mean(sst_sharp_data['SFC_rad']).quantile([0, 1], dim='model').compute())
#toa_data = monthly_data['TOA'].groupby('time.month').map(calc_anomaly).chunk({'time':-1})
#toa_slope = xs.linslope(normed_SST_sharp_anomaly, toa_data, dim='time')
#print(glob_mean(toa_slope).std('model').compute())

#ax1_1 = subfigs1[0].subplots(1, 3, subplot_kw={'projection':ccrs.Robinson(central_longitude=210)})
#fig1a.plot(ax=ax1_1[0], vmin=-0.3, vmax=0.3, cmap='bwr',transform=ccrs.PlateCarree(), cbar_kwargs={'orientation':'horizontal', 'label':'$\\mathrm{K/\\sigma}$'})
#ax1_1[0].set_title('a) Surface Temperature')
#cbar = TOA_plot.plot(ax=ax1_1[1], vmin=-5, vmax=5, cmap='bwr', extend='both',transform=ccrs.PlateCarree(), cbar_kwargs={'ax':ax1_1[1:],'orientation':'horizontal', 'label':'$\mathrm{W/m^2/\\sigma}$'})
#ax1_1[1].set_title('b) TOA Energy Flux')
#SFC_plot.plot(ax=ax1_1[2], vmin=-5, vmax=5, cmap='bwr',transform=ccrs.PlateCarree(), add_colorbar=False)
#ax1_1[2].set_title('c) SFC Energy Flux')

#ax1_2 = subfigs1[1].subplots(1, 2, subplot_kw={'projection':ccrs.Robinson(central_longitude=210)})
#wap_clim.sel({'lat':slice(-30, 30)}).mean('model').plot.contour(ax=ax1_2[0], levels=[0], colors='black', transform=ccrs.PlateCarree())
#dwapdSST.sel({'lat':slice(-30, 30)}).mean('model').plot(cmap='bwr', ax=ax1_2[0], transform=ccrs.PlateCarree(), vmin=-0.005, vmax=0.005, extend='both',
#                                                        cbar_kwargs={'orientation':'horizontal', 'label':'Pa/$\\sigma$', 'shrink':0.8})
#ax1_2[0].set_title('d) Mid-tropospheric vertical velocity')

#wap_clim.sel({'lat':slice(-30, 30)}).mean('model').plot.contour(ax=ax1_2[1], levels=[0], colors='black', transform=ccrs.PlateCarree())
#(100*RH250_dSST).mean('model').plot(cmap='bwr', ax=ax1_2[1], transform=ccrs.PlateCarree(), vmin=-2, vmax=2, extend='both',
#                                    cbar_kwargs={'orientation':'horizontal', 'label':'%/$\\sigma$', 'shrink':0.8})
#ax1_2[1].set_title('e) 250hPa Relative humidity')

#ax4 = subfigs1[2].subplots(1,2)
#ta_dSST.mean(('lon', 'model')).plot(ax = ax4[0], yincrease=False)
#ax4[0].set_title('f) Zonal mean temperature')

#ax4[0].scatter(frac_ascend_clim, frac_ascend_delta-frac_ascend_clim)
#ax4[0].set_xlabel('Climatological ascent fraction')
#ax4[0].set_ylabel('Change in ascent fraction (1 std)')

#slp_slope = sst_sharp_data['SLP']

#clim_walker = Walker_strength(monthly_data['SLP'].mean('time'))
#diff_walker = Walker_strength(slp_slope)

#ax4[1].scatter(clim_walker, diff_walker)
#ax4[1].set_xlabel('climatological Walker strength')
#ax4[1].set_ylabel('change in walker strength')
#ax4[1].set_title('g) Walker circulation')


#for i in range(3):
#    ax1_1[i].coastlines()
#    if i != 2:
#        ax1_2[i].coastlines()

#fig1.savefig('Plots/Plot1_MLR.png')

# %%

#Plot 2a: Show the radiative kernel responses to the SST# anomaly using the fixed q method

#toa_kern = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/TOA_kern_fix_q.nc')
#sfc_kern = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/SFC_kern_fix_q.nc')

#toa_kern['CLD_NET'] = toa_kern['CLD_SW'] + toa_kern['CLD_LW']
#sfc_kern['CLD_NET'] = sfc_kern['CLD_SW'] + sfc_kern['CLD_LW']

#toa_kern = toa_kern[['Planck', 'LR', 'WV', 'Albedo', 'CLD_SW', 'CLD_LW', 'CLD_NET']]
#sfc_kern = sfc_kern[['Planck', 'LR', 'WV', 'Albedo', 'CLD_SW', 'CLD_LW', 'CLD_NET']]

#toa_kern = toa_kern.to_dataarray(dim='component')
#sfc_kern = sfc_kern.to_dataarray(dim='component')

def fig2_process():
    toa = SST_sharp_contribution(regressor_variables, toa_kern)
    sfc = SST_sharp_contribution(regressor_variables, sfc_kern)
    atm = toa - sfc
    return xr.concat([toa, atm, sfc], pd.Index(['TOA', 'ATM', 'SFC'], name='loc'))

#fig2_data = fig2_process()

#fig2 = fig2_data.plot(col='loc', row='component', cmap='bwr', aspect=1.6,
#                      transform=ccrs.PlateCarree(), subplot_kws={'projection':ccrs.Robinson(central_longitude=210)})

#for ax in fig2.axes.flat:
#    ax.coastlines()

#fig2.fig.savefig('Plots/Plot2a.png')


# %%
#Plot 2b: Show the radiative kernel responses to the SST# anomaly using the fixed RH method

toa_kern = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/TOA_kern_fix_RH_inc_strato.nc')
sfc_kern = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/SFC_kern_fix_RH_inc_strato.nc')

toa_kern['CLD_NET'] = toa_kern['CLD_SW'] + toa_kern['CLD_LW']
sfc_kern['CLD_NET'] = sfc_kern['CLD_SW'] + sfc_kern['CLD_LW']

toa_kern = toa_kern[['Planck', 'LR', 'RH', 'Albedo', 'CLD_SW', 'CLD_LW', 'CLD_NET']]
sfc_kern = sfc_kern[['Planck', 'LR', 'RH', 'Albedo', 'CLD_SW', 'CLD_LW', 'CLD_NET']]

toa_kern['RAD_NET'] = monthly_data['TOA']
sfc_kern['RAD_NET'] = monthly_data['SFC_rad']


toa_kern = toa_kern.to_dataarray(dim='component')
sfc_kern = sfc_kern.to_dataarray(dim='component')

fig2_data = fig2_process()
# %%
fig2_data.to_netcdf('fig2_data.nc')
fig2_data.close()

# %%
fig2_data = xr.open_dataarray('fig2_data.nc')
# %%
fig2 = fig2_data.mean('model').plot(col='loc', row='component', cmap='RdBu_r', aspect=1.7, size=2.5,
                      transform=ccrs.PlateCarree(),subplot_kws={'projection':ccrs.Robinson(central_longitude=210)}, 
                      cbar_kwargs={'orientation':'horizontal', 'pad':0.02,'label':'Radiative flux ($\\mathrm{W/m^2/\\sigma}$)', 'shrink':0.8, 'extend':'both'})

fig2.set_titles(template="{value}", fontsize=18)
fig2.cbar.ax.tick_params(labelsize=15)
fig2.cbar.ax.set_xlabel('Radiative flux sensitivity to SST# ($\\mathrm{W/m^2/\\sigma}$)', fontsize=18)

for ax in fig2.axs.flat:
    ax.coastlines()

fig2.fig.savefig('Plots/Plot2b_MLR.png')

# %%

#dHF = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/dHF.nc')
dHF_temp = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/dHF_temp.nc')
dHF_wind = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/dHF_wind.nc')

thermodynamic = dHF_temp['ql'] + dHF_temp['qh']
dynamic = dHF_wind['ql'] + dHF_wind['qh']

dHF_thermo = SST_sharp_contribution(regressor_variables, thermodynamic, calc_anom=False)
dHF_wind = SST_sharp_contribution(regressor_variables, dynamic, calc_anom=False)

fig, ax = plt.subplots(2, 2, layout='constrained', figsize=(8, 5), subplot_kw={'projection':ccrs.Robinson(central_longitude=210)})

sst_sharp_data['SFC_rad'].mean('model').plot(ax= ax[0,0], vmin=-8, vmax=8, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree(), add_colorbar=False)
sst_sharp_data['SFC_rad'].quantile([2/7, 5/7],dim='model').prod('quantile').plot.contourf(ax=ax[0,0], levels=[-1000, 0], colors='none', transform=ccrs.PlateCarree(), hatches=['..', None], add_colorbar=False)

(-sst_sharp_data['hfls'] - sst_sharp_data['hfss']).mean('model').plot(ax=ax[0,1], vmin=-8, vmax=8, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree(), add_colorbar=False)
(-sst_sharp_data['hfls'] - sst_sharp_data['hfss']).quantile([2/7, 5/7],dim='model').prod('quantile').plot.contourf(ax=ax[0,1], levels=[-1000, 0], colors='none', transform=ccrs.PlateCarree(), hatches=['..', None], add_colorbar=False)

(dHF_thermo).mean('model').plot(ax=ax[1,0], vmin=-8, vmax=8, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree(), add_colorbar=False)
(dHF_thermo).quantile([2/7, 5/7],dim='model').prod('quantile').plot.contourf(ax=ax[1,0], levels=[-1000, 0], colors='none', transform=ccrs.PlateCarree(), hatches=['..', None], add_colorbar=False)

(dHF_wind).mean('model').plot(ax=ax[1,1], vmin=-8, vmax=8, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree(), cbar_kwargs={'ax':ax[:,:], 'label':'Turbulent Heat Flux Sensitivity to SST# ($\\mathrm{W/m^2/\\sigma}$)'})
(dHF_wind).quantile([2/7, 5/7],dim='model').prod('quantile').plot.contourf(ax=ax[1,1], levels=[-1000, 0], colors='none', transform=ccrs.PlateCarree(), hatches=['..', None], add_colorbar=False)


ax[0,0].set_title('Radiative')
ax[0,1].set_title('Turbulent')
ax[1,0].set_title('Thermodynamic component (Bulk)')
ax[1,1].set_title('Dynamic component (Bulk)')

ax[0,0].coastlines()
ax[0,1].coastlines()
ax[1,0].coastlines()
ax[1,1].coastlines()

fig.savefig('Plots/bulk_formulae.png')

# %%

GM_kernel = glob_mean(fig2_data)

fig, ax = plt.subplots(nrows=3, figsize=(6, 6), layout='constrained')

for j in range(3):
    for i in range(6):
        ax[j].scatter(8*[i] + 0.1*np.random.rand(8), GM_kernel.isel({'component':i, 'loc':j}), c=range(8), cmap='Dark2')
        print(GM_kernel.isel({'component':i, 'loc':j}).compute())
    # Add radiative total in
    ax[j].scatter(8*[6] + 0.1*np.random.rand(8), GM_kernel.isel({'component':7, 'loc':j}), c=range(8),cmap='Dark2')
    ax[j].set_xticklabels([])
    ax[j].set_xlim([-0.5, 10.5])
    ax[j].set_ylim([-0.4, 0.2])
    ax[j].set_yticks([-0.3, -0.2, -0.1, 0, 0.1])
    ax[j].set_xticks([])
    ax[j].grid(axis='y')
mods = ['CanESM5', 'CESM2', 'CNRM-CM6-1', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6', 'MRI-ESM2-0', 'TaiESM1']
cmap = plt.cm.Dark2
for k in range(8):
    ax[1].scatter([], [], c=cmap(k), cmap='Dark2',label=f"{mods[k]}")

ax[0].set_title('Global-mean energy budget sensitivity to SST# ($\\mathrm{W/m^2/\\sigma}$)')
ax[0].set_ylabel('TOA')
ax[1].set_ylabel('ATM')
ax[2].set_ylabel('SFC')
ax[1].scatter(8*[7] + 0.1*np.random.rand(8), glob_mean(sst_sharp_data['hfls']), c=range(8),cmap='Dark2')
ax[1].scatter(8*[8] + 0.1*np.random.rand(8), glob_mean(sst_sharp_data['hfss']), c=range(8),cmap='Dark2')

ax[1].scatter(8*[9] + 0.1*np.random.rand(8), -glob_mean(dHF_thermo), c=range(8),cmap='Dark2')
ax[1].scatter(8*[10] + 0.1*np.random.rand(8), -glob_mean(dHF_wind), c=range(8),cmap='Dark2')

ax[2].scatter(8*[7] + 0.1*np.random.rand(8), -glob_mean(sst_sharp_data['hfls']), c=range(8), cmap='Dark2')
ax[2].scatter(8*[8] + 0.1*np.random.rand(8), -glob_mean(sst_sharp_data['hfss']), c=range(8), cmap='Dark2')

ax[2].scatter(8*[9] + 0.1*np.random.rand(8), glob_mean(dHF_thermo), c=range(8),cmap='Dark2')
ax[1].scatter(8*[10] + 0.1*np.random.rand(8), glob_mean(dHF_wind), c=range(8),cmap='Dark2')

fig.legend(loc='center', bbox_to_anchor=(0.5, -0.05), ncols=4)
ax[2].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['Planck', 'LR', 'RH', 'Albedo', 'SW CLD', 'LW CLD', 'RAD TOTAL','LHF', 'SHF', 'Turb Thermo', 'Turb Dyn'])

fig.savefig('Plots/model_spread.png', bbox_inches='tight')

# %%

annual_data = monthly_data.resample({'time':'YS'}).mean()
GMST_annual_anomaly = glob_mean(annual_data['tas'])
annual_data['SFC_SWCRE'] = (annual_data['rsds'] - annual_data['rsus']) - (annual_data['rsdscs'] - annual_data['rsuscs'])
annual_data['SFC_LWCRE'] = annual_data['rlds'] - annual_data['rldscs']
annual_sfc_data = annual_data[['SFC', 'SFC_rad', 'SFC_rad_cs', 'hfls', 'hfss', 'SFC_SWCRE', 'SFC_LWCRE']]
sfc_fdbk = calc_fdbk(annual_sfc_data, GMST_annual_anomaly, yrs=30)

global_sfc_fdbk = glob_mean(sfc_fdbk)

global_sfc_fdbk = global_sfc_fdbk.to_dataarray()
print(xr.corr(global_sfc_fdbk.sel({'variable':'SFC'}), global_sfc_fdbk, dim='time').compute())
#fig_fdbk = global_sfc_fdbk.plot.line(x='time', col='variable', col_wrap=2)

#fig_fdbk.fig.savefig('Plots/time_evolving_fdbk.png')

#fig_fdbk, ax_fdbk = 