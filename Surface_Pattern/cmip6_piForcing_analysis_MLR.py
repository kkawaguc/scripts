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

def SST_sharp_contribution(X, Y, return_val='SST_sharp'):
    '''Calculates the contribution to a given quantity from SST sharp
    Inputs:
        X: the regressor variables (GMST and SST sharp in standard deviation units)
        Y: the regressand
    Outputs:
        SST_sharp_coef: contribution from one standard deviation of SST sharp'''
    Y_anom = Y.groupby('time.month').map(calc_anomaly).chunk({'time':-1})
    X_anom_rolling = X.rolling(dim={'time':7}, center=True).mean().dropna('time', how='all').chunk({'time':-1})
    Y_anom_rolling = Y_anom.rolling(dim={'time':7}, center=True).mean().dropna('time', how='all').chunk({'time':-1})
    print(X_anom_rolling)
    print(Y_anom_rolling)
    T_coef, SST_sharp_coef, intercept = xr.apply_ufunc(fit_lr, X_anom_rolling, Y_anom_rolling, 
               input_core_dims=[['vars','time'], ['time']], output_core_dims=[[], [], []], 
               vectorize=True, dask='parallelized',
               output_dtypes=[float, float, float])
    fit = T_coef * X_anom_rolling.sel({'vars':'T'}) + SST_sharp_coef * X_anom_rolling.sel({'vars':'SST_sharp'}) + intercept
    if return_val == 'SST_sharp':
        return SST_sharp_coef
    elif return_val == 'T':
        return T_coef
    elif return_val == 'intercept':
        return intercept
    elif return_val == 'fit':
        return xr.corr(fit, Y_anom_rolling, dim='time')
    else:
        print('return_val must be one of SST_sharp, T, intercept or fit')
        return None

# %%

monthly_data = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/amip_piForcing_new_monthly.nc', chunks={'time':-1, 'lat':24, 'lon':24, 'model':1})

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

fig1, ax1 = plt.subplots(nrows=3, ncols=2, figsize=(10, 12), layout='constrained', subplot_kw={'projection':ccrs.Robinson(central_longitude=210)})
T_mediated_data['ts'].mean('model').plot(ax=ax1[0, 0], transform=ccrs.PlateCarree())
T_mediated_data['TOA'].mean('model').plot(ax=ax1[1, 0], transform=ccrs.PlateCarree())
T_mediated_data['SFC'].mean('model').plot(ax=ax1[2, 0], transform=ccrs.PlateCarree())
sst_sharp_data['ts'].mean('model').plot(ax=ax1[0, 1], transform=ccrs.PlateCarree())
sst_sharp_data['TOA'].mean('model').plot(ax=ax1[1, 1], transform=ccrs.PlateCarree())
sst_sharp_data['SFC'].mean('model').plot(ax=ax1[2, 1], transform=ccrs.PlateCarree())

for i in range(3):
    ax1[i, 0].coastlines()
    ax1[i,1].coastlines()

ax1[0, 0].set_title('T_mediated')
ax1[0, 1].set_title('SST# mediated')



fig1.savefig('Plots/sst_sharp_T_mediated.png')

wap = xr.open_dataset("/gws/ssde/j25a/csgap/kkawaguchi/surface_data/amip_piForcing_monthly_clouds.nc")

wap = wap['wap'].sel({'plev':50000})
wap_clim = wap.mean('time')

dwapdSST = SST_sharp_contribution(regressor_variables, wap)

#wap_clim_land = xr.where(land_mask == True, wap_clim, np.nan).sel({'lat':slice(-30, 30)})
#dwapdSST_land = xr.where(land_mask == True, dwapdSST, np.nan).sel({'lat':slice(-30, 30)})

#ascend_clim = xr.where(wap_clim_land < 0, 1, 0)
#frac_ascend_clim = ascend_clim.mean('lon').weighted(lat_weighting).mean('lat')

#ascend_delta = xr.where(wap_clim_land + dwapdSST_land < 0, 1, 0)
#frac_ascend_delta = ascend_delta.mean('lon').weighted(lat_weighting).mean('lat')

tropical_q250 = monthly_data['hus'].sel({'plev':25000, 'lat':slice(-30, 30)})
tropical_T250 = monthly_data['ta'].sel({'plev':25000, 'lat':slice(-30, 30)})
tropical_RH250 = calc_relhum(tropical_T250, tropical_q250, 25000)
RH250_dSST = SST_sharp_contribution(regressor_variables, tropical_RH250)

ta_dSST = sst_sharp_data['ta']
#%%


fig1 = plt.figure(layout='constrained', figsize=(10,8))
subfigs1 = fig1.subfigures(3, 1, height_ratios=[1.3, 1, 1.2])

fig1a = sst_sharp_data['ts'].mean('model')
TOA_plot = sst_sharp_data['TOA'].mean('model')
SFC_plot = sst_sharp_data['SFC'].mean('model')

print(glob_mean(TOA_plot).compute())
print(glob_mean(sst_sharp_data['TOA'].quantile([0.17, 0.25, 0.75, 0.83], dim='model')).compute())
print(glob_mean(sst_sharp_data['SFC_rad'].mean('model')).compute())
print(glob_mean(sst_sharp_data['SFC_rad'].quantile([0.17, 0.25, 0.75, 0.83], dim='model')).compute())
#toa_data = monthly_data['TOA'].groupby('time.month').map(calc_anomaly).chunk({'time':-1})
#toa_slope = xs.linslope(normed_SST_sharp_anomaly, toa_data, dim='time')
#print(glob_mean(toa_slope).std('model').compute())

ax1_1 = subfigs1[0].subplots(1, 3, subplot_kw={'projection':ccrs.Robinson(central_longitude=210)})
fig1a.plot(ax=ax1_1[0], vmin=-0.3, vmax=0.3, cmap='bwr',transform=ccrs.PlateCarree(), cbar_kwargs={'orientation':'horizontal', 'label':'$\\mathrm{K/\\sigma}$'})
ax1_1[0].set_title('a) Surface Temperature')
cbar = TOA_plot.plot(ax=ax1_1[1], vmin=-5, vmax=5, cmap='bwr', extend='both',transform=ccrs.PlateCarree(), cbar_kwargs={'ax':ax1_1[1:],'orientation':'horizontal', 'label':'$\mathrm{W/m^2/\\sigma}$'})
ax1_1[1].set_title('b) TOA Energy Flux')
SFC_plot.plot(ax=ax1_1[2], vmin=-5, vmax=5, cmap='bwr',transform=ccrs.PlateCarree(), add_colorbar=False)
ax1_1[2].set_title('c) SFC Energy Flux')

ax1_2 = subfigs1[1].subplots(1, 2, subplot_kw={'projection':ccrs.Robinson(central_longitude=210)})
wap_clim.sel({'lat':slice(-30, 30)}).mean('model').plot.contour(ax=ax1_2[0], levels=[0], colors='black', transform=ccrs.PlateCarree())
dwapdSST.sel({'lat':slice(-30, 30)}).mean('model').plot(cmap='bwr', ax=ax1_2[0], transform=ccrs.PlateCarree(), vmin=-0.005, vmax=0.005, extend='both',
                                                        cbar_kwargs={'orientation':'horizontal', 'label':'Pa/$\\sigma$', 'shrink':0.8})
ax1_2[0].set_title('d) Mid-tropospheric vertical velocity')

wap_clim.sel({'lat':slice(-30, 30)}).mean('model').plot.contour(ax=ax1_2[1], levels=[0], colors='black', transform=ccrs.PlateCarree())
(100*RH250_dSST).mean('model').plot(cmap='bwr', ax=ax1_2[1], transform=ccrs.PlateCarree(), vmin=-2, vmax=2, extend='both',
                                    cbar_kwargs={'orientation':'horizontal', 'label':'%/$\\sigma$', 'shrink':0.8})
ax1_2[1].set_title('e) 250hPa Relative humidity')

ax4 = subfigs1[2].subplots(1,2)
ta_dSST.mean(('lon', 'model')).plot(ax = ax4[0], yincrease=False)
ax4[0].set_title('f) Zonal mean temperature')

#ax4[0].scatter(frac_ascend_clim, frac_ascend_delta-frac_ascend_clim)
#ax4[0].set_xlabel('Climatological ascent fraction')
#ax4[0].set_ylabel('Change in ascent fraction (1 std)')

slp_slope = sst_sharp_data['SLP']

clim_walker = Walker_strength(monthly_data['SLP'].mean('time'))
diff_walker = Walker_strength(slp_slope)

ax4[1].scatter(clim_walker, diff_walker)
ax4[1].set_xlabel('climatological Walker strength')
ax4[1].set_ylabel('change in walker strength')
ax4[1].set_title('g) Walker circulation')


for i in range(3):
    ax1_1[i].coastlines()
    if i != 2:
        ax1_2[i].coastlines()

fig1.savefig('Plots/Plot1_MLR.png')

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

toa_kern = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/TOA_kern_fix_RH.nc')
sfc_kern = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/SFC_kern_fix_RH.nc')

toa_kern['CLD_NET'] = toa_kern['CLD_SW'] + toa_kern['CLD_LW']
sfc_kern['CLD_NET'] = sfc_kern['CLD_SW'] + sfc_kern['CLD_LW']

toa_kern = toa_kern[['Planck', 'LR', 'RH', 'Albedo', 'CLD_SW', 'CLD_LW', 'CLD_NET']]
sfc_kern = sfc_kern[['Planck', 'LR', 'RH', 'Albedo', 'CLD_SW', 'CLD_LW', 'CLD_NET']]




toa_kern = toa_kern.to_dataarray(dim='component')
sfc_kern = sfc_kern.to_dataarray(dim='component')

fig2_data = fig2_process()

fig2 = fig2_data.mean('model').plot(col='loc', row='component', cmap='bwr', aspect=1.6,
                      transform=ccrs.PlateCarree(), subplot_kws={'projection':ccrs.Robinson(central_longitude=210)})

for ax in fig2.axes.flat:
    ax.coastlines()

fig2.fig.savefig('Plots/Plot2b_MLR.png')

print(glob_mean(fig2_data.mean('model').compute()))
print(glob_mean(fig2_data.quantile([0.17, 0.25, 0.75, 0.83], dim='model').compute()))

fig2_std = fig2_data.std('model').plot(col='loc', row='component', aspect=1.6,
                      transform=ccrs.PlateCarree(), subplot_kws={'projection':ccrs.Robinson(central_longitude=210)})

for ax in fig2_std.axes.flat:
    ax.coastlines()

fig2_std.fig.savefig('Plots/Plot2_std_MLR.png')

SW_cld = fig2_data.sel({'component':'CLD_SW', 'loc':'TOA'})
LW_cld = fig2_data.sel({'component':'CLD_LW', 'loc':'TOA'})

low_clouds = xr.where(np.tan(np.radians(22.5))*abs(SW_cld) > abs(LW_cld), SW_cld + LW_cld, 0)
nonlow_clouds = xr.where(np.tan(np.radians(22.5))*abs(SW_cld) < abs(LW_cld), SW_cld + LW_cld, 0)

low_clds = low_clouds.plot(col='model', col_wrap=4,aspect=1.6,
                      transform=ccrs.PlateCarree(), subplot_kws={'projection':ccrs.Robinson(central_longitude=210)})
for ax in low_clds.axes.flat:
    ax.coastlines()

nonlow_clds = nonlow_clouds.plot(col='model', col_wrap=4,aspect=1.6,
                      transform=ccrs.PlateCarree(), subplot_kws={'projection':ccrs.Robinson(central_longitude=210)})
for ax in nonlow_clds.axes.flat:
    ax.coastlines()

low_clds.fig.savefig('Plots/lowclds.png')
nonlow_clds.fig.savefig('Plots/nonlowclds.png')
# %%

annual_data = monthly_data.drop_vars(['ta', 'hus']).resample({'time':'YS'}).mean()
GMST_annual_anomaly = glob_mean(annual_data['tas'])
annual_data['SFC_SWCRE'] = (annual_data['rsds'] - annual_data['rsus']) - (annual_data['rsdscs'] - annual_data['rsuscs'])
annual_data['SFC_LWCRE'] = annual_data['rlds'] - annual_data['rldscs']
annual_sfc_data = annual_data[['SFC', 'SFC_rad', 'SFC_rad_cs', 'hfls', 'hfss', 'SFC_SWCRE', 'SFC_LWCRE']]
sfc_fdbk = calc_fdbk(annual_sfc_data, GMST_annual_anomaly, yrs=30)

global_sfc_fdbk = glob_mean(sfc_fdbk)

global_sfc_fdbk = global_sfc_fdbk.to_dataarray()
print(xr.corr(global_sfc_fdbk.sel({'variable':'SFC'}), global_sfc_fdbk, dim='time').compute())
fig_fdbk = global_sfc_fdbk.plot.line(x='time', col='variable', col_wrap=2)

fig_fdbk.fig.savefig('Plots/time_evolving_fdbk.png')

#fig_fdbk, ax_fdbk = 