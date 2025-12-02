import xarray as xr
from Surface_Pattern.CMIP6_analysis_functions import *
import xskillscore as xs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def main():
    data = xr.open_dataset('/gws/nopw/j04/csgap/kkawaguchi/surface_data/piControl.nc')

    land_mask = calc_land_mask(data)
    ocean_data = xr.where(land_mask == True, data, np.nan)

    tropical_lat_weighting = np.cos(np.radians(data.sel({'lat':slice(-30, 30)}).lat))

    sst_sharp = calc_Fueglistaler_idx(ocean_data['ts'])

    sst_sharp_anom = sst_sharp.groupby('time.month').apply(remove_time_mean)
    sst_idx = (sst_sharp_anom - sst_sharp_anom.mean('time'))/sst_sharp_anom.std('time')


    lag_times = np.linspace(-7, 7, 15, dtype=int)

    LH_flux_anomaly = (-ocean_data['hfls']).groupby('time.month').apply(remove_time_mean)

    LH_flux_lagged = xr.concat([LH_flux_anomaly.shift({'time':lag}) for lag in lag_times], dim='lag')
    LH_flux_lagged = LH_flux_lagged.assign_coords({'lag':lag_times})

    sst_anomaly = ocean_data['ts'].groupby('time.month').apply(remove_time_mean)
    sst_anomaly_lagged = xr.concat([sst_anomaly.shift({'time':lag}) for lag in lag_times], dim='lag')
    sst_anomaly_lagged = sst_anomaly_lagged.assign_coords({'lag':lag_times})

    lagged_LH_regression = xs.linslope(sst_idx, LH_flux_lagged, dim='time', skipna=True)
    lagged_sst_regression = xs.linslope(sst_idx, sst_anomaly_lagged, dim='time', skipna=True)

    plot1 = lagged_sst_regression.mean('model').plot(col='lag', robust=True,col_wrap=5, figsize=(18, 5),subplot_kws={'projection':ccrs.Robinson(central_longitude=180)}, transform=ccrs.PlateCarree())
    for ax in plot1.axs.flatten():
        ax.coastlines()
    plot1.fig.savefig('piControl_laggedSST_regression.png')

    plot = lagged_LH_regression.mean('model').plot(col='lag', robust=True,col_wrap=5, figsize=(18, 5),subplot_kws={'projection':ccrs.Robinson(central_longitude=180)}, transform=ccrs.PlateCarree())
    for ax in plot.axs.flatten():
        ax.coastlines()
    plot.fig.savefig('piControl_laggedLH_regression.png')

    fig, ax = plt.subplots(1, 1)

    ax.plot(lagged_LH_regression.sel({'lat':slice(-30, 30)}).weighted(tropical_lat_weighting).mean(('model', 'lat', 'lon')))
    ax.plot(lagged_sst_regression.sel({'lat':slice(-30, 30)}).weighted(tropical_lat_weighting).mean(('model', 'lat', 'lon')))
    fig.savefig('Tropical_mean_lagged_regression.png')
    return None

main()