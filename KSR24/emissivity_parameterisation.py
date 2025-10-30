import xarray as xr
from KSR24_functions import *
import numpy as np
import xhistogram.xarray as xhist
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats

def main():
    data = xr.open_mfdataset("/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/*.nc", engine="netcdf4").resample({'valid_time':'MS'}).mean().compute()
    
    data['RH'] = calc_vapor_pressure(data['d2m'])/calc_vapor_pressure(data['t2m'])
    data['z_cb'] = calc_cloud_height(data['t2m'], data['RH'])
    data['emissivity'] = (1-(np.exp(-158*data['tclw'])))

    lat_weights = np.cos(np.radians(data.latitude))

    land_histogram = xhist.histogram(data['z_cb'], data['emissivity'], bins=30,weights=lat_weights * data['lsm'], block_size=None)
    ocean_histogram = xhist.histogram(data['z_cb'], data['emissivity'], bins=30,weights=lat_weights *(1- data['lsm']), block_size=None)
    land_histo_numbers = xhist.histogram(data['z_cb'], bins=30, weights=lat_weights*data['lsm'], block_size=None)
    ocean_histo_numbers = xhist.histogram(data['z_cb'], bins=30, weights=lat_weights*(1-data['lsm']), block_size=None)

    land_emissivity_weights = land_histogram * land_histogram.emissivity_bin
    ocean_emissivity_weights = ocean_histogram * ocean_histogram.emissivity_bin

    land_mean_emissivity = land_emissivity_weights.sum(dim='emissivity_bin')/land_histo_numbers
    ocean_mean_emissivity = ocean_emissivity_weights.sum(dim='emissivity_bin')/ocean_histo_numbers
    
    land_regression = stats.linregress(land_histo_numbers.z_cb_bin, land_mean_emissivity)
    ocean_regression = stats.linregress(ocean_histo_numbers.z_cb_bin.sel({'z_cb_bin':slice(None, 5000)}), ocean_mean_emissivity.sel({'z_cb_bin':slice(None, 5000)}))
    x = np.linspace(0, 7000, 50)

    print(land_regression.slope)
    print(land_regression.intercept)
    print(land_regression.rvalue)

    print(ocean_regression.slope)
    print(ocean_regression.intercept)
    print(ocean_regression.rvalue)

    fig, ax = plt.subplots(2, 1, layout='constrained')
    land_histogram.T.plot(ax=ax[0], norm=colors.LogNorm(vmin=0.1))
    ax[0].scatter(land_histo_numbers.z_cb_bin, land_mean_emissivity,c='m')
    ax[0].plot(x, land_regression.slope * x + land_regression.intercept, 'red')
    ax[0].set_title('Land')

    ocean_histogram.T.plot(ax=ax[1], norm=colors.LogNorm(vmin=0.1))
    ax[1].scatter(ocean_histo_numbers.z_cb_bin, ocean_mean_emissivity, c='m')
    ax[1].plot(x, ocean_regression.slope * x + ocean_regression.intercept, 'red')
    ax[1].set_title('Ocean')
    fig.savefig('emissivity_plot.png')
    return None

main()

