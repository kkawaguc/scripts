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
    histogram = xhist.histogram(data['z_cb'], data['emissivity'], bins=30,weights=lat_weights, block_size=None)
    histo_numbers = xhist.histogram(data['z_cb'], bins=30, weights=lat_weights, block_size=None)

    emissivity_weights = histogram * histogram.emissivity_bin

    mean_emissivity = emissivity_weights.sum(dim='emissivity_bin')/histo_numbers

    regression = stats.linregress(histo_numbers.z_cb_bin, mean_emissivity)
    x = np.linspace(0, 7000, 50)

    print(mean_emissivity)
    print(regression.slope)
    print(regression.intercept)
    print(regression.rvalue)
    fig, ax = plt.subplots(1, 1)
    histogram.T.plot(ax=ax, norm=colors.LogNorm(vmin=0.1))
    ax.scatter(histo_numbers.z_cb_bin, mean_emissivity, c='k')
    ax.plot(x, regression.slope * x + regression.intercept, 'red')
    fig.savefig('emissivity_plot.png')
    return None

#result is -1.214e-4 * cb + 0.9884
main()

