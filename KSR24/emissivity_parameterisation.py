import xarray as xr
from KSR24_functions import *
import numpy as np
import xhistogram.xarray as xhist
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats
from scipy.optimize import curve_fit

def emiss_fit(x, a, b, c):
    return a*np.exp(-x/b) + c

def piecewise_linear(x, x0,y0, k1):
    return xr.where(x >= x0, y0, k1*x + y0-k1*x0)

def main():
    data = xr.open_dataset("/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/hourly/lwp_and_cbh.nc", engine="netcdf4")
    data['emissivity'] = (1-(np.exp(-158*data['tclw'])))

    lat_weights = np.cos(np.radians(data.latitude))

    histogram = xhist.histogram(data['cbh'], data['emissivity'], bins=30, weights=lat_weights,range=[[0, 17000], [0,1]], block_size=None)
    histo_numbers = xhist.histogram(data['cbh'], bins=30, weights=lat_weights, range=[0, 17000], block_size=None)

    emissivity_weights = histogram * histogram.emissivity_bin

    mean_emissivity = emissivity_weights.sum(dim='emissivity_bin')/histo_numbers
    
    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(emiss_fit, histo_numbers.cbh_bin, mean_emissivity, p0=[1, 4000, 0])

    ptest, ptestcov = curve_fit(piecewise_linear, histo_numbers.cbh_bin, mean_emissivity, p0=[8000, 0.1, 1e-4])

    #regression = stats.linregress(histo_numbers.cbh_bin, mean_emissivity)
    x = np.linspace(0, 17000, 100)

    print(popt)
    print(ptest)

    fig, ax = plt.subplots(1, 1, layout='constrained')
    histogram.T.plot(ax=ax, norm=colors.LogNorm(vmin=0.1))
    ax.scatter(histo_numbers.cbh_bin, mean_emissivity,c='m')
    ax.plot(x, popt[0]*np.exp(-x/popt[1])+popt[2], 'red')
    ax.plot(x, piecewise_linear(x, ptest[0], ptest[1], ptest[2]))
    ax.set_title('Emissivity Parameterisation')
    fig.savefig('emissivity_plot.png')
    return None

main()

