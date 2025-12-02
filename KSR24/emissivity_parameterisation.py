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
    tcc_data = xr.open_dataset("/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/hourly/tcc.nc", engine="netcdf4")
    tcc_data = tcc_data['tcc']

    tcc_data = xr.where(tcc_data > 0.01, tcc_data, np.nan)

    data['emissivity'] = (1-(np.exp(-158*data['tclw']/tcc_data)))

    lat_weights = np.cos(np.radians(data.latitude))

    histogram = xhist.histogram(data['cbh'], data['emissivity'], bins=25, weights=lat_weights,range=[[0, 17000], [0,1]], block_size=None)
    histo_numbers = xhist.histogram(data['cbh'], bins=25, weights=lat_weights, range=[0, 17000], block_size=None)

    emissivity_weights = histogram * histogram.emissivity_bin

    mean_emissivity = emissivity_weights.sum(dim='emissivity_bin')/histo_numbers
    
    variance = ((histogram * (histogram.emissivity_bin - mean_emissivity)**2)).sum(dim='emissivity_bin') / (histo_numbers - 1)


    #Calculate the standard error of the sample mean
    mean_std_error = np.sqrt(variance/histo_numbers)

    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(emiss_fit, histo_numbers.cbh_bin, mean_emissivity, p0=[1, 4000, 0], sigma=mean_std_error)

    ptest, ptestcov = curve_fit(piecewise_linear, histo_numbers.cbh_bin, mean_emissivity, p0=[8000, 0.1, 1e-4], sigma=mean_std_error)

    #regression = stats.linregress(histo_numbers.cbh_bin, mean_emissivity)
    x = np.linspace(0, 17000, 100)

    print(popt)
    print(ptest)

    fig, ax = plt.subplots(1, 1, layout='constrained')
    histogram.T.plot(ax=ax, norm=colors.LogNorm(vmin=0.1))
    ax.errorbar(histo_numbers.cbh_bin, mean_emissivity, yerr=10*mean_std_error, fmt='x',color='black')
    #ax.plot(x, popt[0]*np.exp(-x/popt[1])+popt[2], 'red')
    ax.plot(x, piecewise_linear(x, ptest[0], ptest[1], ptest[2]), 'black')

    x1 = histo_numbers.cbh_bin
    exp_est = popt[0]*np.exp(-x1/popt[1])+popt[2]
    lin_est = piecewise_linear(x1, ptest[0], ptest[1], ptest[2])

    print(xr.corr(mean_emissivity, exp_est, weights=1/mean_std_error))
    print(xr.corr(mean_emissivity, lin_est, weights=1/mean_std_error))
    ax.set_title('Emissivity Parameterisation')
    fig.savefig('emissivity_plot.png')
    return None

main()

