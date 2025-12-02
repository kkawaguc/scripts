import xarray as xr
from KSR24_functions import *
import matplotlib.pyplot as plt 

sigma = 5.67e-8

def main():
    '''Main script for calculating the optimized coefficients
    for the models we are comparing against'''

    data = xr.open_mfdataset("/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/*.nc", engine="netcdf4").resample({'valid_time':'MS'}).mean().compute()
    data['vp_sat'] = calc_vapor_pressure(data['t2m'])
    data['vp'] = calc_vapor_pressure(data['d2m'])
    data['rh'] = data['vp']/data['vp_sat']
    
    true_DLR = data['avg_sdlwrf']
    
    est_DLR = xr.concat([KSR24(data['t2m'], data['d2m'], data['sp'], data['tcc'], data['lsm']),
               DO98_UM75(data['t2m'], data['tcc'], data['tcwv']),
               C14(data['t2m'], data['rh'], data['tcc']),
               CN14(data['t2m'], data['vp'], data['tcc']),
               deK20(data['t2m'], data['avg_sdswrfcs'], data['rh']),
               SR21_BE23(data['t2m'], data['d2m'], data['tcwv'], data['sp'], data['rh']),
               B32_BE23(data['t2m'], data['d2m'], data['rh'])], dim='model')
    
    est_DLR = est_DLR.assign_coords({'model':['KSR24', 'DO98-UM75', 'C14', 'CN14', 'deK20',
                                              'SR21-BE23', 'B32-BE23']})
    
    plot = (est_DLR.mean('valid_time')-true_DLR.mean('valid_time')).plot(col='model', col_wrap=3)
    plot.fig.savefig('test_plot.png')
    
    plot_histogram(est_DLR, true_DLR, 'parameterisation.png', globe=True)

    print(RMSE(true_DLR, est_DLR))
    print((est_DLR-true_DLR).weighted(np.cos(np.radians(est_DLR.latitude))).mean(('valid_time', 'latitude', 'longitude')))
    #array([ 7.90113688,  7.23587302, 15.25444936,  9.41461849, 10.35658033,
    #    8.86408141])
    return None

def main2():
    data = xr.open_mfdataset("/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/*.nc", engine="netcdf4").resample({'valid_time':'MS'}).mean().compute()
    data['vp_sat'] = calc_vapor_pressure(data['t2m'])
    data['vp'] = calc_vapor_pressure(data['d2m'])
    data['rh'] = data['vp']/data['vp_sat']
    
    true_DLR = data['avg_sdlwrf']
    
    alpha_set = np.linspace(0.88, 0.9, 21)
    for alpha in alpha_set:
        est_DLR = KSR24(data['t2m'], data['d2m'], data['sp'], data['tcc'], data['lsm'], ppm=400, alpha=alpha) #SR21(data['t2m'], data['d2m'], data['tcwv'], data['sp'])
        print(alpha)
        print(RMSE(true_DLR, est_DLR))
    return None

main()