import xarray as xr
from KSR24_functions import *
import random
import scipy as sp
import numpy as np

init = [0.869, 0.823]

def clear_sky(a, temp, dpt, tcwv, ps):
    '''Function for the clear-sky correction to account for temperature inversions.
    In the case that an inversion exists, we assume that the tcwv is constant (as it is
    a direct ERA5 input), but the specific humidity scales (assume that the RH at near surface
    is equal to RH at inversion top). https://doi.org/10.1002/joc.7780
    Inputs:
        a: coefficients to be optimised
        vars: variables needed to calculate the clear-sky DLR
    Returns:
        adjusted_SR21: estimated DLR clear-sky'''
    temp_eff = xr.where(temp >= 270, temp, a[1]*temp + 270 * (1-a[1]))

    vp = calc_vapor_pressure(dpt)
    sat_vap = calc_vapor_pressure(temp)
    rh = vp/sat_vap
    sat_vap_eff = calc_vapor_pressure(temp_eff)

    vp_eff = xr.where(temp >= 270, vp, rh * sat_vap_eff)
    dpt_eff = xr.where(temp >= 270, dpt, vp2dpt(vp_eff))

    ps_eff = a[0]*ps
    return SR21(temp_eff, dpt_eff, tcwv, ps_eff, theta=52.7, ppm=400)

def tcwv_est(a, dpt, lsm):
    '''Estimates the tcwv using the equation from Reitan (1963).
    Separate parameters for land and ocean.'''
    land = xr.where(lsm > 0.5, True, False)
    land_tcwv = np.exp(a[0]+a[1]*dpt)
    ocean_tcwv = np.exp(a[2]+a[3]*dpt)
    return xr.where(land, land_tcwv, ocean_tcwv)

def error_function(a, true_DLR_cs, temp, dpt, tcwv, ps):
    '''Error function to optimise'''
    est = clear_sky(a, temp, dpt, tcwv, ps)
    return RMSE(est, true_DLR_cs)

def tcwv_error(a, true_tcwv, dpt, lsm):
    est = tcwv_est(a, dpt, lsm)
    weights = np.cos(np.radians(true_tcwv.latitude))
    tcwv_mean = true_tcwv.weighted(weights).mean()
    return RMSE(est, true_tcwv)/tcwv_mean

def main():
    '''Main script for calculating the optimized coefficients
    for the clear-sky adjustment'''

    #Open dataset and calculate all values we need to estimate DLR
    data = xr.open_mfdataset("/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/*.nc", engine="netcdf4").resample({'valid_time':'MS'}).mean().compute()
    
    true_DLR = data['avg_sdlwrfcs']
    random.seed(10)
    
    #Implement least squares for global minimum
    res = sp.optimize.least_squares(error_function, x0=np.array([0.869, 0.823]), bounds=((0, 0), (1, 1)), 
                                    args=(true_DLR, data['t2m'], data['d2m'], data['tcwv'], data['sp']))
    print(res.x)

    #Implement least squares for global minimum
    res = sp.optimize.least_squares(tcwv_error, x0=np.array([-15.6, 0.0656, -14.3, 0.0605]), 
                                    args=(data['tcwv'], data['d2m'], data['lsm']))
    print(res.x)
    return None

main()

#Returned values: [0.75915116 0.67185189]