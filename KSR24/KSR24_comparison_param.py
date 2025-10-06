#Refined script to calculate the parameter values for the comparison for KSR24

import xarray as xr
import scipy as sp
import numpy as np
import random

#Stefan-Boltzmann constant as global variable
sigma = 5.67e-8

def calc_vapor_pressure(temp):
    '''Calculates the vapor pressure from the temperature.
    Inputting the 2m temperature will return the saturation vapor pressure,
    while inputting the 2m dew point temperature will return the vapor pressure.

    Applies the Huang (2018) formula https://doi.org/10.1175/JAMC-D-17-0334.1

    Inputs:
        temp - Temperature (K): xarray dataarray
    Returns:
        vp - Vapor Pressure (Pa): xarray dataarray'''
    temp_celsius = temp - 273.15

    pos_vap = (np.exp(34.494 - (4924.99/(temp_celsius + 237.1)))/
               (temp_celsius + 105)**1.57)
    neg_vap = (np.exp(43.494 - (6545.8/(temp_celsius + 278)))/
               (temp_celsius + 868)**2)
    vp = xr.where(temp_celsius > 0, pos_vap, neg_vap)
    return vp

def calc_specific_humidity(ps, dpt):
    '''Calculates specific humidity from the surface pressure and the
    dew point temperature.
    Inputs:
        ps - Surface Pressure (Pa): xarray dataarray
        dpt - Dew point temperature (Pa): xarray dataarray
    Returns:
        q - Specific humidity (kg/kg): xarray dataarray'''
    vp = calc_vapor_pressure(dpt)
    w = 0.622 * vp /(ps - vp)
    q = w / (1+w)
    return q

def calc_Heff(tcwv, rho, ps, theta):
    '''Calculates the effective height scale for the SR21 parameterisation
    Inputs:
        tcwv - Total column water vapor (kg/m^2): xarray dataarray
        rho - Near surface water vapor density (kg/m^3): xarray dataarray
        ps - surface pressure (Pa): xarray dataarray
        theta - Effective zenith angle (0-90): float
    Returns:
        Heff - Effective scale height (m): xarray dataarray'''
    H = tcwv / rho
    Heff = H / np.cos(np.radians(theta)) * (ps/1e5)**1.8
    return Heff
    

def DO98_UM75(a, vals):
    '''Estimates the downwelling longwave given the parameters
    and temperature, cloud fraction and precipitable water.
    Implements the clear-sky parameterisation of Dilley and O'Brien (1998)
    https://doi.org/10.1002/qj.49712454903 with the cloud correction of 
    Unsworth and Monteith (1975) https://doi.org/10.1002/qj.49710142703

    Inputs:
        vals - list containing:
            temp - Near-surface air temperature (K): xarray dataarray
            cf - Cloud fraction (0-1): xarray dataarray
            pw - column precipitable water (kg/m^2): xarray dataarray
        a - tuning parameters: floats
    Returns:
        L - estimated downwelling longwave (W/m^2)'''
    a3 = a[3]/100    #Factor 100 correction to ensure all parameters have the same order of magnitude
    L_clr = a[0] + a[1]*(vals[0]/273.16)**6 + a[2]*np.sqrt(vals[2]/25)
    L = (1-a3*vals[1])*L_clr + a3*vals[1]*sigma*vals[0]**4
    return L

def C14(a, vals):
    '''Estimates the downwelling longwave given the parameters
    and temperature, relative humidity and cloud fraction.
    Implements the method from Eq. 28 of Carmona et al. (2014)
    https://doi.org/10.1007/s00704-013-0891-3

    Inputs:
        vals - list containing:
            temp - Near-surface air temperature (K): xarray dataarray
            RH - relative humidity (0-1):xarray dataarray
            cf - Cloud fraction (0-1): xarray dataarray
        a - tuning parameters: floats
    Returns:
        L - estimated downwelling longwave (W/m^2)'''
    a1 = a[1]/100 #Factor 100 correction to ensure all parameters have the same order of magnitude
    L = (a[0] + a1*vals[0] + a[2]*vals[1] + a[3]*vals[2])* sigma * vals[0] ** 4
    return L

def CN14(a, vals):
    '''Estimates the downwelling longwave given the parameters
    and temperature, relative humidity and cloud fraction.
    Implements the method from Table 4 in Cheng and Nnadi (2014)
    https://doi.org/10.1155/2014/525148

    Inputs:
        temp - Near-surface air temperature (K): xarray dataarray
        vp - Vapor pressure (Pa): xarray dataarray
        cf - Cloud fraction (0-1): xarray dataarray
        a1 ~ a6 - tuning parameters: floats
    Returns:
        L - estimated downwelling longwave (W/m^2)'''
    #adjust params (want inputs to be same order of magnitude for basin hopping)
    a1 = a[1]*100

    eps_clr = (1+a[0]*np.exp(-vals[0]/a1)+a[2]*np.exp(-vals[1]/a[3]))
    L_clr = eps_clr * sigma * vals[0]**4
    L = L_clr * (1 + a[4]*vals[2]**a[5])
    return L

def deK20(a,vals):
    '''Estimates the downwelling longwave given the parameters
    and temperature, relative humidity and cloud fraction.
    Implements the method from deKok et al. (2020)
    https://doi.org/10.1002/joc.6249

    Inputs:
        temp - Near-surface air temperature (K): xarray dataarray
        sw_clr - Incoming clear-sky shortwave (W/m^2):xarray dataarray
        RH - Relative Humidity (0-1): xarray dataarray
        a1 ~ a6 - tuning parameters: floats
    Returns:
        L - estimated downwelling longwave (W/m^2)'''
    a2 = a[2] / 100
    a5 = a[5] / 100
    
    day = xr.where(vals[1] < 50, True, False)

    clr_branch = a[0] + a[1]*vals[2] + a2*sigma*vals[0]**4
    cloud_branch = a[3] + a[4]*vals[2] + a5*sigma*vals[0]**4
    
    day_L = xr.where(vals[2] < 0.6, clr_branch, cloud_branch)
    night_L = xr.where(vals[2] < 0.8, clr_branch, cloud_branch)

    L = xr.where(day == True, day_L, night_L)
    return L

def SR21(temp, dpt, tcwv, ps, theta=40.3, ppm = 400):
    '''Estimates the clear-sky downwelling longwave given the
    temperature, specific humidity, surface pressure.
    Implements the method from Shakespeare and Roderick (2021)
    https://doi.org/10.1002/qj.4176

    Inputs:
        temp - Near-surface air temperature (K): xarray dataarray
        dpt - Dew point temperature (K): xarray dataarray
        tcwv - Total column water vapor (kg/m^2):xarray dataarray
        ps - Surface Pressure (Pa): xarray dataarray
        theta - effective zenith angle (0-90 degrees): float
        ppm - CO2 concentration (ppm): float 
    Returns:
        L_clr - estimated clear-sky downwelling longwave (W/m^2)'''
    SR21_data = sp.io.loadmat("/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/data.mat")
    Heff_lookup = SR21_data['Heff'].flatten()
    q_lookup = SR21_data['q'].flatten()
    tau_lookup = SR21_data['tau_eff_'+str(ppm)]
    
    q = calc_specific_humidity(ps, dpt)
    
    #Molar mass of air and ideal gas constant
    M = 0.02897
    R = 8.3145
    # Calculate the WV density using the ideal gas law
    rho = q * ps * M/(R * temp)

    #Effective height
    Heff = calc_Heff(tcwv, rho, ps, theta)

    #Calculate the optical depth as a function of q and Heff
    tau_interp = sp.interpolate.RectBivariateSpline(q_lookup, Heff_lookup, tau_lookup)
    tau_func = lambda x,y,z: tau_interp(x, y, grid=z)

    tau = xr.apply_ufunc(tau_func, q, Heff, False, output_dtypes=[[float]])

    L_clr = sigma * temp**4 * (1 - np.exp(-tau))
    return L_clr

def SR21_BE23(a, vals):
    '''Estimates the downwelling longwave given the
    temperature, specific humidity, surface pressure.
    Implements the method from Bright and Eisner (2023) https://doi.org/10.1029/2023GL103790, 
    using the clear-sky formula of Shakespeare and Roderick (2021)
    https://doi.org/10.1002/qj.4176

    Inputs:
        temp - Near-surface air temperature (K): xarray dataarray
        dpt - Dew point temperature (K): xarray dataarray
        tcwv - Total column water vapor (kg/m^2):xarray dataarray
        ps - Surface Pressure (Pa): xarray dataarray
        RH - relative humidity (0-1): xarray dataarray 
        a1~a4 - tuning parameters: float
    Returns:
        L - estimated clear-sky downwelling longwave (W/m^2)'''
    a1 = a[1]/100
    a3 = a[3]/100
    L_clr = SR21(vals[0], vals[1], vals[2], vals[3], theta=40.3, ppm=400)
    e_sat = calc_vapor_pressure(vals[0])
    cloud_correction = (a[0] + a1*e_sat)*vals[4]**(a[2] + a3*e_sat)
    L = (1 - cloud_correction)*L_clr + cloud_correction*sigma* vals[0] **4
    return L

def B32_BE23(a, vals):
    '''Estimates the downwelling longwave given the
    temperature, specific humidity, surface pressure.
    Implements the method from Bright and Eisner (2023) https://doi.org/10.1029/2023GL103790, 
    using the clear-sky formula of Brunt (1932)
    https://doi.org/10.1002/qj.49705824704

    Inputs:
        temp - Near-surface air temperature (K): xarray dataarray
        dpt - Dew point temperature (K): xarray dataarray
        tcwv - Total column water vapor (kg/m^2):xarray dataarray
        ps - Surface Pressure (Pa): xarray dataarray
        RH - relative humidity (0-1): xarray dataarray 
        a1~a6 - tuning parameters: floats
    Returns:
        L - estimated clear-sky downwelling longwave (W/m^2)'''
    a1 = a[1]/100
    a3 = a[3]/100
    a4 = a[4]*10
    a5 = a[5]/10
    
    vp = calc_vapor_pressure(vals[1])
    L_clr = (a[0] + a1*np.sqrt(vp))*sigma*vals[0]**4
    e_sat = calc_vapor_pressure(vals[0])
    cloud_correction = (a[2] + a3*e_sat)*vals[2]**(a4 + a5*e_sat)
    L = (1 - cloud_correction)*L_clr + cloud_correction*sigma* vals[0] **4
    return L

def RMSE(est, true):
    '''Returns the root mean squared error
    Inputs:
        est - estimated values: xarray dataarray
        true - true values: xarray dataarray
    Returns:
        rmse - root mean squared error: float'''
    SE = (est - true)**2
    weighting = np.cos(np.radians(SE.latitude))
    MSE = SE.weighted(weighting).mean()
    return np.sqrt(MSE)
    

def error_function(a, func, true_DLR, vals):
    '''Calculate the error between the function estimated DLR
    and the true DLR from ERA5.
    Inputs:
        a - parameters we are trying to optimize (list of floats)
        func - the DLR estimation method
        true_DLR - the ERA5 downwelling longwave (W/m^2): xarray dataarray
        vals - the other physical data we are using: list of xarray dataarrays
    Returns:
        RMSE - the root mean squared error'''
    est = func(a, vals)
    return RMSE(est, true_DLR)

def main():
    '''Main script for calculating the optimized coefficients
    for the models we are comparing against'''

    #Open dataset and calculate all values we need to estimate DLR
    data = xr.open_mfdataset("/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/*.nc", engine="netcdf4").resample({'valid_time':'MS'}).mean().compute()
    data['vp_sat'] = calc_vapor_pressure(data['t2m'])
    data['vp'] = calc_vapor_pressure(data['d2m'])
    data['rh'] = data['vp']/data['vp_sat']
    
    true_DLR = data['avg_sdlwrf']
    random.seed(10)
    
    function_dict = {DO98_UM75:[data['t2m'], data['tcc'], data['tcwv']], 
                     C14:[data['t2m'], data['rh'], data['tcc']],
                     CN14:[data['t2m'], data['vp'], data['tcc']],
                     deK20:[data['t2m'], data['avg_sdswrfcs'], data['rh']],
                     SR21_BE23:[data['t2m'], data['d2m'], data['tcwv'], data['sp'], data['rh']],
                     B32_BE23:[data['t2m'], data['d2m'], data['rh']]}
    
    #Initial guesses (use values from the original papers)
    x0_list = [np.array([59.38, 113.7, 96.96, 84]),
               np.array([-0.34, 0.336, 0.194, 0.213]),
               np.array([-1.977, 149, 0.03066, 4.255, 0.316, 0.8506]),
               np.array([-75.28, 82, 79, -212.59, 189, 106]),
               np.array([0.9385, -1.14, 2.326, 2.91]),
               np.array([0.5856, 0.525, 1.043, -1.72, 0.3061, -0.308])]
    #Alter stepsizes depending on the size of the x0 we have
    step_list = [5, 0.05, 0.25, 7.5, 0.1, 0.05]
    for func, x0, step in zip(function_dict, x0_list, step_list):
        #Implement basin hopping for global minimum
        res = sp.optimize.basinhopping(error_function, x0=x0, stepsize=step, niter_success=10, 
                                       minimizer_kwargs={'args':(func, true_DLR, function_dict[func])})
        print(res.x)
    return None

main()
