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
    L = (1-a3*vals[2])*L_clr + a3*vals[2]*sigma*vals[0]**4
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

def CN14(temp, vp, cf, a1 = -19.087, a2=66.064, a3=0.658, a4=3652, a5=0.249, a6=1.884):
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
    eps_clr = (1-a1*np.exp(-temp/a2)+a3*np.exp(-vp/a4))
    L_clr = eps_clr * sigma * temp**4
    L = L_clr * (1 + a5*cf**a6)
    return L

def deK20(temp, sw_clr, RH, a1 = -75.28, a2=82, a3=0.79, a4=-212.59, a5=189, a6=1.06):
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
    day = xr.where(sw_clr < 50, True, False)

    clr_branch = a1 + a2*RH + a3*sigma*temp**4
    cloud_branch = a4 + a5*RH + a6*sigma*temp**4
    
    day_L = xr.where(RH < 0.6, clr_branch, cloud_branch)
    night_L = xr.where(RH < 0.8, clr_branch, cloud_branch)

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
    Heff_lookup = SR21_data['Heff']
    q_lookup = SR21_data['q']
    tau_lookup = SR21_data['tau_eff_'+str(ppm)]
    
    q = calc_specific_humidity(ps, dpt)
    
    #Molar mass of air and ideal gas constant
    M = 28.97
    R = 8.3145
    # Calculate the WV density using the ideal gas law
    rho = q * ps * M/(R * temp)

    #Effective height
    Heff = calc_Heff(tcwv, rho, ps, theta)

    #Calculate the optical depth as a function of q and Heff
    tau_interp = sp.interp.RegularGridInterpolator((q_lookup, Heff_lookup), tau_lookup)

    tau = tau_interp(q, Heff)

    L_clr = sigma * temp**4 * (1 - np.exp(tau))
    return L_clr

def SR21_BE23(temp, dpt, tcwv, ps, RH, a1=0.9385, a2=-0.0114, a3=2.326, a4=0.0291):
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
    L_clr = SR21(temp, dpt, tcwv, ps, theta=40.3, ppm=400)
    e_sat = calc_vapor_pressure(temp)
    cloud_correction = (a1 + a2*e_sat)*RH**(a3 + a4*e_sat)
    L = (1 - cloud_correction)*L_clr + cloud_correction*sigma* temp **4
    return L

def B32_BE23(temp, dpt, RH, a1=0.5856, a2=0.00525, a3=1.043, a4=-0.0172, a5=3.061, a6=-0.0308):
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
    vp = calc_vapor_pressure(dpt)
    L_clr = (a1 + a2*np.sqrt(vp))*sigma*temp**4
    e_sat = calc_vapor_pressure(temp)
    cloud_correction = (a3 + a4*e_sat)*RH**(a5 + a6*e_sat)
    L = (1 - cloud_correction)*L_clr + cloud_correction*sigma* temp **4
    return L

def RMSE(est, true):
    '''Returns the root mean squared error
    Inputs:
        est - estimated values: xarray dataarray
        true - true values: xarray dataarray
    Returns:
        rmse - root mean squared error: float'''
    SE = (est - true)**2
    #TODO: add the latitudinal weighting
    weighting = np.cos(np.radians(SE.latitude))
    MSE = SE.weighted(weighting).mean()
    return np.sqrt(MSE)
    

def error_function(a, func, true_DLR, vals):
    est = func(a, vals)
    return RMSE(est, true_DLR)

def main():
    '''Main script for calculating the optimized coefficients
    for the models we are comparing against'''

    data = xr.open_mfdataset("/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/*.nc", engine="netcdf4").resample({'valid_time':'MS'}).mean().compute()
    data['vp_sat'] = calc_vapor_pressure(data['t2m'])
    data['vp'] = calc_vapor_pressure(data['d2m'])
    data['rh'] = data['vp']/data['vp_sat']
    #a1 = 59.38, a2 = 113.7, a3 = 96.96, a4 = 0.84
    
    true_DLR = data['avg_sdlwrf']
    random.seed(10)
    #res = sp.optimize.least_squares(error_function, x0=(59.38, 113.7, 96.96, 0.84), xtol=1e-5,
    #                                args=(true_DLR, data['t2m'], data['tcc'], data['tcwv']))
    function_dict = {DO98_UM75:[data['t2m'], data['tcc'], data['tcwv']], 
                     C14:[data['t2m'], data['rh'], data['tcc']]}
    x0_list = [np.array([59.38, 113.7, 96.96, 84]),
               np.array([-0.34, 0.336, 0.194, 0.213])]
    step_list = [5, 0.05]
    for func, x0, step in zip(function_dict, x0_list, step_list):
        res = sp.optimize.basinhopping(error_function, x0=x0, stepsize=step, niter_success=10, 
                                       minimizer_kwargs={'args':(func, true_DLR, function_dict[func])})
        print(res.x)
    return None

main()
