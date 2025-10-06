import xarray as xr
import scipy as sp
import numpy as np

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

def vp2dpt(vp):
    '''Calculates the dew point from the vapor pressure using
    the improved Magnus formula with respect to water and ice
    Taken from Huang (2018).
    Inputs:
        vp: vapor pressure in Pa
    Returns:
        dpt: 2m dew point temperature'''
    a1 = 243.04*np.log(vp/610.94)
    a2 = 17.625-np.log(vp/610.94)
    b1 = 273.86*np.log(vp/611.21)
    b2 = 22.587-np.log(vp/611.21)
    return np.where(vp>610.94, a1/a2+273.15, b1/b2+273.15)

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
    

def DO98_UM75(temp, cf, pw):
    '''Estimates the downwelling longwave given the parameters
    and temperature, cloud fraction and precipitable water.
    Implements the clear-sky parameterisation of Dilley and O'Brien (1998)
    https://doi.org/10.1002/qj.49712454903 with the cloud correction of 
    Unsworth and Monteith (1975) https://doi.org/10.1002/qj.49710142703

    Inputs:
        temp - Near-surface air temperature (K): xarray dataarray
        cf - Cloud fraction (0-1): xarray dataarray
        pw - column precipitable water (kg/m^2): xarray dataarray
        a - tuning parameters: floats
    Returns:
        L - estimated downwelling longwave (W/m^2)'''
    L_clr = 36.07 + 126.3*(temp/273.16)**6 + 98.02*np.sqrt(pw/25)
    L = (1-0.6436*cf)*L_clr + 0.6436*cf*sigma*temp**4
    return L

def C14(temp, RH, cf):
    '''Estimates the downwelling longwave given the parameters
    and temperature, relative humidity and cloud fraction.
    Implements the method from Eq. 28 of Carmona et al. (2014)
    https://doi.org/10.1007/s00704-013-0891-3

    Inputs:
        temp - Near-surface air temperature (K): xarray dataarray
        RH - relative humidity (0-1):xarray dataarray
        cf - Cloud fraction (0-1): xarray dataarray
        a1 ~ a4 - tuning parameters: floats
    Returns:
        L - estimated downwelling longwave (W/m^2)'''
    L = (-0.05835 + 0.002448*temp + 0.2003*RH + 0.1068*cf)* sigma * temp ** 4
    return L

def CN14(temp, vp, cf):
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
    #eps_clr = (1-6.437*np.exp(-temp/78.71)-0.2826*np.exp(-vp/1.802e-2))
    #L_clr = eps_clr * sigma * temp**4
    #L = L_clr * (1 + 0.1214*cf**3.388)
    eps_clr = (1-0.163*np.exp(-temp/763700)-0.1504*np.exp(-vp/2197))
    L_clr = eps_clr * sigma * temp**4
    L = L_clr * (1 + 0.2149*cf**0.9609)
    return L

def deK20(temp, sw_clr, RH):
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

    clr_branch = -126.1 + 114.5*RH + 0.9687*sigma*temp**4
    cloud_branch = -99.74 + 86.73*RH + 0.9706*sigma*temp**4
    
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

def SR21_BE23(temp, dpt, tcwv, ps, RH):
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
    #cloud_correction = (0.9364 + 1.115e-5*e_sat)*RH**(0.1392 - 1.190e-5*e_sat)
    cloud_correction = (0.9142 - 1.457e-4*e_sat)*RH**(1.947 + 1.18e-4*e_sat)
    L = (1 - cloud_correction)*L_clr + cloud_correction*sigma* temp **4
    return L

def B32_BE23(temp, dpt, RH):
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
    L_clr = (0.5996 + 0.005174*np.sqrt(vp))*sigma*temp**4
    e_sat = calc_vapor_pressure(temp)
    cloud_correction = (1.013 - 0.0001594*e_sat)*RH**(2.433 + 0.0001635*e_sat)
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
    weighting = np.cos(np.radians(SE.latitude))
    MSE = SE.weighted(weighting).mean(('valid_time', 'latitude', 'longitude'))
    return np.sqrt(MSE)
