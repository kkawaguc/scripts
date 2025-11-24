import xarray as xr
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import xhistogram.xarray as xhist
import matplotlib.colors as colors

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
    return xr.where(vp>610.94, a1/a2+273.15, b1/b2+273.15)

def tcwv_est(dpt, lsm):
    '''Estimates the total column water vapour using the equation from Reitan (1963).
    Separate parameters for land and ocean.
    Inputs:
        dpt: dew point temperature (K)
        lsm: land sea mask (0-1), 1 if land
    Returns:
        tcwv_est: estimate of the tcwv'''
    land = xr.where(lsm > 0.5, True, False)
    land_tcwv = np.exp(-15.6+0.06579*dpt)
    ocean_tcwv = np.exp(-14.3+0.061*dpt)
    return xr.where(land, land_tcwv, ocean_tcwv)

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
    Returns:
        L - estimated clear-sky downwelling longwave (W/m^2)'''
    vp = calc_vapor_pressure(dpt)
    L_clr = (0.5996 + 0.005174*np.sqrt(vp))*sigma*temp**4
    e_sat = calc_vapor_pressure(temp)
    cloud_correction = (1.013 - 0.0001594*e_sat)*RH**(2.433 + 0.0001635*e_sat)
    L = (1 - cloud_correction)*L_clr + cloud_correction*sigma* temp **4
    return L

def SR21_inversion(temp, dpt, tcwv, ps, ppm = 400):
    '''Function for the clear-sky correction to account for temperature inversions.
    In the case that an inversion exists, we assume that the tcwv is constant (as it is
    a direct ERA5 input), but the specific humidity scales (assume that the RH at near surface
    is equal to RH at inversion top). https://doi.org/10.1002/joc.7780
    Inputs:
        a: coefficients to be optimised
        vars: variables needed to calculate the clear-sky DLR
    Returns:
        adjusted_SR21: estimated DLR clear-sky'''
    temp_eff = xr.where(temp >= 270, temp, 0.6719*temp + 270 * (1-0.6719))

    vp = calc_vapor_pressure(dpt)
    sat_vap = calc_vapor_pressure(temp)
    rh = vp/sat_vap
    sat_vap_eff = calc_vapor_pressure(temp_eff)

    vp_eff = xr.where(temp >= 270, vp, rh * sat_vap_eff)
    dpt_eff = xr.where(temp >= 270, dpt, vp2dpt(vp_eff))

    ps_eff = 0.7592*ps
    return SR21(temp_eff, dpt_eff, tcwv, ps_eff, theta=52.7, ppm=ppm)

def calc_cloud_height(temp, RH):
    '''Calculate the cloud height, assuming the atmospheric lapse rate is
    equal to the dry adiabatic lapse rate of 9.8 K/km. Calculated from the
    Clausius-Clapeyron saturation height.
    Inputs:
        temp: 2m temperature (K)
        RH: near surface relative humidity (0-1)
    Returns:
        cb: estimated cloud base height (m)'''
    Gamma = 6.5e-3  #Dry adiabatic lapse rate
    M_w = 1.8016e-2 #Molar mass of water
    L_v = 2.5e6     #Latent heat of vaporisation
    R = 8.3145      #Ideal gas constant
    cb = -np.log(RH)*R*(temp**2)/(L_v*M_w*Gamma)
    cb = xr.where(cb < 0, 0, cb)
    return cb

def tcwv_adjustment(temp, dpt, ps, tcwv, cb):
    '''Calculate the TCWV adjustment due to having a cloud.
    Inputs:
        temp: 2m temperature (K)
        dpt: 2m dew point temperature (K)
        ps: surface pressure (Pa)
        tcwv: total column water vapor (kg/m^2)
        cb: cloud base height (m)
    Returns:
        cwv_frac = fraction of column water below the cloud'''
    q = calc_specific_humidity(ps, dpt)
    M = 0.02897
    R = 8.3145

    rho = q * ps * M/(R * temp)
    H = tcwv/rho

    cwv_frac = (1 - np.exp(-cb/H))
    return cwv_frac

def KSR24(temp, dpt, ps, cf, lsm, ppm=400):
    '''Calculates the all-sky estimation from Kawaguchi et al. (2024),
    with updated coefficients (area-weighted and including humidity in
    the clear-sky inversion calculation, and calculating ).'''
    R = 8.3145
    L_v = 2.5e6
    M_w = 1.8016e-2

    tcwv = tcwv_est(dpt, lsm)

    clear_sky = SR21_inversion(temp, dpt, tcwv, ps, ppm=ppm)

    RH = calc_vapor_pressure(dpt)/calc_vapor_pressure(temp)
    cb = calc_cloud_height(temp, RH)

    cwv_fraction = tcwv_adjustment(temp, dpt, ps, tcwv, cb)
    eff_cwv = tcwv*cwv_fraction

    below_cloud = SR21(temp, dpt, eff_cwv, 0.7592*ps, theta=52.7, ppm=ppm)
    below_cloud_emissivity = below_cloud/(sigma * temp ** 4)

    cloud_temp  = temp + np.log(RH) * R * temp **2 / (L_v * M_w)
    cloud_temp = xr.where(cloud_temp > temp, temp, cloud_temp)

    #land_cld_emiss = 0.9389 - 1.113e-4*cb
    #ocean_cld_emiss = 1.011 - 1.745e-4*cb

    #cld_emiss = land_cld_emiss * lsm + ocean_cld_emiss * (1-lsm)
    #cld_emiss = xr.where(cld_emiss < 0, 0, cld_emiss)

    #cld_emiss = 0.8705 * cld_emiss
    cld_emiss = 0.844 * (1.03*np.exp(-cb/4470) + 0.01574) #xr.where(cb >= 7409, 0.09157, -1.169e-4*cb + 0.09157+1.169e-04*7409)

    cloud_layer = (1-below_cloud_emissivity) * sigma * cld_emiss * cloud_temp**4
    return (1-cf) * clear_sky + cf * (below_cloud + cloud_layer)

def round_to_sig_figs(num, sig_figs):
    if num != 0:
        return round(num, -int(np.floor(np.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0

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

def plot_histogram(estimates, true_DLR, plot_name='', globe=True):
    '''Plotting routine for true_DLR/estimation histograms for KSR24
    Inputs:
        Estimates: list of xarray dataarrays
        true_DLR: xarray dataarrays
        plot_name: string, include the filetype suffix
        globe: whether to area-weight all terms, true if reanalysis, false if in-situ data'''

    fig, ax = plt.subplots(nrows=3, ncols=3, layout="constrained", figsize = (10, 8))

    model_names = ["New", "DO98 + UM75",  "C14", "CN14", "deK20",  "SR21+BE23", "B32+BE23"]

    if globe == True:
        weights = np.cos(np.radians(true_DLR.latitude))
    else:
        weights = xr.ones_like(true_DLR)

    for i in range(len(estimates)):
        error = estimates[i] - true_DLR
        print(error.head())
        RMSE = round_to_sig_figs(float(np.sqrt((error**2).weighted(weights).mean())), 3)
        MBE = round_to_sig_figs(float(error.weighted(weights).mean()), 3)
        r = round_to_sig_figs(float(xr.corr(estimates[i], true_DLR, weights=weights)), 3)

        histo_data = xhist.histogram(true_DLR.rename('True DLR'), estimates[i].rename('estimate'), bins=50, range=[[0, 650], [0,650]], weights=weights, block_size=15000000)

        if i == 0:
            histo_data.T.plot(ax=ax[0, 1], norm=colors.LogNorm(vmin=0.1), add_colorbar=False)
            ax[0,1].text(25, 600, model_names[i], weight="bold")
            ax[0,1].text(25, 550, "RMSE =" + str(RMSE))
            ax[0,1].text(25, 500, "MBE =" + str(MBE))
            ax[0,1].text(25, 450, "r =" + str(r))
            ax[0, 1].set_yticks([0, 200, 400, 600])
            ax[0,1].plot((0, 650), (0, 650), color='k')
        else:
            histo_data.T.plot(ax=ax[(i-1)//3+1, (i-1)%3], norm=colors.LogNorm(vmin=0.1), add_colorbar=False)
            ax[(i-1)//3+1, (i-1)%3].text(25, 600, model_names[i], weight="bold")
            ax[(i-1)//3+1, (i-1)%3].text(25, 550, "RMSE =" + str(RMSE))
            ax[(i-1)//3+1, (i-1)%3].text(25, 500, "MBE =" + str(MBE))
            ax[(i-1)//3+1, (i-1)%3].text(25, 450, "r =" + str(r))
            ax[(i-1)//3+1, (i-1)%3].plot((0, 650), (0, 650), color='k')
    
    
        ax[(i-1)//3+1, (i-1)%3].set_yticks([0, 200, 400, 600])
        ax[(i-1)//3+1, (i-1)%3].set_ylabel("")
        ax[(i-1)//3+1, (i-1)%3].set_xlabel("")

    ax[2, 0].set_xlabel("Observed DLR ($\\mathrm{W/m^2}$)")
    ax[2, 1].set_xlabel("Observed DLR ($\\mathrm{W/m^2}$)")
    ax[2, 2].set_xlabel("Observed DLR ($\\mathrm{W/m^2}$)")

    ax[0, 0].set_ylabel("Predicted DLR ($\\mathrm{W/m^2}$)")
    ax[1, 0].set_ylabel("Predicted DLR ($\\mathrm{W/m^2}$)")
    ax[2, 0].set_ylabel("Predicted DLR ($\\mathrm{W/m^2}$)")

    ax[0, 1].set_xlabel("Observed DLR ($\\mathrm{W/m^2}$)")
    ax[0, 1].set_ylabel("Predicted DLR ($\\mathrm{W/m^2}$)")

    fig.delaxes(ax[0,0])
    fig.delaxes(ax[0,2])

    fig.savefig('Plots/'+plot_name)
    return None

def plot_main3(estimates, true_DLR, temp_data, pressure_data, plot_name='', globe=True):
    '''Plot the RMSE based on the temperature, pressure and true DLR bins
    Inputs:
        estimates: list of xarray dataarrays
        true_DLR: xarray dataarray
        temp_data: xarray dataarray
        pressure_data: xarray dataarray'''
    true_DLR = true_DLR.rename('Downwelling LW')
    temp_data = temp_data.rename('Temperature')
    pressure_data = pressure_data.rename('Pressure')
    
    nbins = 25

    tmin = float(temp_data.min())
    tmax = float(temp_data.max())
    t_bins = np.linspace(tmin, tmax, nbins)
    t_bin_plot = np.linspace(np.mean(t_bins[:2]), np.mean(t_bins[-2:]), nbins-1)

    Pmin = float(pressure_data.min())
    Pmax = float(pressure_data.max())

    P_bins = np.linspace(Pmin, Pmax, nbins)
    P_bin_plot = np.linspace(np.mean(P_bins[:2]), np.mean(P_bins[-2:]), nbins-1)

    LWmin = float(true_DLR.min())
    LWmax = float(true_DLR.max())

    LW_bins = np.linspace(LWmin, LWmax, nbins)
    LW_bin_plot = np.linspace(np.mean(LW_bins[:2]), np.mean(LW_bins[-2:]), nbins-1)

    fig, ax = plt.subplots(nrows=3, ncols=1, layout="constrained", figsize=(4, 8))

    results_cut = [estimates[0], estimates[1], estimates[2], estimates[5], estimates[6]]
    model_names_cut = ["New", "DO98 + UM75",  "C14", "SR21+BE23", "B32+BE23"]

    if globe == True:
        loc_weights = np.cos(np.radians(true_DLR.latitude))
    else:
        loc_weights = xr.ones_like(true_DLR)

    for i in range(len(results_cut)):
        error = results_cut[i] - true_DLR
        n1 = xhist.histogram(true_DLR, bins=LW_bins, weights=loc_weights.fillna(0), block_size=15000000)
        sy1 = xhist.histogram(true_DLR, bins=LW_bins, weights=(error**2 * loc_weights).fillna(0), block_size=15000000)

        if i == 0:
            ax[0].plot(LW_bin_plot, np.sqrt(sy1/n1), "k", zorder=10)
        else:
            ax[0].plot(LW_bin_plot, np.sqrt(sy1/n1), zorder=i)
        if globe == True:
            ax[0].set_title("a) Reanalysis Downwelling Longwave")
        else:
            ax[0].set_title("b) In-situ Downwelling Longwave")
        ax[0].set_ylabel("RMSE ($\\mathrm{W/m^2}$)")
        ax[0].set_xlabel("Downwelling Longwave ($\\mathrm{W/m^2}$)")

        ax[0].set_ylim([0, 60])
        ax[0].set_xticks([100, 200, 300, 400, 500])
    
        n2 = xhist.histogram(temp_data, bins=t_bins, weights=loc_weights.fillna(0), block_size=15000000)
        sy2 = xhist.histogram(temp_data, bins=t_bins, weights=(error**2 * loc_weights).fillna(0), block_size=15000000)
    
        if i== 0:
            ax[1].plot(t_bin_plot, np.sqrt(sy2/n2), "k", zorder=10)
        else:
            ax[1].plot(t_bin_plot, np.sqrt(sy2/n2), zorder=i)
        if globe == True:
            ax[1].set_title("c) Reanalysis Temperature")
        else:
            ax[1].set_title("d) In-situ Temperature")
        ax[1].set_ylabel("RMSE ($\\mathrm{W/m^2}$)")
        ax[1].set_xlabel("Temperature (K)")

        ax[1].set_ylim([0, 60])
    
        n3 = xhist.histogram(pressure_data, bins=P_bins, weights=loc_weights.fillna(0), block_size=15000000)
        sy3 = xhist.histogram(pressure_data, bins=P_bins, weights=(error**2 * loc_weights).fillna(0), block_size=15000000)

        a = sy3/n3
        mask = np.isnan(a)

        # Use numpy's interp() function to interpolate the NaN values
        a[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), a[~mask])

        if i== 0:
            ax[2].plot(P_bin_plot, np.sqrt(a), "k", zorder=10)
        else:
            ax[2].plot(P_bin_plot, np.sqrt(a), zorder=i)
        ax[2].set_title("f) In-situ Pressure (Pa)")
        ax[2].set_ylabel("RMSE ($\\mathrm{W/m^2}$)")
        ax[2].set_xlabel("Pressure (Pa)")

        ax[2].set_ylim([0, 60])

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    fig.legend(model_names_cut, bbox_to_anchor=[1.1, 0.5], loc="center left")
    fig.savefig('Plots/' + plot_name)
    return None