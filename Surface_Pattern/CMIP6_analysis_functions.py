#import xskillscore as xs
import xarray as xr
import numpy as np
#import global_land_mask as lm
#import easyclimate.core.utility as utility
#import easyclimate.field.boundary_layer.aerobulk as aerobulk
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import scipy as sp

def calc_land_mask(data):
    '''Calculate a land mask
    Inputs:
        data - any gridded data to calculate the mask from
    Returns:
        mask - land mask with ocean == True'''
    lon_pts, lat_pts = np.meshgrid(data.lon, data.lat)
    lon_pts = np.where(lon_pts > 180, lon_pts - 360, lon_pts)
    lon_transform = xr.where(data.lon > 180, data.lon - 360, data.lon)
    land_mask = lm.globe.is_ocean(lat_pts, lon_pts)
    land_mask = xr.DataArray(land_mask, coords = {"lat": data.lat, "lon": lon_transform})
    land_mask = utility.transfer_xarray_lon_from180TO360(land_mask)
    return land_mask

def calc_dHF(data, data_climo, climo_variables=[]):
    '''Calculates the anomalies in turbulent heat fluxes associated
    with changes in specific variables:
    Inputs:
        data - the time-varying climate model data
        data_climo - the climatological data
        climo_variables - a list of strings specifying which variables should be taken from the climo data
    Returns:
        dHF - the anomalous turbulent heat flux'''
    sel_merge_data = selective_merge(data, data_climo, climo_variables)
    HF = aerobulk.calc_turbulent_fluxes_without_skin_correction(sel_merge_data["ts"], "degK", sel_merge_data["tas"], "degK", 
                                                                sel_merge_data["huss"], "kg/kg", sel_merge_data["uas"], sel_merge_data["vas"], 
                                                                sel_merge_data["ps"], "Pa")
    HF_climo = aerobulk.calc_turbulent_fluxes_without_skin_correction(data_climo["ts"], "degK", data_climo["tas"], "degK", 
                                                                      data_climo["huss"], "kg/kg", data_climo["uas"], data_climo["vas"], 
                                                                      data_climo["ps"], "Pa")
    dHF = HF - HF_climo
    return dHF

def selective_merge(A: xr.Dataset, B: xr.Dataset, use_B_indices: list):
    """
    Combines variables from A and B based on index selection.
    Returns a new xarray.Dataset.
    """
    result = xr.Dataset()

    # Infer number of variables from A
    var_names = [var for var in A.data_vars]
    for var in var_names:
        if var in use_B_indices:
            result[var] = B[var]
        else:
            result[var] = A[var]
    return result

def calc_fdbk(data, tas, yrs=30):
    '''
    Calculate the time-varying feedback for the given variable at relative to 
    changes in global mean surface temperature. 
    Inputs:
        data: the variable for which the feedback is being calculated
        tas: the time-series of GMST
        yrs: number of years in each rolling window
    Returns:
        fdbk: time-varying feedback
    '''
    
    if 'lat' in data.dims:
        data_roll = data.rolling(dim={"time":yrs}, center=True).construct(window_dim="regress_time").chunk({'lat':18, 'lon':36})
    else:
        data_roll = data.rolling(dim={"time":yrs}, center=True).construct(window_dim="regress_time")
    tas_roll = tas.rolling(dim={"time":yrs}, center=True).construct(window_dim="regress_time")
    fdbk = xs.linslope(tas_roll, data_roll, dim='regress_time').dropna(dim='time', how='all')
    if 'lat' in fdbk.dims:
        return fdbk.chunk({'time':-1, 'lat':36, 'lon':36})
    else:
        return fdbk.compute()


def calc_Fueglistaler_idx_norm(ocean_ts):
    '''Calculates the normalised Fueglistaler (2019) SST# time-series
    Inputs:
        Ocean_ts: Sea surface temperature
    Returns:
        Fueglistaler_idx_norm: Normalised SST#'''

    tropical_SST = ocean_ts.sel({'lat':slice(-30, 30)}).compute()
    weights = np.cos(np.radians(tropical_SST.lat))
    Fueglistaler_threshold = tropical_SST.weighted(weights).quantile(0.7, dim=('lat', 'lon'))
    Fueglistaler_idx_raw = xr.where(tropical_SST > Fueglistaler_threshold, tropical_SST, np.nan).weighted(weights).mean(('lat', 'lon')) - tropical_SST.weighted(weights).mean(('lat', 'lon'))

    Fueglistaler_idx_norm = (Fueglistaler_idx_raw-Fueglistaler_idx_raw.mean('time'))/Fueglistaler_idx_raw.std(dim='time')
    return Fueglistaler_idx_norm

def calc_Fueglistaler_idx(ocean_ts):
    '''Calculates the Fueglistaler (2019) SST# time-series
    Inputs:
        Ocean_ts: Sea surface temperature
    Returns:
        Fueglistaler_idx: SST#'''

    tropical_SST = ocean_ts.sel({'lat':slice(-30, 30)}).compute()
    weights = np.cos(np.radians(tropical_SST.lat))
    Fueglistaler_threshold = tropical_SST.weighted(weights).quantile(0.7, dim=('lat', 'lon'))
    Fueglistaler_idx = xr.where(tropical_SST > Fueglistaler_threshold, tropical_SST, np.nan).weighted(weights).mean(('lat', 'lon')) - tropical_SST.weighted(weights).mean(('lat', 'lon'))
    return Fueglistaler_idx

def remove_time_mean(x):
    '''Remove time mean for purposes of applying climatology removal'''
    return x - x.mean(dim='time')

def facetplot(data, dims_to_reduce=None, facet_dim=None, title='', **kwargs):
    """
    Plotting routine for a facetplot.

    Parameters:
        data: xarray.DataArray or xarray.Dataset
        dims_to_reduce: str or list/tuple of str, dimensions to average over
        facet_dim: str, dimension to facet along
        title: title of the plot
        **kwargs: additional keyword arguments passed to .plot()
    """
    # Ensure dims_to_reduce is a tuple if provided
    if dims_to_reduce is not None and isinstance(dims_to_reduce, list):
        dims_to_reduce = tuple(dims_to_reduce)

    # Pad longitude and assign new coordinates
    data_padded = data.pad({'lon': 1}, mode='wrap').assign_coords(
        {'lon': np.linspace(-1.25, 361.25, data.sizes['lon'] + 2)}
    )

    # Extract and override subplot_kws if needed
    subplot_kws = kwargs.pop('subplot_kws', {})
    subplot_kws.setdefault('projection', ccrs.Robinson(central_longitude=180))

    # Apply mean reduction if specified
    if dims_to_reduce is not None:
        data_padded = data_padded.mean(dims_to_reduce)

    # Plot with all remaining kwargs
    plot = data_padded.plot(
        col=facet_dim,
        subplot_kws=subplot_kws,
        transform=ccrs.PlateCarree(),
        **kwargs
    )

    plt.suptitle(title)

    # Add coastlines
    if facet_dim == None:
        plot.axes.coastlines()
    else:
        for ax in plot.axs.flatten():
            ax.coastlines()

    plt.show()
    return None
    

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

def q2dpt(q, ps):
    '''Calculates the dew point from the specific humidity'''
    w = q/1-q
    vp = w * ps/ (0.622 + w)
    dpt = vp2dpt(vp)
    return dpt

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
    land_tcwv = np.exp(-15.6+0.0656*dpt)
    ocean_tcwv = np.exp(-14.3+0.0605*dpt)
    return lsm * land_tcwv + (1-lsm)*ocean_tcwv

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

def SR21(temp, q, tcwv, ps, theta=40.3, ppm = 280):
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
    sigma = 5.67e-8
    SR21_data = sp.io.loadmat("/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/data.mat")
    Heff_lookup = SR21_data['Heff']
    q_lookup = SR21_data['q']
    if ppm == 280:
        tau_lookup_200 = SR21_data['tau_eff_200']
        tau_lookup_400 = SR21_data['tau_eff_400']
        tau_lookup = 0.6 * tau_lookup_200 + 0.4*tau_lookup_400
    
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

    tau = xr.apply_ufunc(tau_func, q, Heff, False, output_dtypes=[float], dask='parallelized')

    L_clr = sigma * temp**4 * (1 - np.exp(-tau))
    return L_clr

def SR21_inversion(temp, q, tcwv, ps, ppm = 280):
    '''Function for the clear-sky correction to account for temperature inversions.
    In the case that an inversion exists, we assume that the tcwv is constant (as it is
    a direct ERA5 input), but the specific humidity scales (assume that the RH at near surface
    is equal to RH at inversion top). https://doi.org/10.1002/joc.7780
    Inputs:
        a: coefficients to be optimised
        vars: variables needed to calculate the clear-sky DLR
    Returns:
        adjusted_SR21: estimated DLR clear-sky'''
    temp_eff = xr.where(temp >= 270, temp, 0.823*temp + 270 * (1-0.823))

    #vp = calc_vapor_pressure(dpt)
    #sat_vap = calc_vapor_pressure(temp)
    #rh = vp/sat_vap
    #sat_vap_eff = calc_vapor_pressure(temp_eff)

    #vp_eff = xr.where(temp >= 270, vp, rh * sat_vap_eff)
    #dpt_eff = xr.where(temp >= 270, dpt, vp2dpt(vp_eff))

    ps_eff = 0.869*ps
    return SR21(temp_eff, q, tcwv, ps_eff, theta=52.7, ppm=ppm)