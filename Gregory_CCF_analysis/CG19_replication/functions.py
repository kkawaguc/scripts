import numpy as np
import xesmf as xe

Lhvap = 2.5E6    # Latent heat of vaporization (J / kg)
Lhsub = 2.834E6   # Latent heat of sublimation (J / kg)
cp = 1004.     # specific heat at constant pressure for dry air (J / kg / K)
Rd = 287.         # gas constant for dry air (J / kg / K)
kappa = 2/7
Rv = 461.       # gas constant for water vapor (J / kg / K)
g = 9.8          # gravitational acceleration (m / s**2)
eps = Rd/Rv

def potential_temperature(T,p):
    """Compute potential temperature for an air parcel.

    Input:  T is temperature in Kelvin
            p is pressure in mb or hPa
    Output: potential temperature in Kelvin.

    """
    theta = T*(1000/p)**kappa
    return theta

def lifting_condensation_level(TS, RH):
    '''Compute the Lifiting Condensation Level (LCL) for a given temperature and relative humidity

    Inputs:  T is temperature in Kelvin
            RH is relative humidity (dimensionless)

    Output: LCL in meters

    This is height (relative to parcel height) at which the parcel would become saturated during adiabatic ascent.

    Based on approximate formula from Lawrence (2005 AMS) as given by Romps (2017 JAS)

    For an exact formula see Romps (2017 JAS), doi:10.1175/JAS-D-17-0102.1
    '''
    return (20 + (TS - 273.15)/5)*(100 - 80)

def qsat(T,p):
    """Compute saturation specific humidity as function of temperature and pressure.

    Input:  T is temperature in Kelvin
            p is pressure in hPa or mb
    Output: saturation specific humidity (dimensionless).

    """
    import numpy as np
    es = (1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*(T-273.15)/(240.97+(T-273.15)))
    wsl = 1000*.622*es/(p-es) # saturation mixing ratio w.r.t. liquid water (g/kg)
    es = (1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*(T-273.15)/(272.55+(T-273.15)))
    wsi = 1000*.622*es/(p-es) # saturation mixing ratio w.r.t. ice (g/kg)
    ws = np.where(T<273.15, wsi, wsl)
    wskg = ws/1000
    #es = clausius_clapeyron(T)
    #q = eps * es / (p - (1 - eps) * es )
    return wskg/(1+wskg)

def estimated_inversion_strength(T0,T700, ps):
    '''Compute the Estimated Inversion Strength or EIS,
    following Wood and Bretherton (2006, J. Climate)

    Inputs: T0 is surface temp in Kelvin
           T700 is air temperature at 700 hPa in Kelvin
           ps is surface pressure in hPa

    Output: EIS in Kelvin

    EIS is a normalized measure of lower tropospheric stability acccounting for
    temperature-dependence of the moist adiabat.
    '''
    # Interpolate to 850 hPa
    T850 = (T0+T700)/2.;
    # Assume 80% relative humidity to compute LCL, appropriate for marine boundary layer
    LCL = lifting_condensation_level(T0, 0.8)
    # Lower Tropospheric Stability (theta700 - theta0)
    LTS = T700*(1000/700)**kappa - T0*(1000/ps)**kappa
    #  Gammam  = -dtheta/dz is the rate of potential temperature decrease along the moist adiabat
    #  in K / m
    Gammam = (g/cp*(1.0 - (1.0 + Lhvap*qsat(T850,850) / Rd / T850) /
             (1.0 + Lhvap**2 * qsat(T850,850)/ cp/Rv/T850**2)))
    #  Assume exponential decrease of pressure with scale height given by surface temperature
    z700 = (Rd*T0/g)*np.log(ps/700.)
    return LTS - Gammam*(z700 - LCL)

def lower_tropospheric_stability(T0,T700, ps):
    '''Compute the Estimated Inversion Strength or LTS,
    following Klein and Hartmann (1993)

    Inputs: T0 is surface temp in Kelvin
           T700 is air temperature at 700 hPa in Kelvin
           ps is surface pressure in hPa

    Output: EIS in Kelvin

    EIS is a normalized measure of lower tropospheric stability acccounting for
    temperature-dependence of the moist adiabat.
    '''
    # Lower Tropospheric Stability (theta700 - theta0)
    LTS = T700*(1000/700)**kappa - T0*(1000/ps)**kappa
    return LTS

def regrid(data):
    ds_out = xe.util.grid_global(5, 5, True, lon1=180)
    regridder = xe.Regridder(data, ds_out, 'bilinear', periodic=True)
    return regridder(data)