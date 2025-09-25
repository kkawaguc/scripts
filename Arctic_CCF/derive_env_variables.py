import xarray as xr
import scipy as sp
import numpy as np

#Code to calculate the tropopause temperature and lower tropospheric stability

def calc_tropo_temp(ta):
    '''Estimates the tropopause temperature using the atmospheric column.
    Calculates the value by interpolating a cubic spline to 100 equally spaced
    pressure levels in plausible tropopause pressure levels (50-400hPa), 
    then taking the minimum temperature.
    Inputs:
        ta (K) - temperature of the column at ERA5 pressure levels: xarray dataarray with 
        pressure_level coordinate
    Returns:
        tropopause_temp (K) - estimated tropopause temperature'''
    plev = np.array([1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,  775.,
        750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,  350.,  300.,
        250.,  225.,  200.,  175.,  150.,  125.,  100.,   70.,   50.,   30.,
         20.,   10.,    7.,    5.,    3.,    2.,    1.])
    T_interp = sp.interpolate.CubicSpline(plev[::-1], np.flip(ta, axis=-1), axis=-1)
    interp_locs = np.linspace(50, 400, 100)
    T_interpolated = T_interp(interp_locs)
    tropopause_temp = min(T_interpolated)
    return tropopause_temp

def calc_tropo_gridded(vert_temp):
    tropopause_gridded = xr.apply_ufunc(calc_tropo_temp, vert_temp, input_core_dims=[['pressure_level']], output_core_dims=[],
                                        dask='parallelize')
    return tropopause_gridded

def calc_LTS(vert_temp, surf_temp):
    '''Calculates the lower tropospheric stability, defined as the
    difference between the 700hPa potential temperature wrt 1000hPa and
    the surface skin temperature
    Inputs:
        Vert_temp (K): the vertical temperature profile'''
    LTS = vert_temp.sel({'pressure_level':700})*(10/7)**0.286 - surf_temp
    return LTS


def main():
    vert_temp = xr.open_dataset("/gws/nopw/j04/csgap/kkawaguchi/era5/ArcticCCF_temp.nc")
    surf_data = xr.open_dataset("/gws/nopw/j04/csgap/kkawaguchi/era5/polarCCF_surface.nc")
    
    surf_data['trp_temp'] = calc_tropo_gridded(vert_temp['t'])
    surf_data['LTS'] = calc_LTS(vert_temp, surf_data['skt'])
    surf_data.to_netcdf("/gws/nopw/j04/csgap/kkawaguchi/era5/polarCCF_singlelevel.nc")
    return None

main()