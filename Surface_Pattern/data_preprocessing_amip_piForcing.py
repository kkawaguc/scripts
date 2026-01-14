import xarray as xr
import numpy as np
import xesmf as xe
#import xcdat as xc
import os
import metpy.calc as mpcalc
import metpy.units as units
import scipy as sp
from CMIP6_analysis_functions import *

#experiments = {"amip-piForcing":"CFMIP"}
variables =  ["rsut", "rlut", "rsds", "rlds", "rlus", "rsus", "hfss", "hfls", "tas", "huss", "ts", "ps", "ua", "va", 
              'rsutcs', 'rlutcs', 'rsuscs', 'rsdscs', 'rldscs']

def regrid_data(model_data):
    '''Regrid model data conservatively to a common 3 degree grid
    Inputs:
        model_data: the model data to be regridded
    Outputs:
        regridded_data: data regridded conservatively at 3 degrees'''
    output_grid = xe.util.grid_global(3, 3, lon1=360)
    # NEED TO ADD LON BOUNDS HERE
    if 'lon' not in model_data.dims:
        model_data = model_data.rename({'longitude':'lon', 'latitude':'lat'})
    model_lon = model_data.lon
    model_lat = model_data.lat
    dlon = np.diff(model_lon, 1)[0]
    print(dlon)
    model_data = model_data.assign_coords({'lon_b':np.linspace(float(model_lon.min()) - dlon/2, 
                                                 float(model_lon.max()) + dlon/2, 
                                                 len(model_lon)+1),
                                           'lat_b':np.linspace(-90,90,len(model_lat)+1)})
    regridder_conservative = xe.Regridder(model_data, output_grid, 'conservative', periodic=True)
    return regridder_conservative(model_data)

def annual_mean_and_regrid(dataset):
    '''Compute the annual mean data'''
    annual_mean = dataset.resample({'time':'YS'}).mean()
    annual_mean_regridded = regrid_data(annual_mean)
    return annual_mean_regridded

def sel_lowest_pressure_above_ground(surf_pres_plus_10):
    plev = np.linspace(100000., 40000., 61)
    above_ground = np.where(plev < 100*surf_pres_plus_10, False, True) 
    return plev[np.argmin(above_ground)]

def add_boundary_knots(spline):
    """
    Add knots infinitesimally to the left and right.

    Additional intervals are added to have zero 2nd and 3rd derivatives,
    and to maintain the first derivative from whatever boundary condition
    was selected. The spline is modified in place.
    """
    # determine the slope at the left edge
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)

    # add a new breakpoint just to the left and use the
    # known slope to construct the PPoly coefficients.
    leftxnext = np.nextafter(leftx, leftx - 1)
    leftynext = lefty + leftslope*(leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    # repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    rightslope = spline(rightx,nu=1)
    rightxnext = np.nextafter(rightx, rightx + 1)
    rightynext = righty + rightslope * (rightxnext - rightx)
    rightcoeffs = np.array([0, 0, rightslope, rightynext])
    spline.extend(rightcoeffs[..., None], np.r_[rightxnext])


def wind_lowest_pressure_above_ground(ua, surf_pres):
    sel_plev = sel_lowest_pressure_above_ground(surf_pres)
    plev_init = np.array([100000.,  92500.,  85000.,  70000.,  60000.,  50000.,  40000.,  30000.,
                          25000.,  20000.,  15000.,  10000.,   7000.,   5000.,   3000.,   2000.,
                          1000.,    500.,    100.])
    plev_mask = plev_init[~np.isnan(ua)]
    ua_mask = ua[~np.isnan(ua)]

    sort_idx = np.argsort(plev_mask)
    plev_sorted = plev_mask[sort_idx]
    ua_sorted = ua_mask[sort_idx]
    
    ua_upscale = sp.interpolate.CubicSpline(plev_sorted, ua_sorted, bc_type='clamped')
    add_boundary_knots(ua_upscale)
    return ua_upscale(sel_plev)


def calculate_u10(ua, surf_pres):
    '''Calculate the 10m wind fields by using a logarithmic profile.
    Use a roughness length of 0.0002 if mean surface pressure is 
    greater than 980hPa, 0.003 otherwise.'''
    surf_pressure_climatology = surf_pres.mean('time')
    z_0 = xr.where(surf_pressure_climatology > 98000, 0.0002, 0.003) * units.m

    z_surf = mpcalc.pressure_to_height_std(surf_pres * units.Pa)
    P_10 = mpcalc.add_height_to_pressure(surf_pres * units.Pa, 15 * units.m)
    lowest_pressure_level = xr.apply_ufunc(sel_lowest_pressure_above_ground, P_10.metpy.dequantify(), input_core_dims=None, vectorize=True,dask='parallelized')
    lowest_pressure_level = lowest_pressure_level * units.Pa
    z_lpl = mpcalc.pressure_to_height_std(lowest_pressure_level)
    z_bottom = z_lpl - z_surf
    u_bottom = xr.apply_ufunc(wind_lowest_pressure_above_ground, ua, surf_pres ,input_core_dims=[['plev'], []], output_core_dims=[[]], output_dtypes=[float],vectorize=True,dask='parallelized')
    u10 = u_bottom * np.log(10 * units.m/z_0) / np.log(z_bottom/z_0)
    return u10.metpy.dequantify()

def compute_and_regrid_model_data(mod_name, mod_attrs):
    '''Code block to compute and regrid model data'''
    base_dir = "/badc/cmip6/data/CMIP6/CFMIP"
    
    var_data_list = []

    for var in variables:
        path_to_file = os.path.join(base_dir, mod_attrs[0], mod_name, 'amip-piForcing', mod_attrs[1], "Amon", var, "g*/latest/*.nc")
        check_path = os.path.join(base_dir, mod_attrs[0], mod_name, 'amip-piForcing', mod_attrs[1], "Amon", var)
        print(path_to_file)
        if os.path.exists(check_path):
            var_data = xr.open_mfdataset(path_to_file, preprocess = annual_mean_and_regrid)
        if not os.path.exists(check_path):
            if var == 'ps':
                #Some of the ps data is in csgap
                var_data = xr.open_mfdataset('/gws/ssde/j25a/csgap/kkawaguchi/amip_piForcing/*'+mod_name+'*.nc', preprocess = annual_mean_and_regrid)
            elif var in ['uas', 'vas']:
                try:
                    path_to_ps_file = os.path.join(base_dir, mod_attrs[0], mod_name, 'amip-piForcing', mod_attrs[1], "Amon", 'ps', "g*/latest/*.nc")
                    ps = xr.open_mfdataset(path_to_ps_file, preprocess = annual_mean_and_regrid)
                except:
                    path_to_ps_file = '/gws/ssde/j25a/csgap/kkawaguchi/amip_piForcing/*'+mod_name+'*.nc'
                    ps = xr.open_mfdataset(path_to_ps_file, preprocess = annual_mean_and_regrid)
                if var == 'uas':
                    path_to_ua_file = os.path.join(base_dir, mod_attrs[0], mod_name, 'amip-piForcing', mod_attrs[1], "Amon", 'ua', "g*/latest/*.nc")
                    u_winds = xr.open_mfdataset(path_to_ua_file, preprocess = annual_mean_and_regrid)
                    var_data = calculate_u10(u_winds['ua'], ps['ps']).rename('uas')
                elif var == 'vas':
                    path_to_va_file = os.path.join(base_dir, mod_attrs[0], mod_name, 'amip-piForcing', mod_attrs[1], "Amon", 'va', "g*/latest/*.nc")
                    v_winds = xr.open_mfdataset(path_to_va_file, preprocess = annual_mean_and_regrid)
                    var_data = calculate_u10(v_winds['va'], ps['ps']).rename('vas')
            elif var == 'rldscs' and mod_name == 'TaiESM1':
                tas = xr.open_mfdataset(os.path.join(base_dir, mod_attrs[0], mod_name, 'amip-piForcing', mod_attrs[1], "Amon/tas/g*/latest/*.nc"),
                                        preprocess=annual_mean_and_regrid)
                tas = tas['tas']
                huss = xr.open_mfdataset(os.path.join(base_dir, mod_attrs[0], mod_name, 'amip-piForcing', mod_attrs[1], "Amon/huss/g*/latest/*.nc"),
                                        preprocess=annual_mean_and_regrid)
                huss = huss['huss']
                ps = xr.open_mfdataset('/gws/ssde/j25a/csgap/kkawaguchi/amip_piForcing/*'+mod_name+'*.nc', preprocess = annual_mean_and_regrid)
                ps = ps['ps']
                dpt = q2dpt(huss, ps)
                lsm = xr.open_mfdataset("/gws/ssde/j25a/csgap/kkawaguchi/KSR24_data/data_stream-moda_stepType-avgua.nc", engine="netcdf4")
                lsm = regrid_data(lsm)
                lsm = lsm['lsm']
                tcwv = tcwv_est(dpt, lsm)
                var_data = SR21_inversion(tas, huss, tcwv, ps, ppm=280).rename('rldscs')
            else:
                print('error occurred with ' + var +' '+mod_name)
        var_data_list.append(var_data)
        mod_data = xr.merge(var_data_list, compat='minimal')
        if mod_name == 'CESM2':
            mod_data = mod_data.isel({'time':slice(None, -1)})
            mod_data['time'] = xr.date_range(start="1870", periods=145, freq="YS", use_cftime=True)
        elif mod_name == 'TaiESM1':
            mod_data = mod_data.isel({'time':slice(20, None)})
            mod_data['time'] = xr.date_range(start="1870", periods=145, freq="YS", use_cftime=True)
        else:
            mod_data['time'] = xr.date_range(start="1870", periods=145, freq="YS", use_cftime=True)
    return mod_data

def main():
    '''Preprocessing script for amip-piForcing to analyze the SST pattern effect
    on surface heat fluxes and circulation.'''
    models = {"CanESM5":["CCCma", "r1i1p2f1"], "CESM2":["NCAR", "r1i1p1f1"], 'CNRM-CM6-1':['CNRM-CERFACS', 'r1i1p1f2'],
              #'GISS-E2-1-G':['NASA-GISS', 'r1i1p1f1'], 
              "HadGEM3-GC31-LL": ["MOHC", "r1i1p1f3"], 'IPSL-CM6A-LR':['IPSL', 'r1i1p1f1'], 
              'MIROC6':['MIROC', 'r1i1p1f1'], 
              "MRI-ESM2-0":["MRI", "r1i1p1f1"],'TaiESM1':['AS-RCEC', 'r1i1p1f1']}
    
    output_list = []
    model_list = []

    for mod in models:
        temp_data = compute_and_regrid_model_data(mod, models[mod])
        output_list.append(temp_data)
        model_list.append(mod)

    output_data = xr.concat(output_list, dim='model', coords='minimal', compat='override')
    output_data.assign_coords({'model':model_list})

    output_data.to_netcdf("/gws/ssde/j25a/csgap/kkawaguchi/surface_data/amip_piForcing_new.nc")    
    return None

def main_test_vars():
    models = {"CanESM5":["CCCma", "r1i1p2f1"], "CESM2":["NCAR", "r1i1p1f1"], 'CNRM-CM6-1':['CNRM-CERFACS', 'r1i1p1f2'],
              #'GISS-E2-1-G':['NASA-GISS', 'r1i1p1f1'], 
              "HadGEM3-GC31-LL": ["MOHC", "r1i1p1f3"], 'IPSL-CM6A-LR':['IPSL', 'r1i1p1f1'], 
              #'MIROC6':['MIROC', 'r1i1p1f1'], 
              "MRI-ESM2-0":["MRI", "r1i1p1f1"],'TaiESM1':['AS-RCEC', 'r1i1p1f1']}

    for mod in models:
        temp_data = compute_and_regrid_model_data(mod, models[mod])
    return None

main()