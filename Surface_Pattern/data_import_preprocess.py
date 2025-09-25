import xarray as xr
import os
import numpy as np
import cftime
import scipy as sp

models = {"HadGEM3-GC31-LL": ["MOHC", "r1i1p1f*", "gn"], "CESM2":["NCAR", "r1i1p1f1", "gn"], "CanESM5":["CCCma", "r1i1p2f1", "gn"],
          "MRI-ESM2-0":["MRI", "r1i1p1f1", "gn"]}
experiments = {"piControl":"CMIP","amip-piForcing":"CFMIP", "historical":"CMIP"}
variables = ["rsut", "rlut", "rsds", "rlds", "rlus", "rsus", "hfss", "hfls", "tas", "huss", "ts", "ps", "uas", "vas"]

def _preprocess(dataset):
    dataset = dataset.resample({"time":"YS"}).mean()
    long = dataset.lon
    dlon = np.diff(long).mean()  # average spacing

    # Pad longitude with wrap, then adjust coordinates to avoid duplication
    dataset = dataset.pad(lon=2, how="wrap")
    new_long = np.concatenate([
        [long[0] - 2 * dlon, long[0] - dlon],
        long,
        [long[-1] + dlon, long[-1] + 2 * dlon]
    ])
    dataset = dataset.assign_coords({"lon":new_long})

    interp_lat = np.linspace(-88.75, 88.75, 72)
    interp_lon = np.linspace(1.25, 358.75, 144)
    dataset = dataset.interp({"lat":interp_lat, "lon":interp_lon})
    for bounds_var in ['lat_bnds', 'lon_bnds']:
        if bounds_var in dataset:
            dataset = dataset.drop_vars(bounds_var)
    return dataset

plev = np.array([100000.,  92500.,  85000.,  70000.,  60000.,  50000.,  40000.,  30000.,
                 25000.,  20000.,  15000.,  10000.,   7000.,   5000.,   3000.,   2000.,
                 1000.,    500.,    100.])

print(plev)

def SP_interp(x, evalpt):
    logp = np.log(plev)
    y = x[~np.isnan(x)]
    if len(y) == 0:
        return np.nan
    logp = logp[~np.isnan(x)]
    lin_interp = sp.interpolate.interp1d(logp, y, axis=0, kind="linear", fill_value="extrapolate")
    res = lin_interp(np.log(evalpt))
    return res

base_dir = "/badc/cmip6/data/CMIP6/"

for exp in experiments:
    print(exp)
    for mod in models:
        print(mod)
        for var in variables:
            print(var)
            file_path = os.path.join(base_dir, experiments[exp], models[mod][0], mod, exp, models[mod][1], "Amon", var, models[mod][2], "latest/*.nc")
            if var == "uas":
                print(os.path.isdir(os.path.join(base_dir, experiments[exp], models[mod][0], mod, exp, models[mod][1], "Amon", var)))
                if os.path.isdir(os.path.join(base_dir, experiments[exp], models[mod][0], mod, exp, models[mod][1], "Amon", var)):
                    temp_ds = xr.open_mfdataset(file_path, engine="netcdf4", parallel=True, preprocess=_preprocess, use_cftime=True, decode_times=True)
                elif mod == "HadGEM3-GC31-LL":
                    temp_ds = xr.open_mfdataset(file_path, engine="netcdf4", parallel=True, preprocess=_preprocess, use_cftime=True, decode_times=True)
                else:
                    vert_wind = xr.open_mfdataset(os.path.join(base_dir, experiments[exp], models[mod][0], mod, exp, models[mod][1], "Amon", "ua", models[mod][2], "latest/*.nc"), 
                                            engine="netcdf4", parallel=True, preprocess=_preprocess, use_cftime=True, decode_times=True)
                    surf_pres = xr.open_mfdataset(os.path.join(base_dir, experiments[exp], models[mod][0], mod, exp, models[mod][1], "Amon", "ps", models[mod][2], "latest/*.nc"), 
                                            engine="netcdf4", parallel=True, preprocess=_preprocess, use_cftime=True, decode_times=True)
                    temp_da = xr.apply_ufunc(SP_interp, vert_wind["ua"].chunk({"plev":-1}), surf_pres["ps"], input_core_dims = [["plev"], []],output_core_dims=[[]],
                                             vectorize=True, dask="parallelized", output_dtypes=[float])
                    temp_ds = temp_da.to_dataset(name="uas")
            elif var == "vas":
                if os.path.isdir(str(os.path.join(base_dir, experiments[exp], models[mod][0], mod, exp, models[mod][1], "Amon", var, models[mod][2]))):
                    temp_ds = xr.open_mfdataset(file_path, engine="netcdf4", parallel=True, preprocess=_preprocess, use_cftime=True, decode_times=True)
                elif mod == "HadGEM3-GC31-LL":
                    temp_ds = xr.open_mfdataset(file_path, engine="netcdf4", parallel=True, preprocess=_preprocess, use_cftime=True, decode_times=True)
                else:
                    vert_wind = xr.open_mfdataset(os.path.join(base_dir, experiments[exp], models[mod][0], mod, exp, models[mod][1], "Amon", "va", models[mod][2], "latest/*.nc"), 
                                            engine="netcdf4", parallel=True, preprocess=_preprocess, use_cftime=True, decode_times=True)
                    surf_pres = xr.open_mfdataset(os.path.join(base_dir, experiments[exp], models[mod][0], mod, exp, models[mod][1], "Amon", "ps", models[mod][2], "latest/*.nc"), 
                                            engine="netcdf4", parallel=True, preprocess=_preprocess, use_cftime=True, decode_times=True)
                    temp_da = xr.apply_ufunc(SP_interp, vert_wind["va"].chunk({"plev":-1}), surf_pres["ps"], input_core_dims = [["plev"], []],output_core_dims=[[]],
                                             vectorize=True, dask="parallelized", output_dtypes=[float])
                    temp_ds = temp_da.to_dataset(name="vas")
            elif exp == "piControl" and mod == "CanESM5" and var in ["rsus", "rlus"]:
                temp_ds = xr.open_mfdataset("/gws/nopw/j04/csgap/kkawaguchi/CanESM5_picontrol/"+var+"*.nc", engine="netcdf4", parallel=True, 
                                            preprocess = _preprocess, use_cftime=True, decode_times=True)
            else:        
                temp_ds = xr.open_mfdataset(file_path, engine="netcdf4", parallel=True, preprocess=_preprocess, use_cftime=True, decode_times=True)
            if len(temp_ds.time) > 200:
                temp_ds = temp_ds.isel({"time":slice(None, 200)})
            if exp == "piControl":
                temp_ds["time"] = xr.date_range(start="1850", periods=200, freq="YS", calendar="360_day", use_cftime=True)
            if exp == "amip-piForcing":
                if mod == "CESM2":
                    temp_ds = temp_ds.isel({"time":slice(None, 145)})
                temp_ds["time"] = xr.date_range(start="1870", periods=145, freq="YS", calendar="360_day", use_cftime=True)
            if exp == "historical":
                temp_ds["time"] = xr.date_range(start="1850", periods=165, freq="YS", calendar="360_day", use_cftime=True)
            if var == "rsut":
                mod_ds = temp_ds
            else:
                mod_ds = xr.merge([mod_ds, temp_ds], compat="override")
        mod_ds = mod_ds.assign_coords({"model":mod})
        if mod == "HadGEM3-GC31-LL":
            exp_ds = mod_ds
        else:
            exp_ds = xr.concat([exp_ds, mod_ds], dim="model",coords="minimal" ,compat="override", join="outer")
    exp_ds.to_netcdf(f"/gws/nopw/j04/csgap/kkawaguchi/surface_data/{exp}.nc")