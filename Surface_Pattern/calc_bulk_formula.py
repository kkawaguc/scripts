# %%

import xarray as xr
from CMIP6_analysis_functions import *

monthly_data = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/amip_piForcing_new_monthly.nc', 
                               chunks={'time':-1, 'lat':24, 'lon':24, 'model':1}, 
                               drop_variables=['ta', 'hus', 'rsut', 'rlut', 'rsds', 'rsdscs', 'rlds',
                                               'rldscs', 'rsutcs', 'rlutcs', 'rlus', 'rsus', 'hfss', 'hfls',
                                               'rsuscs'])
monthly_data = monthly_data.drop('plev')

# %%

def calc_anomaly(x):
    return (x - x.mean(dim='time'))

def tile_data(to_tile, new_shape):
    """Tile dataset along time axis to match another dataset."""
    # new_shape = _check_time(new_shape)
    if len(new_shape.time) % 12 != 0:
        raise ValueError("dataset time dimension must be divisible by 12")
    tiled = xr.concat(
        [to_tile for i in range(int(len(new_shape.time) / 12))], dim="time"
    )
    tiled["time"] = new_shape.time
    return tiled

climatology = monthly_data.groupby('time.month').mean('time').rename({'month':'time'})
climatology = tile_data(climatology, monthly_data)
# %%

dHF = calc_dHF(monthly_data, climatology, climo_variables=[])
dHF.to_netcdf('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/dHF.nc')

dHF_temp = calc_dHF(monthly_data, climatology, climo_variables=['uas', 'vas'])
dHF_temp.to_netcdf('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/dHF_temp.nc')

dHF_wind = calc_dHF(monthly_data, climatology, climo_variables=['tas', 'ts', 'huss'])
dHF_wind.to_netcdf('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/dHF_wind.nc')

