import xarray as xr
import scipy as sp
import numpy as np

def calc_tropo_temp(ta):
    plev = ta.pressure_level
    plevK = plev**(0.286)

    T_interp = sp.interpolate.CubicSpline(plevK[::-1], ta.isel{"pressure_level":slice(None, None, -1)})
    dTdpK = T_interp.derivative(1)





vert_temp = xr.open_dataset("ArcticCCF_temp.nc")
surf_data = xr.open_dataset("polarCCF_surface.nc")
plev_data = xr.open_dataset("polarCCF_plev.nc")

surf_data = surf_data.sel({"lat":slice(30, None)})
plev_data = plev_data.sel({"lat":slice(30, None)})

ta = vert_temp["t"]
theta_700 = ta.sel({"pressure_level":700})*(10/7)**(0.286)

LTS = theta_700 - surf_data["skt"]