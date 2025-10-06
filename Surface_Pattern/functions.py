import xskillscore as xs
import xarray as xr
import numpy as np
import global_land_mask as lm
import easyclimate.core.utility as utility
import easyclimate.field.boundary_layer.aerobulk as aerobulk
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

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
    