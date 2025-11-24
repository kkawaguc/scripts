import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.interpolate as interpolate
import scipy.stats as stats
import matplotlib.colors as colors
import datetime as dt
import cartopy.crs as ccrs
from KSR24_functions import *
import rioxarray

def _preprocess(dataset):
    return dataset['cldamt'].sel({'lat':slice(-72, 83)})

def isccp_data_retrieval():
    '''Retrieves the ISCCP data, collates the relevant variables and time/latitude bounds.'''
    dec_2010 = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/isccp/hgg_data/201012/*.nc', engine='netcdf4', preprocess=_preprocess, parallel=True)
    all_2011 = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/isccp/hgg_data/2011*/*.nc', engine='netcdf4', preprocess=_preprocess, parallel=True)
    all_2012 = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/isccp/hgg_data/2012*/*.nc', engine='netcdf4', preprocess=_preprocess, parallel=True)
    all_2013 = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/isccp/hgg_data/2013*/*.nc', engine='netcdf4', preprocess=_preprocess, parallel=True)
    all_2014 = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/isccp/hgg_data/2014*/*.nc', engine='netcdf4', preprocess=_preprocess, parallel=True)
    all_2015 = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/isccp/hgg_data/2015*/*.nc', engine='netcdf4', preprocess=_preprocess, parallel=True)
    all_2016 = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/isccp/hgg_data/2016*/*.nc', engine='netcdf4', preprocess=_preprocess, parallel=True)
    jan_2017 = xr.open_mfdataset('/gws/nopw/j04/csgap/kkawaguchi/isccp/hgg_data/201701/*.nc', engine='netcdf4', preprocess=_preprocess, parallel=True)
    cloud_data = xr.concat([dec_2010, all_2011, all_2012, all_2013, all_2014, all_2015, all_2016, jan_2017], dim='time')

    cloud_data = cloud_data['cldamt']
    cloud_data = cloud_data.pad({'lon':2}, 'wrap')
    cloud_data = cloud_data.assign_coords({'lon':np.linspace(-1.5, 361.5, 364)})

    cloud_data = xr.where(cloud_data <= 100, cloud_data, np.nan)
    cloud_data = xr.where(cloud_data >= 0, cloud_data, np.nan)

    cloud_data = cloud_data.rio.write_nodata(np.nan)
    cloud_data = cloud_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    cloud_data = cloud_data.rio.write_crs(ccrs.PlateCarree())
    cloud_data = cloud_data.rio.interpolate_na(method='linear')
    return cloud_data.compute()

def data_cleaning(data):
    '''Removes outlier values in incoming longwave, surface temperature, surface pressure,
    relative humidity and effective emissivity
    Inputs:
        data: the in-situ data we are using, pd dataframe
    Returns:
        data_new: data with outliers removed from each site'''
    data_new = pd.DataFrame()

    data["LAT"] = round(data["LAT"], 4)
    data["LONG"] = round(data["LONG"], 4)

    lat_vals = data["LAT"].unique()

    data["Eff_emissivity"] = data["LW_IN_F"]/(5.67e-8 * data["TA_F"]**4)

    check_cols = ["LW_IN_F", "TA_F", "PA_F", "Eff_emissivity", "RH"]

    lower = 0.02
    upper = 0.98

    for val in lat_vals:
        data_temp = data[data["LAT"] == val]
        low_bounds = []
        up_bounds = []
        for col in check_cols:
            lower_bound, upper_bound = data_temp[col].quantile([lower, upper])
            low_bounds.append(lower_bound)
            up_bounds.append(upper_bound)

        for i in range(len(check_cols)):
            data_temp = data_temp[(data_temp[check_cols[i]] >= low_bounds[i]) 
                                    & (data_temp[check_cols[i]] <= up_bounds[i])]

        if data_new.empty == True:
            data_new = data_temp
        else:
            data_new = pd.concat([data_new, data_temp], axis=0)
    return data_new

def main():
    #FLX_data = pd.read_csv('/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/FLX_data_11_16.csv')
    #BSRN_data = pd.read_csv('/gws/nopw/j04/csgap/kkawaguchi/KSR24_data/BSRN_data_11_16.csv')
    #FLX_data = FLX_data.drop(columns = ['Unnamed: 0', 'VPD_F', 'SAT_VAP'])
    #BSRN_data = BSRN_data.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'])
    
    #print(len(FLX_data))
    #print(len(BSRN_data))

    #data = pd.concat([FLX_data, BSRN_data], axis=0, ignore_index=True)
    
    #data = data.dropna(axis=0, how='any')

    #print(len(data))
    #data["TIMESTAMP_START"] = data["TIMESTAMP_START"].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    #data['LONG_360'] = data['LONG'].where(data['LONG'] > 0, data['LONG']+360)

    #data = data_cleaning(data)

    #data.to_csv('insitu_validation_data_cleaned.csv')

    #print(len(data))
    #cloud_data = isccp_data_retrieval()

    #print('cloud data loaded')

    #cf_vals = cloud_data.values
    #cf_time = cloud_data.time.to_numpy()
    #cf_lat = cloud_data.lat.to_numpy()
    #cf_lon = cloud_data.lon.to_numpy()

    #print(np.shape(cf_vals))
    #print(cf_vals[0:2, 0:2, 0:2])

    #data['CF_est'] = interpolate.interpn((cf_time, cf_lat, cf_lon), cf_vals, 
    #                                     (data["TIMESTAMP_START"], data["LAT"], data["LONG_360"]), bounds_error=False)
    
    #data['CF_est_frac'] = data['CF_est'] / 100

    #print('cloud fraction estimated')
    #We restrict the cloud fraction to between 0-1 and downwelling shortwave (used to switch between regimes for de Kok (2020)) does not exceed the clear-sky
    #downwelling shortwave

    #data.to_csv('insitu_validation_cfestimated.csv')    

    data = pd.read_csv('insitu_validation_cfestimated.csv') 

    data = data.to_xarray()

    data['DPT'] = vp2dpt(data['VP'])
    data['TCWV'] = tcwv_est(data['DPT'], lsm=1)

    print('starting DLR estimation')

    KSR24_results = KSR24(data['TA_F'], data['DPT'], data['PA_F'], data['CF_est_frac'], 1)
    DO98UM75_results = DO98_UM75(data['TA_F'], data['CF_est_frac'], data['TCWV'])
    C14_results = C14(data['TA_F'], data['RH'], data['CF_est_frac'])
    CN14_results = CN14(data['TA_F'], data['VP'], data['CF_est_frac'])
    deK20_results = deK20(data['TA_F'], data['SW_CS'], data['RH'])
    SR21BE23_results = SR21_BE23(data['TA_F'], data['DPT'], data['TCWV'], data['PA_F'], data['RH'])
    B32BE23_results = B32_BE23(data['TA_F'], data['DPT'], data['RH'])

    print(KSR24_results.head())
    print(DO98UM75_results.head())
    print(C14_results.head())
    print(CN14_results.head())
    print(deK20_results.head())

    results = [KSR24_results, DO98UM75_results, C14_results, CN14_results, deK20_results, SR21BE23_results, B32BE23_results]

    print('starting plotting routine')

    plot_histogram(results, data['LW_IN_F'], plot_name='supplementary_figure_9.png', globe=False)

    plot_main3(results, data['LW_IN_F'], data['TA_F'], data['PA_F'], plot_name='main_3_b_d_f.png', globe=False)
    return None

main()
