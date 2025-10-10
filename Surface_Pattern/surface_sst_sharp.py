import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# This notebook calculates fluctuations in the surface energy budget associated with changes in the equatorial Pacific zonal SST gradient""")
    return


@app.cell
def importpackages():
    import marimo as mo
    import xarray as xr
    import numpy as np
    import nc_time_axis
    import xskillscore as xs
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from functions import calc_land_mask, calc_fdbk, calc_dHF, facetplot
    import pandas as pd
    import xrscipy
    return (
        calc_dHF,
        calc_fdbk,
        calc_land_mask,
        ccrs,
        facetplot,
        mo,
        np,
        pd,
        plt,
        xr,
        xs,
    )


@app.cell
def _(calc_land_mask, np, xr):
    #Replace here with amip-piForcing or historical simulations
    base_data = xr.open_dataset("/gws/nopw/j04/csgap/kkawaguchi/surface_data/amip-piForcing.nc", chunks={'lat':36, 'lon':36})

    base_data = base_data.rolling(dim={'time':11}, center=True).mean().dropna(dim='time', how='all').compute()

    base_data['TOA'] = -(base_data['rlut'] + base_data['rsut'])
    base_data['TOA_cs'] = -(base_data['rlutcs'] + base_data['rsutcs'])
    base_data['SFC'] = base_data['rsds'] + base_data['rlds'] - (base_data['rlus'] + base_data['rsus'] + 
                                                                            base_data['hfss'] + base_data['hfls'])
    base_data['SFCrad'] = base_data['rsds'] + base_data['rlds'] - (base_data['rlus'] + base_data['rsus'])
    base_data['SFCrad_cs'] = base_data['rsds'] + base_data['rlds'] - (base_data['rlus'] + base_data['rsus'])

    base_data['TOA_SWCRE'] = base_data['rsutcs'] - base_data['rsut']
    base_data['TOA_LWCRE'] = base_data['rlutcs'] - base_data['rlut']

    base_data['SFC_SWCRE'] = base_data['rsds'] - base_data['rsdscs']
    base_data['SFC_LWCRE'] = base_data['rlds'] - base_data['rldscs']

    #Regrid to calculate the land mask
    land_mask = calc_land_mask(base_data)

    #Calculate data over just the ocean
    ocean_flux = base_data[["rsds", "rlds", "rsus", "rlus", "hfss", "hfls","uas", "vas", "ps", "tas", "ts", "huss"]]
    ocean_flux = xr.where(land_mask == True, ocean_flux, np.nan)
    ocean_flux_climo = ocean_flux.mean("time")

    ocean_flux, ocean_flux_climo = xr.broadcast(ocean_flux, ocean_flux_climo)
    return base_data, ocean_flux, ocean_flux_climo


@app.cell
def _(base_data):
    base_data
    return


@app.cell
def _(T_series, base_data, calc_fdbk):
    TOA = -(base_data['rlut'] + base_data['rsut'])
    TOA_fdbk = calc_fdbk(TOA, T_series)
    return TOA, TOA_fdbk


@app.cell
def _(base_data, sst_idx, xs):
    flux_response = xs.linslope(sst_idx, base_data)

    return (flux_response,)


@app.cell
def _(ccrs, flux_response, plt):
    ## Plot the TOA, Surface and Atmospheric Energy budgets
    fig1, ax1 = plt.subplots(1, 3, figsize=(12, 3), layout='constrained', subplot_kw={'projection':ccrs.Robinson(central_longitude=180)})

    flux_response['TOA'].plot(ax=ax1[0], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, cmap='RdBu_r',extend='both', add_colorbar=False)
    flux_response['SFC'].plot(ax=ax1[1], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, cmap='RdBu_r',extend='both', add_colorbar=False)
    (flux_response['SFC'] - flux_response['TOA']).plot(ax=ax1[2], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, 
                                                       cmap='RdBu_r', extend='both', 
                                                       cbar_kwargs={'shrink':0.8, 'label': '$\\mathrm{W m^{-2}/\\sigma}$'})

    ax1[0].set_title('TOA')
    ax1[1].set_title('SFC')
    ax1[2].set_title('ATM Heat Convergence')

    for i in range(3):
        ax1[i].coastlines()

    fig1.suptitle('Energy Budget Response to SST#', fontsize=18)

    plt.show()
    return


@app.cell
def _(ccrs, flux_response, plt):
    ## Plot the TOA Energy budget and decomposition
    fig2, ax2 = plt.subplots(2, 2, figsize=(8, 5), layout='constrained', subplot_kw={'projection':ccrs.Robinson(central_longitude=180)})

    cbar2 = flux_response['TOA'].plot(ax=ax2[0,0], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, cmap='RdBu_r',extend='both', add_colorbar=False)
    flux_response['TOA_cs'].plot(ax=ax2[0,1], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, 
                                 cmap='RdBu_r',extend='both', add_colorbar=False)
    flux_response['TOA_SWCRE'].plot(ax=ax2[1,0], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, 
                                    cmap='RdBu_r', extend='both', add_colorbar=False)
    flux_response['TOA_LWCRE'].plot(ax=ax2[1,1], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, 
                                    cmap='RdBu_r', extend='both', add_colorbar=False)

    fig2.colorbar(ax = ax2[:], mappable=cbar2, 
                  orientation='vertical', shrink=0.8, label='$\\mathrm{W m^{-2}/\\sigma}$')
    ax2[0,0].set_title('Net')
    ax2[0,1].set_title('Clear-sky')
    ax2[1,0].set_title('SWCRE')
    ax2[1,1].set_title('LWCRE')

    for j in range(4):
        ax2[j//2, j%2].coastlines()

    fig2.suptitle('TOA Response to SST#', fontsize=18)

    plt.show()
    return


@app.cell
def _(ccrs, flux_response, plt):
    ## Plot the SFC Energy budget and decomposition
    fig3, ax3 = plt.subplots(2, 3, figsize=(12, 5), layout='constrained', subplot_kw={'projection':ccrs.Robinson(central_longitude=180)})

    cbar3 = flux_response['SFC'].plot(ax=ax3[0,0], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, cmap='RdBu_r',extend='both', add_colorbar=False)
    flux_response['SFCrad_cs'].plot(ax=ax3[0,1], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, 
                                 cmap='RdBu_r',extend='both', add_colorbar=False)
    flux_response['SFC_SWCRE'].plot(ax=ax3[1,0], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, 
                                    cmap='RdBu_r', extend='both', add_colorbar=False)
    flux_response['SFC_LWCRE'].plot(ax=ax3[1,1], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, 
                                    cmap='RdBu_r', extend='both', add_colorbar=False)
    (-flux_response['hfss']).plot(ax=ax3[0,2], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, 
                                    cmap='RdBu_r', extend='both', add_colorbar=False)
    (-flux_response['hfls']).plot(ax=ax3[1,2], transform=ccrs.PlateCarree(), vmin=-5, vmax=5, 
                                    cmap='RdBu_r', extend='both', add_colorbar=False)

    fig3.colorbar(ax = ax3[:], mappable=cbar3, 
                  orientation='vertical', shrink=0.8, label='$\\mathrm{W m^{-2}/\\sigma}$')
    ax3[0,0].set_title('Net')
    ax3[0,1].set_title('Clear-sky Radiation')
    ax3[1,0].set_title('SWCRE')
    ax3[1,1].set_title('LWCRE')
    ax3[0,2].set_title('Sensible Heat')
    ax3[1,2].set_title('Latent Heat')

    for k in range(6):
        ax3[k%2, k//2].coastlines()

    fig3.suptitle('Surface Response to SST#', fontsize=18)

    plt.show()
    return


@app.cell
def _(ccrs, flux_response, plt):
    ## Plot the TOA, Surface and Atmospheric Energy budgets
    fig4, ax4 = plt.subplots(1, 2, figsize=(8, 3), layout='constrained', subplot_kw={'projection':ccrs.Robinson(central_longitude=180)})

    (flux_response['TOA_SWCRE'] + flux_response['TOA_LWCRE']).plot(ax=ax4[0], transform=ccrs.PlateCarree(), 
                                                                   vmin=-5, vmax=5, cmap='RdBu_r',extend='both', add_colorbar=False)
    (flux_response['SFC_SWCRE'] + flux_response['SFC_LWCRE']).plot(ax=ax4[1], transform=ccrs.PlateCarree(), 
                                                                   vmin=-5, vmax=5, cmap='RdBu_r',extend='both',
                                                                   cbar_kwargs={'shrink':0.8, 'label': '$\\mathrm{W m^{-2}/\\sigma}$'})

    ax4[0].set_title('TOA')
    ax4[1].set_title('SFC')

    for l in range(2):
        ax4[l].coastlines()

    fig4.suptitle('CRE response to SST#', fontsize=18)

    plt.show()
    return


@app.cell
def _(TOA_fdbk, plt, weights):
    TOA_fdbk.weighted(weights).mean(('lat', 'lon')).plot.line(x='time')
    plt.ylim([-3.5, 0.5])
    plt.title('TOA Fdbk ($\\mathrm{W/m^2/K}$)')
    plt.show()
    return


@app.cell
def _(TOA, facetplot, sfc, sst_idx, xr, xs):
    energy_budget = xr.concat([TOA, sfc, TOA-sfc], dim='z')
    energy_budget = energy_budget.assign_coords({'z':['TOA', 'sfc', 'atm']})
    facetplot(xs.linslope(sst_idx, energy_budget, dim='time'), 'model', 'z', 
              title='Energy Response to SST# ($\\mathrm{W m^{-2}/\\sigma}$)', figsize=(8, 2.5), vmin=-5, vmax=5, cmap='RdBu_r', cbar_kwargs={'shrink':0.6})
    return


@app.cell
def _(T_series, base_data, calc_fdbk):
    sfc = (base_data['rsds'] + base_data['rlds'] - base_data['rlus'] - base_data['rsus'] - base_data['hfss'] - base_data['hfls'])
    sfc_fdbk = calc_fdbk(sfc, T_series)
    return sfc, sfc_fdbk


@app.cell
def _(plt, sfc_fdbk, weights):
    sfc_fdbk.weighted(weights).mean(('lat', 'lon')).plot.line(x='time')
    plt.title('Surface Fdbk ($\\mathrm{W/m^2/K}$)')
    plt.ylim([-3.5, 0.5])
    plt.show()
    return


@app.cell
def _(base_data, np, ocean_flux):
    #Calculate surface heat anomalies over the ocean
    ocean_anom_data = ocean_flux - ocean_flux.mean("time")
    ocean_anom_data["sfc_rad"] = ocean_anom_data["rlds"] + ocean_anom_data["rsds"] - ocean_anom_data["rlus"] - ocean_anom_data["rsus"]
    ocean_anom_data["sfc"] = ocean_anom_data["sfc_rad"] - ocean_anom_data["hfss"] - ocean_anom_data["hfls"]

    #Calculate the timeseries we use as the basis for our study
    weights = np.cos(np.radians(base_data.lat))

    T_series = (base_data["tas"].weighted(weights).mean(("lat", "lon")) - base_data["tas"].weighted(weights).mean(("lat", "lon", "time"))).rename('tas')
    SST_series = (ocean_flux["ts"] - ocean_flux["ts"].mean("time")).rename('ts')
    return T_series, ocean_anom_data, weights


@app.cell
def _(np, ocean_flux, weights, xr):
    #Calculate the SST indices (Watanabe and Fueglistaler)
    Watanabe_idx_raw = (ocean_flux['ts'].sel({"lat":slice(-5, 5), "lon":slice(180,270)}).weighted(weights).mean(("lat", "lon")) - ocean_flux['ts'].sel({"lat":slice(-5, 5), "lon":slice(110,180)}).weighted(weights).mean(("lat", "lon")))

    tropical_SST = ocean_flux['ts'].sel({'lat':slice(-30, 30)}).compute()

    Fueglistaler_threshold = tropical_SST.weighted(weights).quantile(0.7, dim=('lat', 'lon'))

    Fueglistaler_idx_raw = xr.where(tropical_SST > Fueglistaler_threshold, tropical_SST, np.nan).weighted(weights).mean(('lat', 'lon')) - tropical_SST.weighted(weights).mean(('lat', 'lon'))

    Fueglistaler_idx_norm = (Fueglistaler_idx_raw-Fueglistaler_idx_raw.mean('time'))/Fueglistaler_idx_raw.std(dim='time')
    return Fueglistaler_idx_norm, Fueglistaler_idx_raw, Watanabe_idx_raw


@app.cell
def _(Fueglistaler_idx_norm, Fueglistaler_idx_raw, Watanabe_idx_raw, mo):
    idx_choice = mo.ui.dropdown(options = ['Equatorial Pacific: Watanabe et al. (2021)',
                                          'SST#: Fueglistaler (2019)', 'Norm SST#'], 
                                value = 'Norm SST#')

    idx_options = {'Equatorial Pacific: Watanabe et al. (2021)': Watanabe_idx_raw, 'SST#: Fueglistaler (2019)': Fueglistaler_idx_raw,
                   'Norm SST#': Fueglistaler_idx_norm}

    idx_choice
    return idx_choice, idx_options


@app.cell
def _(T_series, sst_idx, xs):
    xs.linslope(sst_idx, T_series)
    return


@app.cell
def _(idx_choice, idx_options):
    sst_idx = idx_options[idx_choice.value]
    return (sst_idx,)


@app.cell
def _(ocean_anom_data, sst_idx, xs):
    ocean_slope = xs.linslope(sst_idx, ocean_anom_data, dim='time').drop_vars(['quantile', 'height']).compute()
    return (ocean_slope,)


@app.cell
def _(ccrs, ocean_slope, plt):
    fig5, ax5 = plt.subplots(1, 2, figsize=(8, 3),layout='constrained', subplot_kw={'projection':ccrs.Robinson(central_longitude=180)})

    ocean_slope['ts'].mean('model').plot(ax=ax5[0], vmin=-0.3, vmax=0.3, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree(), add_colorbar=False)
    (ocean_slope['ts']-0.06646294).mean('model').plot(ax=ax5[1], vmin=-0.3, vmax=0.3, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree(), cbar_kwargs={'shrink':0.8})

    ax5[0].coastlines()
    ax5[1].coastlines()

    ax5[0].set_title('Default')
    ax5[1].set_title('GW removed')

    fig5.suptitle('Warming Pattern regressed onto SST index ($\mathrm{K/\\sigma}$)')
    return


@app.cell
def _(facetplot, ocean_slope):
    facetplot(ocean_slope['ts'], 'model', None, title='Warming pattern regressed onto SST index ($\\mathrm{K / \\sigma}$)', figsize=(6, 3), robust=True)
    return


@app.cell
def _(facetplot, ocean_slope):
    facetplot(-ocean_slope['hfss'], 'model', None, title='Sensible Heat',vmin=-4, vmax=4, extend='both',cmap='RdBu_r', figsize=(6,3))
    return


@app.cell
def _(ocean_slope, weights, xr):
    weights_broadcast, junk = xr.broadcast(weights, ocean_slope)
    return (weights_broadcast,)


@app.cell
def _(ocean_slope, weights_broadcast, xs):
    xs.pearson_r(ocean_slope['sfc'], ocean_slope, weights=weights_broadcast, skipna=True)
    return


@app.cell
def _(ocean_slope, weights_broadcast, xs):
    xs.pearson_r(ocean_slope['sfc'].sel({'lat':slice(-50, 50)}), ocean_slope.sel({'lat':slice(-50, 50)}), weights=weights_broadcast.sel({'lat':slice(-50, 50)}), skipna=True)
    return


@app.cell
def _(calc_dHF, ocean_flux, ocean_flux_climo, pd, xr):
    dHF_wind = calc_dHF(ocean_flux, ocean_flux_climo, climo_variables = ['ts', 'tas', 'huss', 'ps'])
    dHF_temp = calc_dHF(ocean_flux, ocean_flux_climo, climo_variables = ['uas', 'vas', 'ps'])
    dHF_all = calc_dHF(ocean_flux, ocean_flux_climo, climo_variables = [])

    dHF = xr.concat([dHF_all, dHF_wind, dHF_temp],
                   dim = pd.Index(['All', 'Wind', 'Temperature'], name='Varied Fields')).compute()
    return (dHF,)


@app.cell
def _(dHF, sst_idx, xs):
    dHF_slope = xs.linslope(sst_idx, dHF, dim='time')
    return (dHF_slope,)


@app.cell
def _(dHF_slope, facetplot):
    facetplot(dHF_slope['ql'], 'model', 'Varied Fields', vmin=-4, vmax=4, extend='both',cmap='RdBu_r', cbar_kwargs={'shrink':0.6})
    return


@app.cell
def _(dHF_slope, facetplot):
    facetplot(dHF_slope['qh'], 'model', 'Varied Fields', vmin=-4, vmax=4, cmap='RdBu_r')
    return


if __name__ == "__main__":
    app.run()
