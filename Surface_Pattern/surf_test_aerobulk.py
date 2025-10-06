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
    return (
        calc_dHF,
        calc_fdbk,
        calc_land_mask,
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


    #Regrid to calculate the land mask
    land_mask = calc_land_mask(base_data)

    #Calculate data over just the ocean
    ocean_flux = base_data[["rsds", "rlds", "rsus", "rlus", "hfss", "hfls","uas", "vas", "ps", "tas", "ts", "huss"]]
    ocean_flux = xr.where(land_mask == True, ocean_flux, np.nan)
    ocean_flux_climo = ocean_flux.mean("time")

    ocean_flux, ocean_flux_climo = xr.broadcast(ocean_flux, ocean_flux_climo)
    return base_data, ocean_flux, ocean_flux_climo


@app.cell
def _(T_series, base_data, calc_fdbk):
    TOA = -(base_data['rlut'] + base_data['rsut'])
    TOA_fdbk = calc_fdbk(TOA, T_series)
    return TOA, TOA_fdbk


@app.cell
def _(TOA_fdbk, plt, weights):
    TOA_fdbk.weighted(weights).mean(('lat', 'lon')).plot.line(x='time')
    plt.ylim([-3.5, 0.5])
    plt.title('TOA Fdbk ($\\mathrm{W/m^2/K}$)')
    plt.show()
    return


@app.cell
def _(Fueglistaler_idx_raw, TOA, facetplot, sfc, xr, xs):
    energy_budget = xr.concat([TOA, sfc], dim='z')
    energy_budget = energy_budget.assign_coords({'z':['TOA', 'sfc']})
    facetplot(xs.linslope(Fueglistaler_idx_raw, energy_budget, dim='time'), 'model', 'z', title='Feedback ($\\mathrm{W/m^2/K}$)',
              vmin=-40, vmax=40, cmap='RdBu_r')
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
def _(base_data, calc_fdbk, np, ocean_flux, xr):
    #Calculate surface heat anomalies over the ocean
    ocean_anom_data = ocean_flux - ocean_flux.mean("time")
    ocean_anom_data["sfc_rad"] = ocean_anom_data["rlds"] + ocean_anom_data["rsds"] - ocean_anom_data["rlus"] - ocean_anom_data["rsus"]
    ocean_anom_data["sfc"] = ocean_anom_data["sfc_rad"] - ocean_anom_data["hfss"] - ocean_anom_data["hfls"]

    #Calculate the timeseries we use as the basis for our study
    weights = np.cos(np.radians(base_data.lat))

    T_series = (base_data["tas"].weighted(weights).mean(("lat", "lon")) - base_data["tas"].weighted(weights).mean(("lat", "lon", "time"))).rename('tas')
    SST_series = (ocean_flux["ts"] - ocean_flux["ts"].mean("time")).rename('ts')

    temp_series = xr.merge([T_series, SST_series])

    ocean_window = calc_fdbk(ocean_anom_data, T_series)
    return SST_series, T_series, ocean_window, weights


@app.cell
def _(SST_series, T_series, calc_fdbk, np, ocean_flux, weights, xr):
    SST_window = calc_fdbk(SST_series, T_series)

    #Calculate the SST indices (Watanabe and Fueglistaler)
    Watanabe_idx_raw = (ocean_flux['ts'].sel({"lat":slice(-5, 5), "lon":slice(180,270)}).weighted(weights).mean(("lat", "lon")) - ocean_flux['ts'].sel({"lat":slice(-5, 5), "lon":slice(110,180)}).weighted(weights).mean(("lat", "lon")))

    Watanabe_idx = calc_fdbk(Watanabe_idx_raw, T_series)

    tropical_SST = ocean_flux['ts'].sel({'lat':slice(-30, 30)}).compute()

    Fueglistaler_threshold = tropical_SST.weighted(weights).quantile(0.7, dim=('lat', 'lon'))

    Fueglistaler_idx_raw = xr.where(tropical_SST > Fueglistaler_threshold, tropical_SST, np.nan).weighted(weights).mean(('lat', 'lon')) - tropical_SST.weighted(weights).mean(('lat', 'lon'))

    Fueglistaler_idx = calc_fdbk(Fueglistaler_idx_raw, T_series)
    return Fueglistaler_idx, Fueglistaler_idx_raw, Watanabe_idx


@app.cell
def _(Fueglistaler_idx, Watanabe_idx, mo):
    idx_choice = mo.ui.dropdown(options = ['Equatorial Pacific: Watanabe et al. (2021)',
                                          'SST#: Fueglistaler (2019)'], 
                                value = 'SST#: Fueglistaler (2019)')

    idx_options = {'Equatorial Pacific: Watanabe et al. (2021)': Watanabe_idx, 'SST#: Fueglistaler (2019)': Fueglistaler_idx}

    idx_choice
    return idx_choice, idx_options


@app.cell
def _(idx_choice, idx_options):

    sst_idx = idx_options[idx_choice.value]
    return (sst_idx,)


@app.cell
def _(sst_idx):
    sst_idx.plot.line(x='time')
    return


@app.cell
def _(ocean_window, sst_idx, xs):
    ocean_slope = xs.linslope(sst_idx, ocean_window, dim='time').drop_vars(['quantile', 'height'])
    return (ocean_slope,)


@app.cell
def _(facetplot, ocean_slope):
    facetplot(ocean_slope['ts'], None, 'model', title='Warming pattern regressed onto SST index ($\\mathrm{K / K K^{-1}}$)',robust=True)
    return


@app.cell
def _(facetplot, ocean_slope):
    facetplot(ocean_slope['sfc'], 'model', None, title='SE imbalance regressed onto SST index ($\\mathrm{W m^{-2} K^{-1} / K K^{-1}}$)',robust=True)
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
def _(dHF):
    dHF['ql']
    return


@app.cell
def _(T_series, calc_fdbk, dHF):
    dHF_fdbk = calc_fdbk(dHF, T_series, yrs=30)
    return (dHF_fdbk,)


@app.cell
def _(dHF_fdbk, sst_idx, xs):
    dHF_slope = xs.linslope(sst_idx, dHF_fdbk, dim='time')
    return (dHF_slope,)


@app.cell
def _(dHF_slope, facetplot):
    facetplot(dHF_slope['ql'], 'model', 'Varied Fields', vmin=-75, vmax=75, cmap='RdBu_r')
    return


@app.cell
def _(dHF_slope, facetplot):
    facetplot(dHF_slope['qh'], 'model', 'Varied Fields', vmin=-75, vmax=75, cmap='RdBu_r')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
