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
    from functions import calc_land_mask, calc_fdbk, calc_dHF
    import pandas as pd
    return calc_dHF, calc_fdbk, calc_land_mask, ccrs, mo, np, pd, plt, xr, xs


@app.cell
def _(calc_land_mask, np, xr):
    #Replace here with amip-piForcing or historical simulations
    base_data = xr.open_dataset("/gws/nopw/j04/csgap/kkawaguchi/surface_data/piControl.nc", chunks={'lat':36, 'lon':36})


    #Regrid to calculate the land mask
    land_mask = calc_land_mask(base_data)

    #Calculate data over just the ocean
    ocean_flux = base_data[["rsds", "rlds", "rsus", "rlus", "hfss", "hfls","uas", "vas", "ps", "tas", "ts", "huss"]]
    ocean_flux = xr.where(land_mask == True, ocean_flux, np.nan)
    ocean_flux_climo = ocean_flux.mean("time")

    ocean_flux, ocean_flux_climo = xr.broadcast(ocean_flux, ocean_flux_climo)
    return base_data, ocean_flux, ocean_flux_climo


@app.cell
def _(ocean_flux):
    ocean_flux
    return


@app.cell
def _(base_data, np, ocean_flux, xr):
    #Calculate surface heat anomalies over the ocean
    ocean_anom_data = ocean_flux - ocean_flux.mean("time")
    ocean_anom_data["sfc_rad"] = ocean_anom_data["rlds"] + ocean_anom_data["rsds"] - ocean_anom_data["rlus"] - ocean_anom_data["rsus"]
    ocean_anom_data["sfc"] = ocean_anom_data["sfc_rad"] - ocean_anom_data["hfss"] - ocean_anom_data["hfls"]

    #Calculate the timeseries we use as the basis for our study
    weights = np.cos(np.radians(base_data.lat))

    T_series = (base_data["tas"].weighted(weights).mean(("lat", "lon")) - base_data["tas"].weighted(weights).mean(("lat", "lon", "time"))).rename('tas')
    SST_series = (ocean_flux["ts"] - ocean_flux["ts"].mean("time")).rename('ts')

    temp_series = xr.merge([T_series, SST_series])
    return SST_series, T_series, weights


@app.cell
def _(SST_series, T_series, calc_fdbk, weights):
    SST_window = calc_fdbk(SST_series, T_series)

    #Calculate the zonal gradient (West - East equatorial Pacific)
    sst_idx = (SST_window.sel({"lat":slice(-5, 5), "lon":slice(110,180)}).weighted(weights).mean(("lat", "lon")) - SST_window.sel({"lat":slice(-5, 5), "lon":slice(180,270)}).weighted(weights).mean(("lat", "lon")))
    return SST_window, sst_idx


@app.cell
def _(SST_window, plt):
    SST_window.isel({"time":0}).plot(col="model", robust=True)
    plt.show()
    return


@app.cell
def _(SST_window, ocean_window, sst_idx, xs):
    sfc_slope = xs.linslope(sst_idx,ocean_window["sfc"], dim="time")
    rad_slope = xs.linslope(sst_idx,ocean_window["sfc_rad"], dim="time")
    lh_slope = xs.linslope(sst_idx,-ocean_window["hfls"], dim="time")
    sh_slope = xs.linslope(sst_idx,-ocean_window["hfss"], dim="time")
    sst_slope = xs.linslope(sst_idx, SST_window, dim="time")
    return lh_slope, rad_slope, sfc_slope, sh_slope, sst_slope


@app.cell
def _(plt, sst_slope):
    sst_slope.plot(col="model", vmin=-3, vmax=3, cmap="coolwarm")
    plt.show()
    return


@app.cell
def _(sst_idx):
    sst_idx.plot.line(x="time")
    return


@app.cell
def _(calc_dHF, ocean_flux, ocean_flux_climo, pd, xr):
    dHF_wind = calc_dHF(ocean_flux, ocean_flux_climo, climo_variables = ['ts', 'tas', 'huss', 'ps'])
    dHF_temp = calc_dHF(ocean_flux, ocean_flux_climo, climo_variables = ['uas', 'vas', 'ps'])
    dHF_all = calc_dHF(ocean_flux, ocean_flux_climo, climo_variables = [])

    dHF = xr.concat([dHF_all, dHF_wind, dHF_temp],
                   dim = pd.Index(['All', 'Wind', 'Temperature'], name='Varied Fields'))
    return (dHF,)


@app.cell
def _(T_series, calc_fdbk, dHF):
    dHF_fdbk = calc_fdbk(dHF, T_series, yrs=30).chunk()
    return


@app.cell
def _(dHF):
    dHF
    return


@app.cell
def _(dHF_temp_window, dHF_wind_window, dHF_window, sst_idx, xs):
    lh_wind_slope = xs.linslope(sst_idx, dHF_wind_window["ql"], dim="time")
    lh_temp_slope = xs.linslope(sst_idx, dHF_temp_window["ql"], dim="time")
    lh_all_slope = xs.linslope(sst_idx, dHF_window["ql"], dim="time")

    sh_wind_slope = xs.linslope(sst_idx, dHF_wind_window["qh"], dim="time")
    sh_temp_slope = xs.linslope(sst_idx, dHF_temp_window["qh"], dim="time")
    sh_all_slope = xs.linslope(sst_idx, dHF_window["qh"], dim="time")
    return (
        lh_all_slope,
        lh_temp_slope,
        lh_wind_slope,
        sh_temp_slope,
        sh_wind_slope,
    )


@app.cell
def _(ccrs, lh_slope, lh_temp_slope, lh_wind_slope, plt):
    fig, ax = plt.subplots(nrows = 2, ncols = 2, layout="constrained", figsize=(8, 4.5), subplot_kw={"projection": ccrs.Robinson(central_longitude=210)})

    lh_wind_slope.mean("model").plot(ax=ax[0,0], vmin=-50, vmax=50, extend="both", cmap="coolwarm", transform=ccrs.PlateCarree(), add_colorbar=False)
    lh_temp_slope.mean("model").plot(ax=ax[0,1], vmin=-50, vmax=50, extend="both", cmap="coolwarm", transform=ccrs.PlateCarree(), add_colorbar=False)
    (lh_wind_slope+lh_temp_slope).mean("model").plot(ax=ax[1,0], vmin=-50, vmax=50, extend="both", cmap="coolwarm",transform=ccrs.PlateCarree(), add_colorbar=False)
    cb=lh_slope.mean("model").plot(ax=ax[1,1], vmin=-50, vmax=50, extend="both", cmap="coolwarm", transform=ccrs.PlateCarree(), add_colorbar=False)

    for i in range(4):
        ax[i//2, i%2].coastlines()

    fig.colorbar(cb, ax=ax[:,:], orientation="vertical", aspect=15, shrink=0.8, extend="both", label="dFlux/d$\\phi$")

    ax[0,0].set_title("Wind")
    ax[0,1].set_title("Temperature")
    ax[1,0].set_title("Sum")
    ax[1,1].set_title("LHF")
    return


@app.cell
def _(ccrs, plt, sh_slope, sh_temp_slope, sh_wind_slope):
    fig1, ax1 = plt.subplots(nrows = 2, ncols = 2, layout="constrained", figsize=(8, 4.5), subplot_kw={"projection": ccrs.Robinson(central_longitude=210)})

    sh_wind_slope.mean("model").plot(ax=ax1[0,0], vmin=-15, vmax=15, extend="both", cmap="coolwarm",transform=ccrs.PlateCarree(), add_colorbar=False)
    sh_temp_slope.mean("model").plot(ax=ax1[0,1], vmin=-15, vmax=15, extend="both", cmap="coolwarm",transform=ccrs.PlateCarree(), add_colorbar=False)
    (sh_wind_slope+sh_temp_slope).mean("model").plot(ax=ax1[1,0], vmin=-15, vmax=15, extend="both", cmap="coolwarm",transform=ccrs.PlateCarree(), add_colorbar=False)
    cb1 = sh_slope.mean("model").plot(ax=ax1[1,1], vmin=-15, vmax=15, extend="both",  cmap="coolwarm",transform=ccrs.PlateCarree(), add_colorbar=False)

    for j in range(4):
        ax1[j//2, j%2].coastlines()

    fig1.colorbar(cb1, ax=ax1[:,:], orientation="vertical", aspect=15, shrink=0.8, extend="both",label="dFlux/d$\\phi$")

    ax1[0,0].set_title("Wind")
    ax1[0,1].set_title("Temperature")
    ax1[1,0].set_title("Sum")
    ax1[1,1].set_title("SHF")
    return


@app.cell
def _(ccrs, lh_slope, plt, rad_slope, sfc_slope, sh_slope, sst_slope):
    fig2 = plt.figure(figsize=(10, 8), layout="constrained")
    subfigs = fig2.subfigures(nrows=2, ncols=1, height_ratios=[0.8, 1])

    ax3 = subfigs[0].subplots(1, 1, subplot_kw={"projection": ccrs.Robinson(central_longitude=210)})
    sst_slope.mean("model").plot(ax=ax3, transform=ccrs.PlateCarree(), vmin=-3, vmax=3, cmap="coolwarm", cbar_kwargs={"label":"dT/d$\\phi$"})
    ax3.set_title("Warming Pattern")
    ax3.set_position([0.2, 0, 0.5, 1])
    ax3.coastlines()


    ax4 = subfigs[1].subplots(2, 2, subplot_kw={"projection": ccrs.Robinson(central_longitude=210)})

    sfc_slope.mean("model").plot(ax=ax4[0,0], transform=ccrs.PlateCarree(), vmin=-40, vmax=40, cmap="coolwarm", extend="both", add_colorbar=False)
    rad_slope.mean("model").plot(ax=ax4[0,1], transform=ccrs.PlateCarree(), vmin=-40, vmax=40, cmap="coolwarm", extend="both", add_colorbar=False)
    lh_slope.mean("model").plot(ax=ax4[1,0], transform=ccrs.PlateCarree(), vmin=-40, vmax=40, cmap="coolwarm", extend="both", add_colorbar=False)
    cb2=sh_slope.mean("model").plot(ax=ax4[1,1], transform=ccrs.PlateCarree(), vmin=-40, vmax=40, cmap="coolwarm", extend="both", add_colorbar=False)

    for k in range(4):
        ax4[k//2, k%2].coastlines()

    fig2.colorbar(cb2, ax=ax4[:,:], orientation="vertical", aspect=15, shrink=0.8, extend="both", label="dFlux/d$\\phi$")

    ax4[0,0].set_title("Total Surface Flux")
    ax4[0,1].set_title("Radiation")
    ax4[1,0].set_title("LH")
    ax4[1,1].set_title("SH")

    plt.show()
    return


@app.cell
def _(lh_all_slope):
    lh_all_slope.mean("model").plot(vmin=-50, vmax=50, extend="both", cmap="coolwarm")
    return


@app.cell
def _(ocean_window, plt, sst_idx, xs):
    xs.linslope(sst_idx,ocean_window["sfc"], dim="time").plot(col="model", robust=True)
    plt.show()
    return


@app.cell
def _(ocean_window, plt, sst_idx, xs):
    xs.linslope(sst_idx,ocean_window["sfc_rad"], dim="time").plot(col="model", robust=True)
    plt.show()
    return


@app.cell
def _(ocean_window, plt, sst_idx, xs):
    xs.linslope(sst_idx,-ocean_window["hfls"], dim="time").plot(col="model", robust=True)
    plt.show()
    return


@app.cell
def _(ocean_window, plt, sst_idx, xs):
    xs.linslope(sst_idx,-ocean_window["hfss"], dim="time").plot(col="model", robust=True)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
