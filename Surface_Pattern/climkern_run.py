import climkern as ck
import xarray as xr
import pandas as pd

data = xr.open_dataset('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/amip_piForcing_new_monthly.nc', chunks={'time':120, 'model':1})

# Calculate the TOA kernels

#LR, Planck = ck.calc_T_feedbacks(data.ta, data.ts, data.ps, data.ta, data.ts, data.ps,
#                                 kern='CloudSat', sky='all-sky', loc='TOA', fixRH=False)

#Prevent OOM killing the script
#LR.to_netcdf('LR_TOA.nc')
#LR.close()

#Planck.to_netcdf('Planck_TOA.nc')
#Planck.close()

#LR_cs, Planck_cs = ck.calc_T_feedbacks(data.ta, data.ts, data.ps, data.ta, data.ts, data.ps,
#                                 kern='CloudSat', sky='clear-sky', loc='TOA', fixRH=False)

#Prevent OOM killing the script
#LR_cs.to_netcdf('LRcs_TOA.nc')
#LR_cs.close()

#Planck_cs.to_netcdf('Planckcs_TOA.nc')
#Planck_cs.close()

#q_sw, q_lw = ck.calc_q_feedbacks(data.hus, data.ta, data.ps, data.hus, data.ps, 
#                                 kern='CloudSat', sky='all-sky', loc='TOA')

#Prevent OOM killing the script
#q_sw.to_netcdf('qsw_TOA.nc')
#q_sw.close()
#q_lw.to_netcdf('qlw_TOA.nc')
#q_lw.close()

#q_swcs, q_lwcs = ck.calc_q_feedbacks(data.hus, data.ta, data.ps, data.hus, data.ps, 
#                                 kern='CloudSat', sky='all-sky', loc='TOA')

#Prevent OOM killing the script
#q_swcs.to_netcdf('qswcs_TOA.nc')
#q_swcs.close()
#q_lwcs.to_netcdf('qlwcs_TOA.nc')
#q_lwcs.close()

#alb_as = ck.calc_alb_feedback(data.rsus, data.rsds, data.rsus, data.rsds, 
#                           kern='CloudSat', sky='all-sky',loc='TOA')

#alb_cs = ck.calc_alb_feedback(data.rsuscs, data.rsdscs, data.rsuscs, data.rsdscs, 
#                           kern='CloudSat', sky='clear-sky',loc='TOA')

#dCRE_SW = ck.calc_dCRE_SW(-data.rsut, -data.rsut, -data.rsutcs, -data.rsutcs)
#dCRE_LW = ck.calc_dCRE_LW(-data.rlut, -data.rlut, -data.rlutcs, -data.rlutcs)

#Reload all of the terms that required 4D computations
#LR = xr.open_dataarray('LR_TOA.nc')
#Planck = xr.open_dataarray('Planck_TOA.nc')
#LR_cs = xr.open_dataarray('LRcs_TOA.nc')
#Planck_cs = xr.open_dataarray('Planckcs_TOA.nc')
#q_lw = xr.open_dataarray('qlw_TOA.nc')
#q_sw = xr.open_dataarray('qsw_TOA.nc')
#q_lwcs = xr.open_dataarray('qlwcs_TOA.nc')
#q_swcs = xr.open_dataarray('qswcs_TOA.nc')

#cloud_LW = ck.calc_cloud_LW(LR+Planck, LR_cs+Planck_cs, q_lw, q_lwcs, dCRE_LW)
#cloud_SW = ck.calc_cloud_LW(alb_as, alb_cs, q_sw, q_swcs, dCRE_SW)

#TOA = xr.concat([Planck, LR, alb_as, q_sw+q_lw, cloud_LW, cloud_SW], 
#                pd.Index(['Planck', 'LR', 'Albedo', 'WV', 'CLD_LW', 'CLD_SW'],name='component'))

#TOA.to_netcdf('TOA_kern.nc')

#LR.close()
#Planck.close()
#q_lw.close()
#q_sw.close()
#q_swcs.close()
#TOA.close()

# Calculate the SFC kernels

#LR, Planck = ck.calc_T_feedbacks(data.ta, data.ts, data.ps, data.ta, data.ts, data.ps,
#                                 kern='CloudSat', sky='all-sky', loc='SFC', fixRH=False)

#Prevent OOM killing the script
#LR.to_netcdf('LR_SFC.nc')
#LR.close()

#Planck.to_netcdf('Planck_SFC.nc')
#Planck.close()

#LR_cs, Planck_cs = ck.calc_T_feedbacks(data.ta, data.ts, data.ps, data.ta, data.ts, data.ps,
#                                 kern='CloudSat', sky='clear-sky', loc='SFC', fixRH=False)

#Prevent OOM killing the script
#LR_cs.to_netcdf('LRcs_SFC.nc')
#LR_cs.close()

#Planck_cs.to_netcdf('Planckcs_SFC.nc')
#Planck_cs.close()


alb_as = ck.calc_alb_feedback(data.rsus, data.rsds, data.rsus, data.rsds, 
                           kern='CloudSat', sky='all-sky',loc='SFC')

alb_cs = ck.calc_alb_feedback(data.rsuscs, data.rsdscs, data.rsuscs, data.rsdscs, 
                           kern='CloudSat', sky='clear-sky',loc='SFC')

#q_sw, q_lw = ck.calc_q_feedbacks(data.hus, data.ta, data.ps, data.hus, data.ps, 
#                                 kern='CloudSat', sky='all-sky', loc='SFC')

#Prevent OOM killing the script
#q_sw.to_netcdf('qsw_SFC.nc')
#q_sw.close()
#q_lw.to_netcdf('qlw_SFC.nc')
#q_lw.close()

#q_swcs, q_lwcs = ck.calc_q_feedbacks(data.hus, data.ta, data.ps, data.hus, data.ps, 
#                                 kern='CloudSat', sky='all-sky', loc='SFC')

#Prevent OOM killing the script
#q_swcs.to_netcdf('qswcs_SFC.nc')
#q_swcs.close()
#q_lwcs.to_netcdf('qlwcs_SFC.nc')
#q_lwcs.close()

dCRE_SW = ck.calc_dCRE_SW(data.rsds-data.rsus, data.rsds-data.rsus, 
                          data.rsdscs-data.rsuscs, data.rsdscs-data.rsuscs)
dCRE_LW = ck.calc_dCRE_LW(data.rlds-data.rlus, data.rlds-data.rlus, 
                          data.rldscs-data.rlus, data.rldscs-data.rlus)

#Reload all of the terms that required 4D computations
LR = xr.open_dataarray('LR_SFC.nc')
Planck = xr.open_dataarray('Planck_SFC.nc')
LR_cs = xr.open_dataarray('LRcs_SFC.nc')
Planck_cs = xr.open_dataarray('Planckcs_SFC.nc')
q_lw = xr.open_dataarray('qlw_SFC.nc')
q_sw = xr.open_dataarray('qsw_SFC.nc')
q_lwcs = xr.open_dataarray('qlwcs_SFC.nc')
q_swcs = xr.open_dataarray('qswcs_SFC.nc')

cloud_LW = ck.calc_cloud_LW(LR+Planck, LR_cs+Planck_cs, q_lw, q_lwcs, dCRE_LW)
cloud_SW = ck.calc_cloud_LW(alb_as, alb_cs, q_sw, q_swcs, dCRE_SW)

SFC = xr.concat([Planck, LR, alb_as, q_sw+q_lw, cloud_LW, cloud_SW], 
                pd.Index(['Planck', 'LR', 'Albedo', 'WV', 'CLD_LW', 'CLD_SW'],name='component'))

SFC.to_netcdf('SFC_kern.nc')

#TOA = xr.open_dataarray('TOA_kern.nc')

#kern_results = xr.concat([TOA, SFC], pd.Index(['TOA', 'SFC'], name='loc'))
#kern_results.to_netcdf('/gws/ssde/j25a/csgap/kkawaguchi/surface_data/amip_piForcing_kernel_calcs.nc')