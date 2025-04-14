import uproot
import awkward as ak
import hist
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# events_path = "/home/etl/Test_Stand/tbanalysis/output_analysis_data/TestRunApr2/start_113239_stop_113338_setup_209_offset_15.0_energy_3.0_power_i1_module_209_bias_235_file_from_DESY_April_cubicLM_unbinned/fullCollection.parquet"
# events_path = "./output_analysis_data/test_run/start_113239_stop_113338_setup_209_offset_15.0_energy_3.0_power_i1_module_209_bias_235_file_from_DESY_April2_goodrun_cubicLM_unbinned/fullCollection.parquet"
events_path = "./output_analysis_data/offsetScan@225/start_114344_stop_114443_setup_209_offset_10.0_energy_5.0_power_i1_module_209_bias_225_file_from_DESY_April_cubicLM_unbinned/fullCollection.parquet"
events = ak.from_parquet(events_path)

events.type.show()

data = pd.read_parquet(events_path)

nhits = ak.flatten(data["nhits"])

row = ak.Array(data["row"])
col = ak.Array(data["col"])
toa = ak.Array(data["toa"])
tot = ak.Array(data["tot"])
cal = ak.Array(data["cal_code"])
clock = np.array(data["clock_timestamp"])
mcp_time = np.array(data["mcp_timestamp"])
mcp_amp = np.array(data["mcp_amplitude"])
dt = np.array(data["dt"])

#plt.hist(tot[nhits>0][:,0])

#clock[clock<0] = -1*clock[clock<0]
 
clock[clock>0] = 12.5-clock[clock>0]
clock[clock<0] = 12.5-clock[clock<0]-25
 
'''
plt.hist(clock, bins = 60)
plt.title("clock")
plt.show() 
'''

sel_row = 7
sel_col = 7

deltat = np.array(np.array(toa[nhits>0][:,0])-(clock[nhits>0]-mcp_time[nhits>0]))
col_nhit = np.array(col[nhits>0][:,0])
row_nhit = np.array(row[nhits>0][:,0])
toa_nhit = np.array(toa[nhits>0][:,0])
tot_nhit = np.array(tot[nhits>0][:,0])
cal_nhit = np.array(cal[nhits>0][:,0])
pix_sel = (col_nhit==sel_col) & (row_nhit==sel_row) & (tot_nhit>2) & (cal_nhit>160) & (cal_nhit<200)

plot_sel = (~np.isnan(mcp_amp) & ~np.isinf(mcp_amp))

mcp_amp = mcp_amp[plot_sel]
mcp_time = mcp_time[plot_sel]

plt.hist(mcp_amp[(mcp_amp > -10) & (mcp_amp<10)],bins=100, range=(-1,0) )
plt.title("MCP amplitude")
plt.savefig("./tmp_output_plots/MCP_ampl.png")
plt.close()

plt.hist(np.array(toa[nhits>0][:,0]), bins = 60)
plt.title("toa")
plt.savefig("./tmp_output_plots/toa_all.png")
plt.close()
 
plt.hist(np.array(tot[nhits>0][:,0]), bins = 60)
plt.title("tot")
plt.savefig("./tmp_output_plots/tot_all.png")
plt.close()

plt.hist(np.array(cal[nhits>0][:,0]), bins = 60)
plt.title("Cal")
plt.savefig("./tmp_output_plots/cal_all.png")
plt.close()

plt.hist2d(np.array(tot[nhits>0][:,0]), np.array(toa[nhits>0][:,0]), bins=60)
plt.title("TOA vs TOT")
plt.savefig("./tmp_output_plots/toa_vs_tot_all.png")
plt.close()

plt.hist2d(clock[nhits>0], np.array(toa[nhits>0][:,0]), bins = 60)
plt.title("TOA vs Clock")
plt.savefig("./tmp_output_plots/toa_vs_clock_all.png")
plt.close()

plt.hist(mcp_time, bins = 60)
plt.title("mcp time")
plt.savefig("./tmp_output_plots/mcp_time_all.png")
plt.close()

print(len(toa))
print(len(plot_sel))
print(len(toa[plot_sel]))
print(len(mcp_time))

deltat = np.array(np.array(toa[plot_sel][nhits[plot_sel]>0][:,0])-(clock[plot_sel][nhits[plot_sel]>0]-mcp_time[nhits[plot_sel]>0]))
plt.hist(deltat, bins = 600)
plt.title("delta T")
plt.savefig("./tmp_output_plots/deltat.png")
plt.close()

for sel_row in range(16):
    for sel_col in range(16):
        print(f"Row: {sel_row}, Col: {sel_col}")
        pix_sel = ((col_nhit==sel_col) & (row_nhit==sel_row) & (tot_nhit>2) & (cal_nhit>160) & (cal_nhit<200) & (plot_sel[nhits>0])) # & (nhits[plot_sel]>0)
        print(len(pix_sel))
        print(len(deltat))
        deltat_pix = deltat[pix_sel]
        toa_pix    = np.array(toa_nhit)[pix_sel]
        tot_pix    = np.array(tot_nhit)[pix_sel]

        plotting_sel = ((~np.isnan(deltat_pix)) & (~np.isinf(deltat_pix)))

        if np.sum(plotting_sel) == 0: continue

        plt.hist(deltat_pix, bins = 100)
        plt.title("delta T")
        plt.savefig("./tmp_output_plots/deltat_{sel_row}_{sel_col}.png")
        plt.close()

        plt.hist2d(deltat_pix[plotting_sel], toa_pix[plotting_sel], range = ([min(deltat_pix[plotting_sel]),max(deltat_pix[plotting_sel])], [min(toa_pix[plotting_sel]),max(toa_pix[plotting_sel])]), bins = 100)
        plt.title("TOA vs deltat")
        plt.savefig(f"./tmp_output_plots/deltat_vs_toa_{sel_row}_{sel_col}.png")
        plt.close()

        plt.hist2d(deltat_pix[plotting_sel], tot_pix[plotting_sel], range = ([min(deltat_pix[plotting_sel]),max(deltat_pix[plotting_sel])], [min(tot_pix[plotting_sel]),max(tot_pix[plotting_sel])]), bins = 100)
        plt.title("TOT vs deltat")
        plt.savefig(f"./tmp_output_plots/deltat_vs_tot_{sel_row}_{sel_col}.png")
        plt.close()


