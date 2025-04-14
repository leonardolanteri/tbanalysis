import uproot
import awkward as ak
import hist
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep
import sys
sys.path.append("/home/etl/Test_Stand/tbanalysis") #stupid python
from utils import plotting as pu
import numpy as np
from importlib import reload

#sm_run =pu.TBplot("/home/etl/Test_Stand/tbanalysis/output_analysis_data/first_small/start_112817_stop_112838_setup_209_offset_10.0_energy_3.0_power_i1_module_209_bias_180_file_from_DESY_April_cubicLM_unbinned",
sm_run =pu.TBplot("/home/etl/Test_Stand/tbanalysis/output_analysis_data/DESY_APRIL_2025_Apr3night_Irene/start_114044_stop_114143_setup_209_offset_15.0_energy_5.0_power_i1_module_209_bias_240_file_from_DESY_April_cubicLM_unbinned",
load_full_collection=True)


sm_run.hit_map()

sm_run.histo1D('toa_code')
sm_run.heatmap("toa_code", "tot_code")


mcp_t = sm_run.events.mcp_timestamp
hist_axis = hist.axis.Regular(100,-1,1, name="mcp")
h = Hist(hist_axis).fill(mcp_t)
hep.histplot(h, color='black')

clk_t = sm_run.events.clock_timestamp
hist_axis = hist.axis.Regular(100,-25,25, name="clock")
h = Hist(hist_axis).fill(clk_t)
hep.histplot(h, color='black')

sys.path.append("/home/etl/Test_Stand/tbanalysis") #stupid python
from utils import plotting as pu
import uproot

from utils.clock_fit import calc_clock
raw = uproot.open("/media/etl/Storage/DESY_April_2025/merged/run_114143.root")["pulse"].arrays()

s, v = raw.clock_seconds, raw.clock_volts
# ct = calc_clock(s*10**9, v, 0.25, 0.8, 0.5)


# i = 65
# plt.axvline(ct[i])
# plt.scatter(s[i]*10**9, v[i])

s = ak.flatten(s)
v = ak.flatten(v)

hist_2d = Hist(
    hist.axis.Regular(100,-25,25, name='clock'),
    hist.axis.Regular(100, -2,2, name='toa'),
).fill(s, v)

hep.hist2dplot(hist_2d)

clock_timestamps2 = sm_run.events.clock_timestamp
toa = sm_run.events.toa

good_clock = clock_timestamps2[ak.num(toa)==1]
good_toa = ak.flatten(toa[ak.num(toa)==1])

print(good_clock)
print(good_toa)

print(len(good_clock))
print(len(good_toa))
hist_2d = Hist(
    hist.axis.Regular(100,-25,25, name='clock'),
    hist.axis.Regular(100, -25,25, name='toa'),
).fill(good_clock, good_toa)

fig, ax = plt.subplots(1,1,figsize=(10,10))
hep.hist2dplot(hist_2d)
fig.savefig(f"test_data_quality.png")

# test = pu.TBplot(
#     "/home/etl/Test_Stand/tbanalysis/output_analysis_data/DESY_APRIL_2025_Apr3night_Irene/start_114044_stop_114143_setup_209_offset_15.0_energy_5.0_power_i1_module_209_bias_240_file_from_DESY_April_cubicLM_unbinned",
#     load_full_collection=True)


# test.heatmap('toa_code', 'tot_code', save_path="./")
# test.histo1D('mcp_amplitude')

#test.histo1D('dt_corr', pix=(8,9), save_path="./")