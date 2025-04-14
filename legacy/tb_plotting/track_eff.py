import numpy as np
import awkward as ak
import uproot
import hist
import matplotlib.pyplot as plt
import mplhep as hep
from os import path
import sys
sys.path.append('/home/users/hswanson13/tbanalysis/')
from utils.analysis import data_mask, elapsed_time
from datetime import datetime

def calc_total_num_events(run_paths, step_size):
    return sum([len(events) for events in uproot.iterate(run_paths, step_size=step_size)])

print("WARNING IT DROPS THE LAST 5000 EVENTS")

plot_dir = '/home/users/hswanson13/tbanalysis/output_analysis_data/eff_plots_old'
telescope_data_path = "/ceph/cms/store/user/azecchin/DESY_ETROC_telescope_merged/ETROC_Telescope_data/"
tel_runs = "Run_2088_2239" #"Run_5050_5200" 
n_merged_files = 7
tel_run_paths = [path.join(telescope_data_path, tel_runs+f"_{i}.root") for i in range(n_merged_files)] #+':mergedTree'

step_size = 50000
chi2_cut_high = 100

# telescope_data_path = "/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/ScopeData/ETROC_Telescope_data_NEW"
# tel_runs = "Run_5050_5199.root:pulse" #Run_4000_4199.root is under pulse
# tel_run_path = path.join(telescope_data_path, tel_runs) #+':mergedTree'

bins_x = 150
bins_y = 150
width=10
xhits_axis = hist.axis.Regular(start=-10, stop=10, bins=bins_x, name="xhits", label="x tracks")
yhits_axis = hist.axis.Regular(start=-10, stop=10, bins=bins_y, name="yhits", label="y tracks")

tot_traks = hist.Hist(xhits_axis,yhits_axis)
traks_w_hits = hist.Hist(xhits_axis, yhits_axis)

start_time = datetime.now()
print("----------CALCULATING Efficiency------------------")
event_cntr = 0
total_num_events = calc_total_num_events(tel_run_paths, step_size)
for events in uproot.iterate(tel_run_paths, step_size=step_size):
    if event_cntr + len(events) == total_num_events:
        events = events[:-5000] #drop last 5000 events, 1 run file

    clock_sel = ((events['Clock'] > -1) & (events['Clock'] < 5))
    chi2_sel = events.chi2 < chi2_cut_high
    bad_traker = (events.x > -500) & (events.y > -500)
    data_sel = data_mask(events, toa_low=20, toa_high=600, tot_low=50, tot_high=200)

    trak_sel = bad_traker & clock_sel & chi2_sel
    tot_traks.fill(events[trak_sel].x, events[trak_sel].y)

    hits_sel = ((events.nhits > 0))
    traks_w_hits.fill(events[trak_sel & hits_sel].x, events[trak_sel & hits_sel].y)
    
    event_cntr += len(events)
    
print(f"----------FINISHED Efficiency ({elapsed_time(start_time)})--------------")

fig, ax = plt.subplots(1,1,figsize=(10,10))
etroc_eff = traks_w_hits / tot_traks
density = etroc_eff.values()
mesh = ax.pcolormesh(*etroc_eff.axes.edges.T, density.T)
fig.colorbar(mesh)
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('Sensor Hit Efficiency')

plt.savefig(f"{plot_dir}/eff_plot_FNALMerged_{tel_runs}.png")