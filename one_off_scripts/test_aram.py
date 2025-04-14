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

base = "./output_analysis_data/test_run/"
events_path = f"{base}start_113239_stop_113338_setup_209_offset_15.0_energy_3.0_power_i1_module_209_bias_235_file_from_DESY_April2_goodrun_cubicLM_unbinned/fullPresel.parquet"

presel = pd.read_parquet(events_path)

print(len(presel))

# ['dt', 'toa', 'tot', 'row', 'col', 'toa_code', 'tot_code', 'cal_code', 'cal_mode', 'dt_corr']

toa = []
tot = []
cal = []
dt  = []
row = []
col = []

check = 0

for i in range(len(presel)):
    if (len(presel.iloc[i][0]) == 0): continue

    for j in range(len(presel.iloc[i][0])):
        if (check==0):
            print(presel.iloc[i][0][j].keys())
            check = 1
        toa.append(presel.iloc[i][0][j]["toa"])
        tot.append(presel.iloc[i][0][j]["tot"])
        cal.append(presel.iloc[i][0][j]["cal_code"])
        dt .append(presel.iloc[i][0][j]["dt"])
        row.append(presel.iloc[i][0][j]["row"])
        col.append(presel.iloc[i][0][j]["col"])

        # print(toa)

plt.hist2d(row, col, bins = 16)
plt.show()
plt.close()

plt.hist(toa, bins = 100, histtype = "step")
plt.show()
plt.close()

plt.hist(tot, bins = 100, histtype = "step")
plt.show()
plt.close()

plt.hist(cal, bins = 100, histtype = "step")
plt.show()
plt.close()

plt.hist(dt, bins = 100, histtype = "step")
plt.show()
plt.close()





