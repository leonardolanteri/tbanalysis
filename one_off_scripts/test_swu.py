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
from scipy.optimize import curve_fit

def fit_gauss(x, *p0):
    A, mu, sigma = p0
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

events = ak.from_parquet(
    "/home/etl/Test_Stand/tbanalysis/output_analysis_data/TestRunApr2/start_113239_stop_113338_setup_209_offset_15.0_energy_3.0_power_i1_module_209_bias_235_file_from_DESY_April_cubicLM_unbinned/fullCollection.parquet")

events.type.show()

events_path = "/home/etl/Test_Stand/tbanalysis/output_analysis_data/TestRunApr2/start_113239_stop_113338_setup_209_offset_15.0_energy_3.0_power_i1_module_209_bias_235_file_from_DESY_April_cubicLM_unbinned/fullCollection.parquet"

events_path = "./test_parquet/fullCollection.parquet"
events_path=  "./output_analysis_data/test_run/start_113239_stop_113338_setup_209_offset_15.0_energy_3.0_power_i1_module_209_bias_235_file_from_DESY_April2_goodrun_cubicLM_unbinned/fullCollection.parquet"

events_path = "/home/etl/Test_Stand/tbanalysis/output_analysis_data/desyapril_dev/start_114244_stop_114343_setup_209_offset_20.0_energy_5.0_power_i1_module_209_bias_225_file_from_DESY_April_cubicLM_unbinned/fullCollection.parquet"
            
#events_path = "./test_parquet/fullCollection.parquet"

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
 
plt.hist(clock, bins = 60)
plt.title("clock")
plt.show() 
    
#for delta T fit

num_bins = 100
minscale = 0
maxscale = 5

deltat = np.array(np.array(toa[nhits>0][:,0])-(clock[nhits>0]-mcp_time[nhits>0]))
col_nhit = np.array(col[nhits>0][:,0])
row_nhit = np.array(row[nhits>0][:,0])
pix_sel = (col_nhit==7) & (row_nhit==7)
n, bins, patches = plt.hist(deltat[pix_sel], bins = 100, density=0, range=[minscale, maxscale], alpha=1, label="Delta T")
plt.title("delta T")
plt.xlabel("DeltaT (ns)")
plt.ylabel("Entries")
    
A1 = np.max(n)
mu1 = bins[np.argmax(n)]
sigma1 = 0.1*mu1

print(np.shape(n),np.shape(bins))

if 1==1:
    fit_params, fit_cov = curve_fit(fit_gauss, bins[0:100], n[0:100], p0=(A1, mu1, sigma1))
    print(fit_params)
    A1 = fit_params[0]
    mu1 = fit_params[1]
    sigma1 = np.abs(fit_params[2])

    print("Sigma and its error")
    print(sigma1)
    plt.plot(bins[0:100], fit_gauss(bins[0:100], A1, mu1, sigma1), 'k--', lw=4, label="sigma = %.3f ns" % (sigma1))
#except:
#    print("not able to perform fit")
plt.legend()
plt.show() 

#end of deltaT fit
                
deltat = np.array(np.array(toa[nhits>0][:,0])-(clock[nhits>0]-mcp_time[nhits>0])) 
plt.hist(deltat, bins = 100, density=0, range=[minscale, maxscale], alpha=1, label="Delta T w/o pixel sel")
#plt.hist(deltat, bins = 600)
plt.title("delta T without pixel selection")
plt.xlabel("DeltaT (ns)")
plt.ylabel("Entries")
plt.show() 

print(deltat[~np.isnan(deltat)])
print(toa[nhits>0][:,0])
                
plt.hist2d(deltat[~np.isnan(deltat)], np.array(toa[nhits>0][:,0])[~np.isnan(deltat)], bins = 100)
plt.title("TOA vs deltat")
#plt.hist2d(clock[nhits>0], np.array(toa[nhits>0][:,0]), bins = 60)
plt.xlabel("DeltaT (ns)")
plt.ylabel("TOA (ns)")
plt.show()
                 
plt.hist2d(deltat[~np.isnan(deltat)], np.array(tot[nhits>0][:,0])[~np.isnan(deltat)], bins = 100)
plt.title("TOT vs deltat")
#plt.hist2d(clock[nhits>0], np.array(toa[nhits>0][:,0]), bins = 60)
plt.xlabel("DeltaT (ns)")
plt.ylabel("TOT (ns)")
plt.show()

#MCP amplitude

mcp_amp = mcp_amp[~np.isnan(mcp_amp)]
plt.hist(mcp_amp[(mcp_amp > -10) & (mcp_amp<10)],bins=100, range=(-1,0) )
plt.title("MCP amplitude")
plt.show()

plt.hist(np.array(toa[nhits>0][:,0]), bins = 60)
plt.title("toa")
plt.show() 
 
plt.hist(np.array(tot[nhits>0][:,0]), bins = 60)
plt.title("tot")
plt.show() 

plt.hist(np.array(cal[nhits>0][:,0]), bins = 60)
plt.title("Cal")
plt.show() 
        
plt.hist2d(np.array(tot[nhits>0][:,0]), np.array(toa[nhits>0][:,0]), bins=60)
plt.title("TOA vs TOT")
#plt.hist2d(clock[nhits>0], np.array(toa[nhits>0][:,0]), bins = 60)
plt.show()

#plt.hist(dt[nhits>0],bins=300, range=(-1,2))
plt.hist2d(clock[nhits>0], np.array(toa[nhits>0][:,0]), bins = 60)
plt.title("TOA vs Clock")
#plt.hist2d(clock[nhits>0], np.array(toa[nhits>0][:,0]), bins = 60)
plt.show()

plt.hist(mcp_time[nhits>0], bins = 60)
plt.title("mcp time")
plt.show() 
  
deltat = np.array(np.array(toa[nhits>0][:,0])-(clock[nhits>0]-mcp_time[nhits>0]))
plt.hist(deltat, bins = 600)
#plt.hist(deltat, bins = 600)
plt.title("delta T")
plt.show() 



