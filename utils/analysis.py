#!/usr/bin/env python3
import os
import numpy as np
import awkward as ak
from typing import List
from datetime import datetime
from hist import Hist
from scipy.optimize import curve_fit
# Error handling
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass
from scipy.stats import stats

import sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def elapsed_time(start_time):
    elapsed_time_calc = datetime.now() - start_time
    total_seconds = elapsed_time_calc.total_seconds()
    total_minutes, remainder = divmod(total_seconds, 60)
    return f"{total_minutes} minutes, {remainder:.1f} seconds"

def convert_dict_to_str(conf):
    """
    Converts a configuration dictionary to directory string
    """
    output_dir = ''
    for key, value in conf.items():
        if key == "run_paths":
            continue
        if key == "file_like":
            continue
        if key == 'offset':
            value = float(value)
        if output_dir == '':
            output_dir+=key+f'_{value}'
        elif type(value) == bool:
            output_dir+=f"_{key}"
        elif key == 'tag':
            output_dir+=f"_{value}"
        else:
            output_dir+=f'_{key}'+f'_{value}'
    return output_dir

@dataclass
class Thresholds:
    mcp_amp_low: float
    mcp_amp_high: float
    toa_code_low: float
    toa_code_high: float
    tot_code_low: float
    tot_code_high: float

    def toa_code_cut(self, toa_code:ak.Array):
        return ((toa_code>self.toa_code_low) & (toa_code<self.toa_code_high))
    
    def tot_code_cut(self, tot_code:ak.Array):
        return (tot_code>self.tot_code_low) & (tot_code<self.tot_code_high)
    
    def mcp_amp_cut(self, mcp_amplitude:ak.Array):
        return ((mcp_amplitude > self.mcp_amp_low) & (mcp_amplitude < self.mcp_amp_high))


def get_run_files(run_data_path: str, run_start:int, run_stop:int, reg_expression: str, verbose=False) -> List[str]:
    import re
    matched_files = []
    found_run_numbers = []
    for data_file in os.listdir(run_data_path): #loop through all files in run data path and search for matches
        data_path = os.path.join(run_data_path, data_file)
        if match:=re.search(reg_expression, data_file): #need to check path exists and if broken links
            matched_groups = match.groups() #all the selected parts of the filename from regular expression
            #-------------Define Filename Match Conditions-----------------#
            #if only one match, better be a run number and it should be between start and stop
            single_run_file_match = (len(matched_groups) > 0 and matched_groups[0].isdigit() and run_start <= int(matched_groups[0]) <= run_stop)
            #if more than 2, run start and stop better be in match
            multi_run_file_match = (len(matched_groups) > 0 and str(run_start) in matched_groups and ((str(run_stop+1) in matched_groups or str(run_stop) in matched_groups)))

            if single_run_file_match or multi_run_file_match: #single run match probably also grabs multi but this is to be more readable...
                if not os.path.isfile(data_path):
                    print(f"Potentially broken link for: {data_path}")
                    continue
                matched_files.append(data_path)
                if verbose:
                    if multi_run_file_match:
                        found_run_numbers.append(data_file)
                    else:
                        found_run_numbers.append(matched_groups[0])
    #find the matched files lists that are not empty, if multiple, raise error
    if verbose:
        print("Found runs:")
        print(sorted(found_run_numbers))
    return matched_files 

def hit_map(events: ak.Array):
    hit_matrix = np.zeros((16,16))
    for row in range(16):
        for col in range(16):
            pix_sel = (events.row==row)&(events.col==col)
            hit_matrix[row][col] += len(ak.flatten(events.cal_code[pix_sel]))
    return hit_matrix

def cal_mode(events: ak.Array, thresholds: Thresholds):
    cal_mode = np.zeros((16,16))
    mcp_amp_sel = thresholds.mcp_amp_cut(events.mcp_amplitude)
    for row in range(16):
        for col in range(16):
            pix_sel = (events.row==row)&(events.col==col)
            cal_val = ak.flatten(events.cal_code[pix_sel & mcp_amp_sel])
            if len(cal_val) != 0:
                cal_mode[row][col] = stats.mode(ak.to_numpy(cal_val))[0]
            else:
                cal_mode[row][col] = -999
    return cal_mode

def fnalOffset(events):
    clockScale = 24.95
    shifts = np.array((events.mcp_timestamp - events.clock_timestamp)/clockScale,dtype=int)
    shifts = ak.where(shifts<0,0,shifts)
    offset = -clockScale*np.array(shifts,dtype=float)
    return offset

def fit_gauss(h: Hist) -> tuple[np.ndarray]:
    """
    Fits gaussian to 1D Hist histogram
    """
    #fitting function
    gaus = lambda x, N, mu, sigma: N*np.exp(-(x-mu)**2/(2.0*sigma**2))

    bin_centers, hist_values = h.axes.centers[0], h.values()
    if np.sum(hist_values) < 10: #number of data points
        #print("Not enough data! Skipping fit.")
        return
    #https://github.com/scikit-hep/hist/blob/6fb3ecd07d1f9a4758cd5d5ccf89559ed572ca9a/src/hist/plot.py#L282
    N = float(hist_values.max())
    mu = (hist_values * bin_centers).sum() / hist_values.sum()
    sigma = (hist_values * np.square(bin_centers - mu)).sum() / hist_values.sum()

    hist_uncert = np.sqrt(h.variances())
    #https://github.com/scikit-hep/hist/blob/6fb3ecd07d1f9a4758cd5d5ccf89559ed572ca9a/src/hist/plot.py#L150
    mask = hist_uncert != 0.0
    try:
        popt, pcov = curve_fit(
            gaus, 
            bin_centers[mask], 
            hist_values[mask], 
            # sigma=hist_uncert[mask],
            # absolute_sigma=True,
            p0=[N, mu, sigma]
        )
    except RuntimeError:
        print("Could not find optimal parameters, skipping fit...")
        return
    #errors on the fitted values
    perr = np.sqrt(np.diagonal(pcov))

    #compute chi square
    r = h.values() - gaus(bin_centers, *popt)
    chisq = np.sum((r/1)**2)

    # Changing from sum -> len, each fitting point is a data point
    deg_freedom = len(h.values()) - 3
    red_chisq = chisq/deg_freedom
    return popt, pcov, perr, red_chisq
