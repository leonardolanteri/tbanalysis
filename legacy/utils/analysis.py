#!/usr/bin/env python3
import os
import numpy as np
import awkward as ak
from typing import List
from datetime import datetime
from hist import Hist
from scipy.optimize import curve_fit
from awkward.highlevel import Array as akArray
# Error handling
import warnings
warnings.filterwarnings("ignore")

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

def run_chunker(run_list):
    chunk_size=40
    return (run_list[pos:pos+chunk_size] for pos in range(0, len(run_list), chunk_size))

def module_select_filter(events: akArray, tb_config: dict, is_telescope_setup: bool, module: int, chunk_counter: int) -> akArray:
    def make_arr_of_fields(events: akArray, event_fields:list, channel_select_fields:dict) -> akArray:
        """
        Turns events from fields of arrays to array of fields or simpler in terms, array of hits.
        
        For example, the loaded events looks like (fields of arrays):
        393932 * {
            i_evt: uint32,
            segment_time: float32,
            channel: 4 * 502 * float32,
            ...
            LP2_20: 4 * float32,
            Clock: float64,
            tot_code: var * float32,
            row: var * int32,
            ...
        }

        However for some cases it can make applying masks confusing. This function converts it to an array of fields:
        393932 * var * {
            row: int32,
            col: int32,
            chipid: int32,
            tot_code: float32,
            toa_code: float32,
            cal_code: float32,
            Clock: float64,
            trigger: float32,
            nhits: int32,
            event: int32
        }
        Or: [[], [], [], ..., [{row: 9, col: 6, chipid: 148, tot_code: 66, ...}], []]

        To be able to get one number per field you have to select the channel from the array and that is what this line does:
        event_dict[field] = events[field][:,channel]
        """ 
        event_dict = {}
        for field in event_fields:
            # if field == 'nhits': 
            #     event_dict[field] = ak.flatten(events[field])
            # else:
            event_dict[field] = events[field]
        for field, channel in channel_select_fields.items():
            # Conceptually important line! Selecting the used channel lets us turn this into an array of fields!
            event_dict[field] = events[field][:,channel]
        return ak.zip(event_dict)
    
    ##
    #### Print selected and available chipids to user
    ##
    print(f'---------CHUNK: {chunk_counter} MODULE SELECT FILTER and CHIPID CHECKS------------')
    flat_chipid = ak.flatten(events.chipid).to_numpy()
    print(f"DETECTED THESE CHIPIDS IN DATA (chosen chipid={module}<<2={module<<2}): {np.unique(flat_chipid)}")  

    ##
    ##### Chipid and row need to be same shape to turn events into an array of fields (which lets us select on chipid id!)
    ##
    shape_requirement = ak.num(events.chipid) == ak.num(events.row)
    tb_config_event_fields = tb_config['EVENT_FIELDS']
    if is_telescope_setup and not ak.all(shape_requirement):
        #for single chip/module tb setups we do not need to drop events
        # ---> especially because for some tb NO chipids have the same shape
        print(f'Chipid does not have same shape as row! {len(events[shape_requirement])}/{len(events)} had good chipid shapes. Cutting bad chipids...')
        bad_shapes = ak.num(events.chipid) != ak.num(events.row)
        print(f"Some bad chipid shapes: {events[bad_shapes].chipid}")
        print(f"Corresponding rows: {events[bad_shapes].row}")
        events = events[shape_requirement]
    elif ak.all(shape_requirement):
        print("Chipid has same shape as row/col!")
    else:
        print("Setup not telescope. Safely removing chipid from all events...")
        tb_config_event_fields = [evf for evf in tb_config['EVENT_FIELDS'] if evf != 'chipid']
    
    ##
    #### Turn events into an array of fields (Only permissable if chipid has the same shape as row! what above code accounts for)
    ##
    print(f"CONVERTING EVENTS TO ARRAY OF FIELDS: {tb_config_event_fields+list(tb_config['CHANNEL_SELECT_FIELDS'].keys())}")
    events = make_arr_of_fields(events, tb_config_event_fields, tb_config['CHANNEL_SELECT_FIELDS'])
    
    ##
    ### Select the events with the correct chipid
    ##
    if is_telescope_setup: # (logic is correct here from above conditionals)
        chipid_sel = (events.chipid==(module)<<2)
        num_chipid_evs = len(ak.flatten(events[chipid_sel]))
        if num_chipid_evs == 0:
            print(f"NO EVENTS WITH SELECTED MODULE ({module}) or CHIPID={module<<2}")
            exit()
        print(f"Number of events with chipid ({module<<2}): {num_chipid_evs}/{len(events)}")
        events = events[chipid_sel]
    print("------------------------------------------------------------------------------------")
    return events

def data_mask(events, cut_eff: dict=None, scope_low=50, scope_high=300, toa_low=20, toa_high=600, tot_low=70, tot_high=130, root_dumper=False):
    """
    (scope_sel) & toa_sel & tot_sel
    """
    if not root_dumper:
        scope_sel = ((events.amp > scope_low) & (events.amp < scope_high))
    
    #Time of Arrival
    toa_sel = ((events.toa_code>toa_low) & (events.toa_code<toa_high))
    #Time of Threshold Selection: time above the threshold daq value
    tot_sel = (events.tot_code>tot_low) & (events.tot_code<tot_high)
    #total selection mask for an event  
    total_sel = (scope_sel) & toa_sel & tot_sel if not root_dumper else toa_sel & tot_sel

    if cut_eff is not None:
        cut_eff['total'] += int(ak.count_nonzero(events.nhits>0))
        cut_eff['scope'] += int(ak.count_nonzero(scope_sel & (events.nhits>0)))
        cut_eff['toa_code_low'] += int(ak.count_nonzero(scope_sel & (events.toa_code>toa_low)))
        cut_eff['toa_code_high'] += int(ak.count_nonzero(scope_sel & toa_sel))
        cut_eff['tot_code_low'] += int(ak.count_nonzero(scope_sel & toa_sel & (events.tot_code>tot_low)))
        cut_eff['tot_code_high'] += int(ak.count_nonzero(scope_sel & toa_sel & tot_sel))
    return total_sel

def fnalOffset(events):
    clockScale = 24.95
    shifts = np.array((events.trigger - events.Clock)/clockScale,dtype=int)
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
    deg_freedom = sum(h.values()) - 3
    red_chisq = chisq/deg_freedom
    return popt, pcov, perr, red_chisq
