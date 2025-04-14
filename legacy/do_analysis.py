import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import json
import uproot
import numpy as np
import awkward as ak
import hist
from scipy.stats import stats
from utils.file_handler import ParquetManager, ensure_dir_exists, save_data
import utils.analysis as au
import utils.timewalk as tw
EVENT_FIELDS = ['row','col','chipid','tot_code','toa_code','cal_code','Clock','nhits','event']
TB_CONFIGS = {
    'DESY': {
        'EVENT_FIELDS': EVENT_FIELDS,
        'CHANNEL_SELECT_FIELDS':{
            'LP2_20': 1,
            'amp': 1,
            'timeoffsets': 2            
        },
        'DATA_PATH': '/ceph/cms/store/user/iareed/ETL/DESY_Mar24',
        'FILENAME_REGEX': r'run_(\d+).root',
    },
    'DESY_MERGED': {
        'EVENT_FIELDS': EVENT_FIELDS,
        'CHANNEL_SELECT_FIELDS':{
            'LP2_20': 1,
            'amp': 1,
            'timeoffsets': 2            
        },
        'DATA_PATH': '/ceph/cms/store/user/azecchin/DESY_ETROC_telescope_merged/ETROC_Telescope_data/',
        'FILENAME_REGEX': r'Run_(\d+)_(\d+)_(\d+).root'
    },
    'SPS': {
        'EVENT_FIELDS': EVENT_FIELDS,
        'CHANNEL_SELECT_FIELDS':{
            'LP2_20': 1,
            'amp': 1,
            'timeoffsets': 2            
        },
        'DATA_PATH': '/ceph/cms/store/user/iareed/ETL/SPS_May24',
        'FILENAME_REGEX': r'run_(\d+)_rb0.root'
    },
    'SPS_FAST': { #for 20Gs SPS data of may
        'EVENT_FIELDS': EVENT_FIELDS,
        'CHANNEL_SELECT_FIELDS':{
            'timeoffsets': 2            
        },
        'DATA_PATH': '/ceph/cms/store/user/iareed/ETL/SPS_May24',
        'FILENAME_REGEX': r'run_(\d+)_rb0.root'
    },
    'SPS_Oct': {
        'EVENT_FIELDS': EVENT_FIELDS,
        'CHANNEL_SELECT_FIELDS':{
            'LP2_5': 0,
            'LP2_10': 0,
            'LP2_20': 0,
            'LP2_40': 0,
            'LP2_80': 0,
            'amp': 0,
            'timeoffsets': 1            
        },
        'DATA_PATH': '/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/ScopeData/LecroyMerged/',
        'FILENAME_REGEX': r'run_(\d+)_rb0.root'
    },
    'FNAL': {
        'EVENT_FIELDS': EVENT_FIELDS,
        'CHANNEL_SELECT_FIELDS':{
            'LP2_20': 7,
            'amp': 6,
            'timeoffsets': 7            
        },
        'DATA_PATH': '/ceph/cms/store/user/iareed/ETL/FNAL_May24',
        'FILENAME_REGEX': r'run_(\d+).root'
    }
}

#for user friendlyness
RUN_CONFIG_REQUIRED_KEYS = ["start", "stop", "module", "bias", "offset", "energy", "power", "file_from"]
RUN_CONFIG_OPTIONAL_KEYS = ['setup']
TW_BINNING_CHOICES = ['unbinned','binned','twodbinned']
if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    #---------------------Required Args----------------------#
    argParser.add_argument('--run_config', action='store', help='Path of json file with desired run configs to analyze')
    argParser.add_argument('--tag', action='store', help='Tag to distinguish different runs of the same hardware configs')
    #---------------------Overwrite Configs------------------#
    argParser.add_argument('--files_like', action='store', choices=list(TB_CONFIGS.keys()), help=f'Select a valid test beam configuration')
    argParser.add_argument('--data_dir', action='store', help='path to directory with all the run files')
    argParser.add_argument('--replace_old', action='store_true', help='replace files in exsisting output directories')
    #---------------------Paramaters(Cuts)-------------------#
    argParser.add_argument('--add_tracks', action='store_true', help='Add tracks to presel parquet file')
    argParser.add_argument('--lp2', action='store', default=20, choices=['5', '10', '15', '20', '25', '30', '35', '40', '50', '60', '70', '80'], help="Threshold to use for trigger pulse")
    argParser.add_argument('--scope_low', action='store', default=0, type = float, help='Low scope cut')
    argParser.add_argument('--scope_high', action='store', default=2000, type = float, help='High scope cut')
    argParser.add_argument('--toa_low', action='store', default=20, type = int, help='Low toa cut')
    argParser.add_argument('--toa_high', action='store', default=600, type = int, help='High toa cut')
    argParser.add_argument('--tot_low', action='store', default=50, type = int, help='Tot low cut')
    argParser.add_argument('--tot_high', action='store', default=200, type = int, help='tot high cut')
    #---------------------Paramaters(Timewalk)---------------#
    argParser.add_argument('--model', action='store', default='cubicLM', help='Model to use for TW Fits')
    argParser.add_argument('--twBinning', action='store', default='unbinned', choices=TW_BINNING_CHOICES, help="Choice of binning for time walk corrections")
    #---------------------Extra / Special--------------------#
    argParser.add_argument('--save_raw', action='store_true', help="Save parquet files of all events")
    argParser.add_argument('--root_dumper', action='store_true', help="root files from root dumper, no clock, trigger, then no dt, dt corrected")
    argParser.add_argument('--config_sel', action='store', type=int, nargs='*', default=[], help='Configs to run on by index in config file')
    argParser.add_argument('--custom_regex', action='store', help="pass this argument in instead of files_like to pass in your own regex matching")
    args = argParser.parse_args()

    if args.twBinning not in TW_BINNING_CHOICES: raise KeyError(f"{args.twBinnig} not a valid option, please use one of {TW_BINNING_CHOICES}")
    if not args.run_config: raise ValueError("Please give the path to the particular run configuration being used. Structure of the run configurations can be found in the github repository. https://gitlab.cern.ch/cms-etl-electronics/tbanalyisis look in the run_configs folder")

    #Get run config, contains board/module, run start and stop, energy, etc...
    with open(args.run_config, "rb") as f:
        run_configs = np.array(json.load(f))
    if len(args.config_sel) > 0:
        run_configs = run_configs[args.config_sel]

    output_paths_for_log = []
    for config in run_configs:
        whole_script_start = datetime.now()
        print(f"----------     SELECTED RUN CONFIGURATION = {config}     ----------")
        if 'board' in config:
            config['module'] = config.pop('board')
        if not set(RUN_CONFIG_REQUIRED_KEYS).issubset(config.keys()):
            print("You did not have the required config keys in the config")
            print(f"The required keys: {RUN_CONFIG_REQUIRED_KEYS}")
            print(f"Optional keys: {RUN_CONFIG_OPTIONAL_KEYS}")
            print('Aborting...')
            exit()
        if args.files_like is not None:
            file_style = args.files_like
        else:
            file_style = config['file_from']
            if file_style not in TB_CONFIGS:
                print(f"ABORTING! Your selected test beam config is not supported, please open this file and add it to the config, valid configurations: {TB_CONFIGS.keys()}")
                exit()       
        if args.data_dir is not None:
            data_path = args.data_dir
        else:
            data_path = TB_CONFIGS[file_style]['DATA_PATH'] 
        if args.custom_regex:
            reg_express = args.custom_regex
        else:
            reg_express = TB_CONFIGS[file_style]['FILENAME_REGEX']
        # if args.lp2 != 20:
        #     raise NotImplementedError("Different LP2's not implemented in code.")
        
        #-----------GET RUN FILES-----------#
        run_files = au.get_run_files(data_path, int(config['start']), int(config['stop']), reg_express, verbose=True)
        if len(run_files) == 0:
            print(f"Could not find any run files in {data_path} with {reg_express}, skipping...")
            continue
        #------------------------------------#
        #--------- detect setup / telescope or single module -------------#
        print("Checking if setup is mulit board / telescope setup...")
        is_telescope_setup = False
        if 'setup' in config and '-' in config['setup']:
            is_telescope_setup = True
            print(f'---------------- DETECTED MULTI BOARD / TELESCOPE SETUP WITH {config["module"]} SELECTED ------------------')
        else:
            print('This run was detected to not be telescope setup')
        #-------- build output data path name --------#
        print(f"Analyzing {len(run_files)} files from {data_path}")
        output_data_dir = au.convert_dict_to_str(config)
        if not args.root_dumper:
            output_data_dir += f'_LP2_{args.lp2}'
        output_data_dir += f'_{args.model}_{args.twBinning}'
        if args.add_tracks:
            output_data_dir += '_add_tracks'
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        output_data_path =  os.path.join(curr_dir, 'output_analysis_data', args.tag, output_data_dir)
        if os.path.exists(output_data_path):
            if args.replace_old:
                print(f"Replacing runs in {output_data_path}")
            else:
                print(f"Path {output_data_path} already exists, aborting")
                quit()
        ensure_dir_exists(output_data_path)
        #--------------------------------------------------------------------------------------------#
        cut_eff = {
            'total':0,
            'scope':0,
            'toa_code_low':0,
            'toa_code_high':0,
            'tot_code_low':0,
            'tot_code_high':0,
            'cal_code':0,
            'tracks':0
        }

        # Selections
        sel_scope_low = float(args.scope_low) 
        sel_scope_high = float(args.scope_high)
        sel_toa_low = float(args.toa_low)
        sel_toa_high = float(args.toa_high)
        sel_tot_low = float(args.tot_low)
        sel_tot_high = float(args.tot_high)
        sel_chi2_high = 50

        thresholds = {
            'scope_low':sel_scope_low,
            'scope_high':sel_scope_high,
            'toa_code_low':sel_toa_low,
            'toa_code_high':sel_toa_high,
            'tot_code_low':sel_tot_low,
            'tot_code_high':sel_tot_high
        }

        tb_config = TB_CONFIGS[file_style]
        all_fields = tb_config['EVENT_FIELDS'] + list(tb_config['CHANNEL_SELECT_FIELDS'].keys())
        #-------------------------------------------------------------CALCULATING CAL MODE-------------------------------------------------------------------#
        start_matrix_time=datetime.now()
        print(f"STARTING CAL MODE CALCULATION AND SAVING")
        cal_vals = np.empty((16,16),dtype=object)
        amp_vals = []
        hit_matrix = np.zeros((16,16))
        chunk_counter = 0
        for chunk in au.run_chunker(run_files):
            chunk_counter += 1
            events = uproot.concatenate(chunk, all_fields)
            events = au.module_select_filter(events, tb_config, is_telescope_setup, int(config['module']), chunk_counter)
            amp_vals += ak.to_list(ak.flatten(events.amp))

            scope_sel = ((events.amp > sel_scope_low) & (events.amp < sel_scope_high)) if not args.root_dumper else np.ones_like(events.event, dtype=bool)
            for row in range(16):
                for col in range(16):
                    pix_sel = (events.row==row)&(events.col==col)
                    selected_cals = ak.flatten(events.cal_code[pix_sel & scope_sel])
                    cal_vals[row][col] = np.append(cal_vals[row][col], selected_cals) if cal_vals[row, col] is not None else selected_cals
                    hit_matrix[row][col] += len(ak.flatten(events.cal_code[pix_sel]))

        amp_mode = stats.mode(ak.to_numpy(ak.Array(amp_vals)))[0]
        # Find the cal code mode now that all the values were found
        cal_mode = np.zeros((16,16))
        for row in range(16):
            for col in range(16):
                if len(cal_vals[row][col]) != 0:
                    cal_mode[row][col] = stats.mode(ak.to_numpy(cal_vals[row][col]))[0]
                else:
                    cal_mode[row][col] = -999

        save_data(cal_mode, output_data_path, 'cal_mode', make_json=True)
        save_data(hit_matrix, output_data_path, 'hit_map', make_json=True)
        print(f"CAL MODE CALCULATION TOOK {au.elapsed_time(start_matrix_time)}")
        #-------------------------------------------------------------EVENT CALCULATIONS---------------------------------------------------------------------#
        start_hist_filling=datetime.now()
        print(f"FLLING HISTS AND SELECTING EVENTS")

        fullCollection = ParquetManager(output_data_path, 'fullCollection') if args.save_raw else None
        fullPresel = ParquetManager(output_data_path, 'fullPresel')
        chunk_counter = 0
        for chunk in au.run_chunker(run_files):
            events = uproot.concatenate(chunk, all_fields)
            with au.HiddenPrints():
                events = au.module_select_filter(events, tb_config, is_telescope_setup, int(config['module']), chunk_counter)
            if not args.root_dumper:
                events['trigger'] = events[f"LP2_{args.lp2}"] * 10**9
                #if file_style == 'desy' or file_style == 'desy_merged' or file_style=='SPS' or file_style=="SPS_Oct":
                #    time = events['trigger'][((scope_sel)&(events.nhits>0))] #TODO: OUTPUT TRIGGER HISTOGRAM HERE
                #else:
                #    time = events['trigger'][ak.flatten(((scope_sel)&(events.nhits>0)))]

            #apply mask on whole data
            data_sel = au.data_mask(
                events,
                cut_eff,
                scope_low=sel_scope_low,
                scope_high=sel_scope_high,
                toa_low=sel_toa_low,
                toa_high=sel_toa_high,
                tot_low=sel_tot_low,
                tot_high=sel_tot_high,
                root_dumper=args.root_dumper
            )

            #apply selections
            presel = ak.zeros_like(events.tot_code)
            cal_array = ak.zeros_like(events.row)
            #amp_sel = abs(events.amp - amp_mode) <= 30

            #Calculate cal code efficiency
            for row in range(16):
                for col in range(16):
                    pix_sel = ((events.row==row)&(events.col==col))
                    cal_array = ak.where(pix_sel, cal_mode[row][col], cal_array) # think about cal_array = cal_mode[events.row, events.col]                    
                    cal_sel = ((events.cal_code<(cal_mode[row][col]+2)) & (events.cal_code>(cal_mode[row][col]-2))) 
                    cut_eff['cal_code'] += int(ak.count_nonzero(pix_sel & data_sel & cal_sel)) 

            cal_sel = abs(events.cal_code - cal_array) <= 2
            presel = data_sel & cal_sel #& amp_sel

            time_bin = 3.125 / cal_array
            toa = 12.5 - time_bin * events.toa_code
            tot = ((2*events.tot_code - np.floor(events.tot_code/32))*time_bin)
            if not args.root_dumper:
                #dt = toa - (trig - (clk + toff) + fnaloff) = toa - trig + clK + TOFF - FNALoff
                dt = toa - events.trigger + events.Clock + events.timeoffsets*10**9
                if file_style=="fnal":# or file_style=='sps_oct':
                    events['dt'] -= au.fnaloffset(events)
                events['dt'] = dt

            if fullCollection is not None:
                if file_style == 'SPS_Oct':
                    #manually set chipid for sps OCTOBER
                    # IF GETTING MERGE ISSUE IT IS BECAUSE SOME RUNS DO NOT HAVE CHIPID!!
                    events['chipid'] = ak.zeros_like(events.row) + 110<<2
                fullCollection.add_events(events)
            #make new awkward array: define it top level
            events_presel_chunk = ak.zip({
                'dt':       dt[presel],
                'toa':      toa[presel],
                'tot':      tot[presel],
                'row':      events.row[presel],
                'col':      events.col[presel],
                'toa_code': events.toa_code[presel],
                'tot_code': events.tot_code[presel],
                'cal_code': events.cal_code[presel],
                'clock': events.Clock[presel],
                'amp': events.amp[presel]
                #'chipid': events.chipid[presel]
            })
            if args.add_tracks:
                events_presel_chunk['x'] = events.x
                events_presel_chunk['y'] = events.y
                events_presel_chunk['chi2'] = events.chi2

            fullPresel.add_events(events_presel_chunk)
            #..................................................................................
        if fullCollection is not None:
            print("saving full collection of events")
            fullCollection.save()

        fullPresel.save()
        with open(f'{output_data_path}/cut_eff.json','w') as f_out:
            json.dump(cut_eff, f_out)
        with open(f'{output_data_path}/thresholds.json','w') as f_out:
            json.dump(thresholds, f_out)

        print(f"finished hist filling, took = {au.elapsed_time(start_hist_filling)}")
        #-------------------------------------------------------------timewalK CALCULATIONS---------------------------------------------------------------------#
        if not args.root_dumper:
            row_axis      = hist.axis.Integer(0, 16, name='row', label='row')
            col_axis      = hist.axis.Integer(0, 16, name='col', label='row')
            dt_axis       = hist.axis.Regular(500, 5, 18, name="dt", label="time")
            dt_corr_axis  = hist.axis.Regular(77,-1,1, name="dt", label="time")
            tot_prof_axis = hist.axis.Regular(10, 3.5, 6.0, name="tot", label="tot")
            dt_hist            = hist.Hist(dt_axis, row_axis, col_axis)
            dt_corr_hist       = hist.Hist(dt_corr_axis, row_axis, col_axis)
            dt_tot_hist        = hist.Hist(dt_axis, tot_prof_axis, row_axis, col_axis)
            dt_tot_raw_data = {a:{b:{'dt':[], 'tot':[]} for b in range(16)} for a in range(16)}

            events_presel = fullPresel.load()
            
            for row in range(16):
                for col in range(16):
                    pix_sel = ((events_presel.row==row) & (events_presel.col==col))
                    dt_flat = ak.flatten(events_presel['dt'][pix_sel])
                    tot_flat = ak.flatten(events_presel['tot'][pix_sel])
                    dt_tot_raw_data[row][col]['dt'] += dt_flat.to_list()
                    dt_tot_raw_data[row][col]['tot'] += tot_flat.to_list()
                    dt_tot_hist.fill(row=row, col=col, dt=dt_flat, tot=tot_flat)
                    dt_hist.fill(row=row, col=col, dt=dt_flat)

            start_timewalk_time=datetime.now()
            print(f"STARTING TIMEWALK CALCULATION")

            tw_corrections = [np.zeros((16,16)), np.zeros((16,16)), np.zeros((16,16)), np.zeros((16,16))]

            res = np.zeros([16, 16])
            res_corr = np.zeros([16, 16])

            dt_corr_builder = ak.flatten(ak.copy(events_presel.dt)).to_numpy()
            for row in range(16):
                for col in range(16):
                    if args.twBinning == 'unbinned':
                        tw_corrected = tw.calc_timewalk_corrections_unbinned(dt_tot_raw_data, row, col, args.model)
                    elif args.twBinning == 'binned':
                        tw_corrected = tw.calc_timewalk_corrections(dt_tot_hist, row, col, args.model)
                    elif args.twBinning == 'twodbinned':
                        tw_corrected = tw.calc_timewalk_corrections_2dbinned(dt_tot_hist, row, col, args.model)
                        
                    tw_corrections[0][row][col] = tw_corrected[0]
                    tw_corrections[1][row][col] = tw_corrected[1]
                    tw_corrections[2][row][col] = tw_corrected[2]
                    tw_corrections[3][row][col] = tw_corrected[3]
                    if tw_corrected[0] < -100:
                        dt_corr = ak.ones_like(dt_tot_raw_data[row][col]['tot'])*-999
                    else:
                        dt = ak.Array(dt_tot_raw_data[row][col]['dt'])
                        tot = ak.Array(dt_tot_raw_data[row][col]['tot'])
                        dt_corr = dt - tw.predict(args.model, tot, tw_corrected)

                    dt_corr_hist.fill(row=row, col=col, dt=dt_corr)

                    #Calculate Resolution
                    if fit_dt_outputs:= au.fit_gauss(dt_hist[{'row':row, 'col':col}]):
                        popt = fit_dt_outputs[0]
                        res[row,col] = popt[2]*1000 #ns to ps conversion :)
                    else:
                        res[row,col] = 0
                    #Calculate TW Corrected Resolution
                    if fit_dt_corr_outputs:= au.fit_gauss(dt_corr_hist[{'row':row, 'col':col}]):
                        popt = fit_dt_corr_outputs[0]
                        res_corr[row,col] = popt[2]*1000
                    else:
                        res_corr[row,col] = 0

                    pix_sel = ak.flatten((events_presel.row == row) & (events_presel.col == col)).to_numpy()
                    dt_corr_builder[pix_sel] = dt_corr.to_numpy()

            events_presel['dt_corr'] = ak.unflatten(ak.from_numpy(dt_corr_builder), ak.num(events_presel.dt))
            fullPresel.overwrite_with(events_presel)

            save_data(tw_corrections, output_data_path, 'tw_corrections', make_json=True)
            save_data(res, output_data_path,'res_heatmap', make_json=True)
            save_data(res_corr, output_data_path,'res_corrected_heatmap', make_json=True)
            improvement = np.where(
                (res_corr != -1)&(res != -1),
                res_corr - res,
                np.ones_like(res_corr)*-999
            )
            save_data(improvement, output_data_path, 'res_improvement_heatmap', make_json=True)
            output_paths_for_log.append(output_data_path)
        print("----------------------------------------------")
        print(f"TOTAL SCRIPT TOOK {au.elapsed_time(whole_script_start)}")
        print(f"OUTPUT PATH:")
        print(f"{output_data_path}")
