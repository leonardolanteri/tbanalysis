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
from utils.file_handler import ensure_dir_exists, save_json
import utils.analysis as au
from utils import clock_fit, mcp_fit
import utils.timewalk as tw
from multiprocessing import Pool
from dataclasses import asdict
import pdb

TB_CONFIGS = {
    'DESY': {
        'DATA_PATH': '/ceph/cms/store/user/iareed/ETL/DESY_Mar24',
        'FILENAME_REGEX': r'run_(\d+).root',
    },
    'DESY_MERGED': {
        'DATA_PATH': '/ceph/cms/store/user/azecchin/DESY_ETROC_telescope_merged/ETROC_Telescope_data/',
        'FILENAME_REGEX': r'Run_(\d+)_(\d+)_(\d+).root'
    },
    'SPS': {
        'DATA_PATH': '/ceph/cms/store/user/iareed/ETL/SPS_May24',
        'FILENAME_REGEX': r'run_(\d+)_rb0.root'
    },
    'SPS_FAST': { #for 20Gs SPS data of may
        'DATA_PATH': '/ceph/cms/store/user/iareed/ETL/SPS_May24',
        'FILENAME_REGEX': r'run_(\d+)_rb0.root'
    },
    'SPS_Oct': {
        'DATA_PATH': '/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/ScopeData/LecroyMerged/',
        'FILENAME_REGEX': r'run_(\d+)_rb0.root'
    },
    'FNAL': {
        'DATA_PATH': '/ceph/cms/store/user/iareed/ETL/FNAL_May24',
        'FILENAME_REGEX': r'run_(\d+).root'
    },
    'DESY_April': {
        'DATA_PATH': '/media/etl/Storage/DESY_April_2025/merged/',
        'FILENAME_REGEX': r'run_(\d+).root',
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
    argParser.add_argument('--multireadout', action='store_true', help='save the data from each chip separately')
    #---------------------Paramaters(Cuts)-------------------#
    argParser.add_argument('--lp2', action='store', default=20, choices=['5', '10', '15', '20', '25', '30', '35', '40', '50', '60', '70', '80'], help="Threshold to use for trigger pulse")
    argParser.add_argument('--scope_low', action='store', default=-0.5, type = float, help='Low scope cut')
    argParser.add_argument('--scope_high', action='store', default=-0.05, type = float, help='High scope cut')
    argParser.add_argument('--toa_low', action='store', default=20, type = int, help='Low toa cut')
    argParser.add_argument('--toa_high', action='store', default=600, type = int, help='High toa cut')
    argParser.add_argument('--tot_low', action='store', default=50, type = int, help='Tot low cut')
    argParser.add_argument('--tot_high', action='store', default=200, type = int, help='tot high cut')
    #---------------------Paramaters(Timewalk)---------------#
    argParser.add_argument('--model', action='store', default='cubicLM', help='Model to use for TW Fits')
    argParser.add_argument('--twBinning', action='store', default='unbinned', choices=TW_BINNING_CHOICES, help="Choice of binning for time walk corrections")
    #---------------------Extra / Special--------------------#
    argParser.add_argument('--save_raw', action='store_true', help="Save parquet files of all events")
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
        
        #-----------GET RUN FILES-----------#
        run_files = au.get_run_files(data_path, int(config['start']), int(config['stop']), reg_express, verbose=True)
        if len(run_files) == 0:
            print(f"Could not find any run files in {data_path} with {reg_express}, skipping...")
            continue
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
        output_data_dir += f'_{args.model}_{args.twBinning}'
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

        load_time=datetime.now()
        print(f"LOADING EVENTS")

        def load_events(chunk: list[str]):
            events = uproot.concatenate(chunk)            
            events = au.module_select_filter(events, is_telescope_setup, int(config['module']))  
            events['clock_timestamp'] = clock_fit.calc_clock(
                events.clock_seconds*1e9, events.clock_volts, 0.25, 0.8, 0.5
            )
            nanoseconds, scaled_volts = mcp_fit.MCPSignalScaler.normalize(
                ak.to_numpy(events['mcp_seconds']) * 1e9, 
                ak.to_numpy(events['mcp_volts']), 
                signal_saturation_level=-2.8
            )
            peak_times, events['mcp_amplitude'] = mcp_fit.MCPSignalScaler.calc_mcp_peaks(nanoseconds, ak.to_numpy(events['mcp_volts']))
            events['mcp_timestamp'] = mcp_fit.linear_interpolation(nanoseconds, scaled_volts, peak_times, threshold=0.4)

            del events['mcp_seconds']
            del events['mcp_volts']
            del events['clock_seconds']
            del events['clock_volts']
            return events
        
        with Pool(processes=5) as pool:
            results = list(pool.imap(load_events, run_files, chunksize=10))
            events = ak.concatenate(results)
        print(f"Loading Events took {au.elapsed_time(load_time)}")

        thresholds = au.Thresholds(
            mcp_amp_high  = float(args.scope_high),
            mcp_amp_low   = float(args.scope_low) ,
            toa_code_high = float(args.toa_high),
            toa_code_low  = float(args.toa_low),
            tot_code_high = float(args.tot_high),
            tot_code_low  = float(args.tot_low)
        )

        scope_sel = thresholds.mcp_amp_cut(events.mcp_amplitude)
        toa_sel = thresholds.toa_code_cut(events.toa_code)
        tot_sel = thresholds.tot_code_cut(events.tot_code)

        cut_eff = {
            'total': int(ak.count_nonzero(events.nhits>0)),
            'scope': int(ak.count_nonzero(scope_sel & (events.nhits>0))),
            'toa_code_low': int(ak.count_nonzero(scope_sel & (events.toa_code>thresholds.toa_code_low))),
            'toa_code_high':int(ak.count_nonzero(scope_sel & toa_sel)),
            'tot_code_low':int(ak.count_nonzero(scope_sel & toa_sel & (events.tot_code>thresholds.toa_code_low))),
            'tot_code_high':int(ak.count_nonzero(scope_sel & toa_sel & tot_sel)),
            'cal_code':0,
        }

        cal_mode = au.cal_mode(events, thresholds)
        cal_array = ak.zeros_like(events.row)
        for row in range(16):
            for col in range(16):
                pix_sel = ((events.row==row)&(events.col==col))
                cal_array = ak.where(pix_sel, cal_mode[row][col], cal_array) # think about cal_array = cal_mode[events.row, events.col]                    
                cal_sel = ((events.cal_code<(cal_mode[row][col]+2)) & (events.cal_code>(cal_mode[row][col]-2))) 
                cut_eff['cal_code'] += int(ak.count_nonzero(pix_sel & scope_sel & toa_sel & tot_sel & cal_sel)) 

        cal_sel = abs(events.cal_code - cal_array) <= 2
        #---------------------------------------------------#
        
        #------------------DT calculation ------------------#
        events['cal_mode'] = cal_array
        time_bin = 3.125 / events.cal_mode
        events['toa'] = 12.5 - time_bin * events.toa_code #ns
        events['tot'] = ((2*events.tot_code - np.floor(events.tot_code/32))*time_bin) #ns
        events['dt'] = events.toa - (events.mcp_timestamp - events.clock_timestamp) #ns
        if file_style=="fnal":# or file_style=='sps_oct':
            events.dt -= au.fnalOffset(events)      
        if file_style=="DESY_April":
            # presel = presel & (events.dt > 0)
            if au.module_select_filter and args.multireadout:
                all_chips = np.unique(ak.flatten(events.chipid))
                for chipid in all_chips:
                    pdb.set_trace()
                    presel = scope_sel & toa_sel & tot_sel & cal_sel & (events.chipid == chipid) & (events.dt > 0)
                    
                    events_presel = ak.zip({
                        'dt':       events.dt[presel],
                        'toa':      events.toa[presel],
                        'tot':      events.tot[presel],
                        'row':      events.row[presel],
                        'col':      events.col[presel],
                        'toa_code': events.toa_code[presel],
                        'tot_code': events.tot_code[presel],
                        'cal_code': events.cal_code[presel],
                        'cal_mode': events.cal_mode[presel]
                    })

                    #-------------------------timewalK CALCULATIONS-----------------------------#
                    row_axis      = hist.axis.Integer(0, 16, name='row', label='row')
                    col_axis      = hist.axis.Integer(0, 16, name='col', label='row')
                    dt_axis       = hist.axis.Regular(500, 5, 18, name="dt", label="time")
                    dt_corr_axis  = hist.axis.Regular(77,-1,1, name="dt", label="time")
                    tot_prof_axis = hist.axis.Regular(10, 3.5, 6.0, name="tot", label="tot")
                    dt_hist            = hist.Hist(dt_axis, row_axis, col_axis)
                    dt_corr_hist       = hist.Hist(dt_corr_axis, row_axis, col_axis)
                    dt_tot_hist        = hist.Hist(dt_axis, tot_prof_axis, row_axis, col_axis)
                    dt_tot_raw_data = {a:{b:{'dt':[], 'tot':[]} for b in range(16)} for a in range(16)}
                    
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
                    events_presel_chips["chip_{chipid}"] = events_presel
            
                    # ---------------------SAVE THE DATA-----------------------------#
        if args.save_raw:
            ak.to_parquet(events, os.path.join(output_data_path, 'fullCollection.parquet'))
        
        ak.to_parquet(events_presel_chips, os.path.join(output_data_path, 'fullPresel.parquet'))
        
        save_json(cal_mode, output_data_path, 'cal_mode')
        save_json(au.hit_map(events), output_data_path, 'hit_map')
        save_json(asdict(thresholds), output_data_path, 'thresholds.json')
        save_json(cut_eff, output_data_path, 'cut_eff.json')
        save_json(tw_corrections, output_data_path, 'tw_corrections')
        save_json(res, output_data_path,'res_heatmap')
        save_json(res_corr, output_data_path,'res_corrected_heatmap')
        improvement = np.where(
            (res_corr != -1)&(res != -1),
            res_corr - res,
            np.ones_like(res_corr)*-999
        )
        save_json(improvement, output_data_path, 'res_improvement_heatmap')
        print("----------------------------------------------")
        print(f"TOTAL SCRIPT TOOK {au.elapsed_time(whole_script_start)}")
        print(f"OUTPUT PATH:")
        print(f"{output_data_path}")
