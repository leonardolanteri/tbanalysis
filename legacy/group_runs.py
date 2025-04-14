import os
import json
import argparse
import pandas as pd
from datetime import datetime

from make_plots import var_reader

def convert_to_datetime(log_time):
    if log_time:
        if file_from == 'DESY':
            day = int(log_time[5:7])
            month = 3 #log_time[8:12]
            year = int(log_time[13:17])
            hour = int(log_time[18:20])
            minute = int(log_time[21:23])
            second = int(log_time[24:26])
            return datetime(year, month, day, hour=hour, minute=minute, second=second)
        else:
            return datetime.strptime(log_time, "%a %d %b %Y %I:%M:%S %p %Z")

def config_from_row(start, stop, row, board, bias, offset, energy, power, file_from):
    configs = []
    base_config = {
        'start': str(start),
        'stop': str(stop),
        'setup': str(board),
        'offset': float(offset),
        'energy': float(energy),
        'power': str(power),
        'file_from': str(file_from)
    }
    if '-' in board:
        for mod, bias in zip(board.split('-'), bias.split('-')):
            base_config['module'] = mod
            base_config['bias'] = bias
            configs.append(base_config)
    else:
        base_config['module'] = board
        base_config['bias'] = bias
        configs.append(base_config)

    return configs

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--run_log', action='store', help='run log file')
    args = argParser.parse_args()

file_from = var_reader(args.run_log, 'run_log')
run_log = f'{os.environ["HOME"]}/tbanalysis/logs/{args.run_log}'
run_df = pd.read_csv(run_log, delimiter=',')

run_df['time'] = run_df['time'].apply(convert_to_datetime)
run_df['run_group'] = (pd.to_numeric(run_df['run_number']).diff() != 1).cumsum()
groupby_config = run_df.groupby(['board_number', 'bias_V', 'offset', 'energy', 'power_mode', 'run_group'])

selected_groups = []
for name, group_df in groupby_config:
    board, bias, offset, energy, power, rg = name
    if energy == 0:
        print(energy)
        continue

    np_df = group_df.to_numpy()
    start_row = np_df[0]
    end_row = np_df[-1]
    for i, row in enumerate(np_df):
        curr_row_num = row[0]
        curr_time = row[7]
        if i == len(np_df)-1: #last row
            selected_groups += config_from_row(start_row[0], curr_row_num, row, board, bias, offset, energy, power, file_from)
            break
        next_row = np_df[i+1]
        next_time = next_row[7]
        if (next_time - curr_time) > pd.Timedelta(minutes=2):
            selected_groups += config_from_row(start_row[0], curr_row_num, row, board, bias, offset, energy, power, file_from)
            start_row = next_row

out_dir = f'{os.environ["HOME"]}/tbanalysis/run_configs'
with open(f'{out_dir}/{file_from}_all_run_configs.json', 'w') as f:
    json.dump(selected_groups, f, indent=4)

sep_run_configs = {
    'multi_runs': [],
    'single_runs': []
}
for run_config in selected_groups:
    if run_config['start'] == run_config['stop']:
        sep_run_configs['single_runs'].append(run_config)
    else:
        sep_run_configs['multi_runs'].append(run_config)

with open(f'{out_dir}/{file_from}_single_run_configs.json', 'w') as f:
    json.dump(sep_run_configs['single_runs'], f, indent=4)

with open(f'{out_dir}/{file_from}_multi_run_configs.json', 'w') as f:
    json.dump(sep_run_configs['multi_runs'], f, indent=4)
