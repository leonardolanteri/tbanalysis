import os
from shutil import copy
import argparse
import numpy as np
import awkward as ak
from itertools import product
import utils.plotting as pu
from utils.file_handler import ensure_dir_exists

STD_BINS = {
    #bins, start, stop
    'dt':       (100,11,15),
    'dt_corr':  (100,-1,1),
    'tot_code': (75,0,400),
    'tot':      (50,0,10),
    'toa_code': (75,0,800),
    'toa':      (50,0,10),
}

MODULE_NUMS = [209]#36,37,38,40]
BEAM_ENERGIES = ['3.0']#'3.0','5.0',
POWER_MODES = ['i1']#,'i4']
PIXELS = []#[(9,8)] # leave empty to run all PIXELS otherwise; (row, col) 

#------------------------------------------------------------------------------------------#

def get_biases(directory: str, module: int, beam_energy: str, power_mode: str) -> dict:
    """
    Expects directory of all the output analysis directories. Example,
    output_analysis_data/
       |___tag/
             |___start_5050_stop_5199... (output analysis directory)
             |___start_5200_stop... (output analysis directory)
    """
    bias_group_paths = {}
    bias_groups = {}
    names = os.listdir(directory)
    for name in names:
        module_num = pu.var_reader(name, 'module')
        if module_num != str(module):
            continue
        offset = pu.var_reader(name, 'offset')
        bias = pu.var_reader(name, 'bias')
        if '-' in bias:
            continue
        energy = pu.var_reader(name, 'energy')
        if energy != beam_energy:
            continue
        power = pu.var_reader(name, 'power')
        if power != power_mode:
            continue
        total_path = os.path.join(directory,name)
        if offset in bias_groups:
            total_path = os.path.join(directory,name)
            bias_group_paths[offset].append(total_path)
            bias_groups[offset].append(bias)
        else:
            bias_group_paths[offset] = [total_path]
            bias_groups[offset] = [int(bias)]
    return bias_group_paths

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--tag', action='store', help='tag of the analysis to plot')
    argParser.add_argument('--output', action='store', default=f'{os.environ["HOME"]}/public_html/ETL/', help='location of the output plots')
    argParser.add_argument('--postCut', action='store', default=False, help='toggle pre/post cut input file')
    args = argParser.parse_args()

    config_groups_dir = f'{os.environ["HOME"]}/Test_Stand/tbanalysis/output_analysis_data/{args.tag}/'
    output_dir = f'{args.output}{args.tag}/'
    print(output_dir)
    ensure_dir_exists(output_dir)
    if args.output == argParser.get_default('output'):
        copy('index.php', output_dir)

    if not PIXELS:
        PIXELS = list(product(list(range(16)),list(range(16))))

    for module_num, beam_energy, power_mode in product(MODULE_NUMS, BEAM_ENERGIES, POWER_MODES):
        bias_groups_paths = get_biases(config_groups_dir, module_num, beam_energy, power_mode)
        if len(bias_groups_paths) == 0:
            print(f'No files for Module: {module_num}, energy: {beam_energy}, power mode: {power_mode}')
            continue
        for offset in bias_groups_paths:
            for path in bias_groups_paths[offset]:
                print(f"Generating plots for: {path}")
                #initialize test beam plotting class
                tb_plot = pu.TBplot(path, load_full_collection=(not args.postCut))

                save_path = os.path.join(output_dir, tb_plot.run_config.as_path())
                ensure_dir_exists(save_path)
                copy('utils/index.php', save_path)
                tb_plot.heatmap('tot_code','toa_code', save_path=save_path)

                tb_plot.cut_eff_plot(save_path=save_path)
                tb_plot.hit_map(save_path=save_path)
                tb_plot.cal_mode_map(save_path=save_path)
                tb_plot.res_map(save_path=save_path)
                tb_plot.res_corr_map(save_path=save_path)
                
                tb_plot.histo1D('dt', save_path=save_path)
                if args.postCut:
                    tb_plot.histo1D('dt_corr', save_path=save_path)

                save_path = os.path.join(save_path, 'per_pixel')
                ensure_dir_exists(save_path)
                copy('utils/index.php', save_path)
                
                #for pix in PIXELS:
                #    tb_plot.histo1D('dt',pix=pix, save_path=save_path)
                #    tb_plot.histo1D('dt_corr',pix=pix, save_path=save_path)
                #    tb_plot.timewalk_scatter(pix, save_path)
                del tb_plot
