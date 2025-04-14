# Analysis Usage

## ENVIRONMENT CONFIGURATION
`conda env create --file=environment.yml`

### Configurations example
Here is how you choose what root files will be loaded. The other information is for book keeping. Except the `file_from` tells the program also what directory to point too as well as some other information see `do_analysis.py` TB_CONFIGS dictionary (Not a super smooth way of doing things but it works.)

Start and Stop are the run numbers. 
```
[
    {
        "start": "4000",
        "stop": "4199",
        "board": "37",
        "bias": "340",
        "offset": 15.0,
        "energy": 5.0,
        "power": "i1",
        "file_from": "DESY"
    },
    {
        "start": "5707",
        "stop": "5714",
        "board": "37-36-38",
        "bias": "290-250-275",
        "offset": 20.0,
        "energy": 5.0,
        "power": "i1",
        "file_from": "SPS"
    }
]
```

## RUNNING ANALYSIS
Minimal analysis routine:

`python do_analysis.py --run_config  run_configs/<desired_config.json> --tag my_tag`

### DESY1 Runs
- `all_run_configs.json`: all the run configurations
- `multi_run_configs.json`: configurations with more than one run (likely the most useful)
- `single_run_configs.json`: configurations with just one run

### SPS Runs
- `SPS_multi_run_configs.json`: configurations with more than one run
- `SPS_v0b_36.json`: configurations for module 36 (likely the most useful)

### FNAL Runs
- `FNAL.json`: all configurations
- `FNAL_useful.json`: configurations that took good data

### April 2025 DESY
- No complete run logs have been completed yet

### Data Paths
#### On the test stand PC
- `/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/ScopeData/LecroyMerged/DESY_March24`
- `/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/ScopeData/LecroyMerged/`: SPS data until fixes done
- FNAL is not on the test stand
- `/media/etl/Storage/DESY_April_2025`

#### CERNBox
- `https://cernbox.cern.ch/s/qlFUVAAs1YzQdfd`

#### On the UAF
- `/ceph/cms/store/user/iareed/ETL/Desy_Mar24`
- `/ceph/cms/store/user/iareed/ETL/SPS_May24`
- `/ceph/cms/store/user/iareed/ETL/FNAL_May24`

#### On LPC
- `/eos/uscms/store/group/cmstestbeam/ETL_DESY_March_2024/`
- `/eos/uscms/store/group/cmstestbeam/ETL_SPS_May_2024/`
- `/eos/uscms/store/group/cmstestbeam/2024_05_FNAL_ETL/`

## Analyzing
After the analysis is complete there will be a directory in `output_analysis_data/your_tag` with the saved parquets and other data. 

Then you can open these parquets (`fullCollection.parquet` has not quality cuts applied while `fullPresel.parquet` has quality cuts applied and timewalk information). 

### Optional 
Alternatively, I have a class that in `utils/plotting` called TBplot that I use to load all the data quickly make plots. Here is how you can make a minimal plotting routine:
```
from utils.plotting import TBplot
PARQUET_DIR = "output_analysis_data/your_tag/start_123_stop_1234_.../ # this dir should contain the parquets, and jsons outputted from the analysis code
run = TBplot(PARQUET_DIR, load_full_collection=False)

# you can change the binning from the default kind
run.hist_bins['toa_code'] = (75,0,800)

run.heatmap('toa_code', 'tot_code')
run.heatmap('dt', 'toa')

run.hit_map()
run.timewalk_scatter(pix=(8,8))
run.histo1D('dt', pix=(8,8), do_fit=True)
run.res_corr_map(cmax=100)
run.res_map()
run.resolution_shape_comparison()
```

### This Routine should work but hasn't been tested in awhile
`make_plots.py` constains the controls for making plots for a given collection of runs. It makes the majority on the fly from a saved parquet dataframe made by the analysis. Additional plots can be generated with the standard shells of 1D hist, 2D heatmap, 2D scatterplot based solely on the name of the variable and the desired binning

Minimal plotting routine:
`python make_plots.py --tag <name_of_analysis_tag>`

Plots generated are grouped by the following values
- module number
- beam energy
- power mode
- offset

Additionally, single pixel plots can be produced for either the whole module or specified pixels

## MISC
Please clear the output inside jupyter notebooks before pushing commits to keep down the file size. This can be done with
`jupyter nbconvert --clear-output --inplace <path/to/notebook>`

or add this function to your `.bashrc` for convenience:
```
function jclean () {
    FILE="$1"
    jupyter nbconvert --clear-output --inplace $FILE
}
```
