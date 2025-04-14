import os
import json
from textwrap import dedent
from typing import Any
from dataclasses import dataclass, asdict
import numpy as np
import awkward as ak
from awkward.highlevel import Array as akArray
import matplotlib.pyplot as plt
import hist
from hist import Hist
import mplhep as hep
plt.style.use(hep.style.CMS)

import utils.timewalk as tw
import utils.analysis as au

COLORS = ['#3f90da','#ffa90e','#bd1f01','#94a4a2','#832db6','#a96b59','#e76300','#b9ac70','#717581','#92dadd']

@dataclass
class RunConfig:
    start:int
    stop: int
    module: int
    bias: float
    offset: float
    beam_energy: float
    power_mode: str

    def as_path(self) -> str:
        return f"module_{self.module}_bias_{self.bias}_offset_{self.offset}_energy_{self.beam_energy}_power_{self.power_mode}_runs_{self.start}_{self.stop}"
    def __repr__(self) -> str:
        return f"Module: {self.module} Bias: {self.bias} Offset: {self.offset} Energy: {self.beam_energy} Power: {self.power_mode} Runs: [{self.start}-{self.stop}]"

@dataclass
class ThresholdsOld:
    tot_code_high: float
    tot_code_low: float
    toa_code_high: float
    toa_code_low: float
    scope_low: float
    scope_high: float


def var_reader(path: str, var: str) -> str:
    return path.split(var+'_')[1].split('_')[0]

def pix_labeler(pix: tuple) -> str:
    """
    Creates label that is added to filenames for output files for a specific pixel
    """
    pix_label = ''
    if pix: pix_label = f"_r{pix[0]}_c{pix[1]}"
    return pix_label

def plot_saver(pix:tuple, pix_label, post_cut:bool, save_path: str, xVar:str, yVar=None, style=None) -> None:
    file_name = 'module'
    if pix: file_name = pix_label[1:]
    file_name += f'_{xVar}'
    if yVar: file_name += f'_{yVar}'
    if style: file_name += '_' + style
    if post_cut: file_name += '_post_cut'
    file_name += '.png'
    plt.savefig(os.path.join(save_path, file_name)) 

def figuration(width=9, height=6, font_size=14):
    """Decorator for initializing plotting function with standard figure parameters"""
    def wrapper(plot_func):
        def wrapped_plot(*args, **kwargs):
            # Set up the figure
            f = plt.figure()
            f.set_figwidth(width)
            f.set_figheight(height)
            font = {'size': font_size}
            plt.rc('font', **font)
            
            outputs = plot_func(*args, **kwargs)
            
            if kwargs.get('save_path') is not None:
                plt.clf()
            
            return outputs
        return wrapped_plot
    return wrapper
    
def plot_gauss_fit(h: Hist, label=None, color=None, fill_between = True) -> None:
    """
    Fits and plots gaussian to 1D Hist histogram
    """
    gaus = lambda x, N, mu, sigma: N*np.exp(-(x-mu)**2/(2.0*sigma**2))

    if fit_params:=au.fit_gauss(h):
        popt, pcov, perr, red_chi2 = fit_params
    else:
        return

    bin_centers = h.axes[0].centers
    label = dedent(f"""
        {label if label is not None else ''}
        {r"$\mu$"}: {popt[1]:.3f}{r"$\pm$"}{perr[1]:.3f}
        {r"$\sigma$"}: {abs(popt[2]):.3f}{r"$\pm$"}{perr[2]:.3f}
    """).strip()
    #{r"$\chi_{red}^{2}$"}: {red_chi2:3f}

    x_values = np.linspace(bin_centers.min(), bin_centers.max(), len(bin_centers)*10)
    y_values = gaus(x_values, *popt)
    plt.plot(
        x_values,
        y_values,
        color=COLORS[2] if color is None else color,
        label=label
    )
    # plt.plot(
    #     bin_centers,
    #     gaus(bin_centers, *popt),
    #     color=COLORS[2] if color is None else color,
    #     label=label
    # )
    if fill_between:
        #https://github.com/scikit-hep/hist/blob/6fb3ecd07d1f9a4758cd5d5ccf89559ed572ca9a/src/hist/plot.py#L336
        if np.isfinite(pcov).all():
            n_samples = 100
            vopts = np.random.multivariate_normal(popt, pcov, n_samples)
            sampled_ydata = np.vstack([gaus(x_values, *vopt).T for vopt in vopts])
            model_uncert = np.nanstd(sampled_ydata, axis=0)
        else:
            model_uncert = np.zeros_like(h.values())

        #https://github.com/scikit-hep/hist/blob/6fb3ecd07d1f9a4758cd5d5ccf89559ed572ca9a/src/hist/plot.py#L377
        plt.fill_between(
            x_values,
            gaus(x_values, *popt) - model_uncert,
            gaus(x_values, *popt) + model_uncert,
            color='lightgrey',
        )

def buildLabel(params, model):
    a, b, c, d = params
    if 'linear' in model:
        return 'Linear Fit \n a: {:.2f} \n b: {:.2f}'.format(a, b)
    elif 'quad' in model:
        return 'Quadradic Fit\n a: {:.2f} \n b: {:.2f} \n c: {:.2f} '.format(a, b, c)
    elif 'cubic' in model:
        return 'Cubic Fit\n a: {:.2f} \n b: {:.2f} \n c: {:.2f} \n d: {:.2f} '.format(a, b, c, d)

def timewalk_heatmap(data, pix, tw_corr, save_path, model='cubicLM'):
    row, col = pix
    a, b, c, d = tw_corr[0][row][col], tw_corr[1][row][col], tw_corr[2][row][col], tw_corr[3][row][col]
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.set_title(f"Row {row} Col {col}")
    ax.set_ylabel(r"$\Delta T$")
    ax.set_xlabel("TOT")
    _, _, _, img = plt.hist2d(np.array(data['tot']), np.array(data['dt']), bins = 50)
    plt.colorbar(img, ax = ax)
    x = np.array(data['tot'])
    x.sort()
    y = tw.predict(model, x, [a, b, c, d])
    plt.plot(x, y, color = COLORS[2])
    plt.savefig(os.path.join(save_path, f'r{row}_c{col}_timewalk_heatmap.png'))
    plt.clf()


class TBplot:
    def __init__(self, data_dir: str, load_full_collection:bool=False):
        print(f"LOADING: {data_dir}")
        self.hist_bins = {
            #bins, start, stop
            'dt':       (500, 5, 25),
            'dt_corr':  (77,-1,1),
            'tot_code': (75,0,400),
            'tot':      (50,0,10),
            'toa_code': (75,0,400),
            'toa':      (50,0,10),
            'Clock':      (50,-25,25),
            'res_shape_comp': (40, 0, 200),
            'amp': (500,300,1400)
        }
        self.data_dir = data_dir
        self.run_config = RunConfig(
            start=int(var_reader(data_dir, 'start')),
            stop=int(var_reader(data_dir, 'stop')),
            module=int(var_reader(data_dir, 'module')),
            bias=float(var_reader(data_dir, 'bias')),
            offset=float(var_reader(data_dir, 'offset')),
            beam_energy=float(var_reader(data_dir, 'energy')),
            power_mode=var_reader(data_dir, 'power')
        )
        #---------LOAD ALL DATA-----------#
        with open(os.path.join(data_dir, 'cut_eff.json')) as f:
            self.cut_eff_data = json.load(f)
        with open(os.path.join(data_dir, 'thresholds.json')) as f:
            thresholds = json.load(f)
            self.thresholds = au.Thresholds(**thresholds)

    
        self.cal_mode_data     = self.load_json_data(os.path.join(data_dir, 'cal_mode.json'))
        self.hit_map_data      = self.load_json_data(os.path.join(data_dir, 'hit_map.json'))
        self.tw_corr_data      = self.load_json_data(os.path.join(data_dir, 'tw_corrections.json'))
        self.res_map_data      = self.load_json_data(os.path.join(data_dir, 'res_heatmap.json'))
        self.res_corr_map_data = self.load_json_data(os.path.join(data_dir, 'res_corrected_heatmap.json'))
        
        self.load_full_collection = load_full_collection
        if load_full_collection:
            self.events = ak.from_parquet(os.path.join(data_dir, 'fullCollection.parquet'))
        else:
            self.events = ak.from_parquet(os.path.join(data_dir, 'fullPresel.parquet'))
    
    def load_json_data(self, path:str) -> akArray:    
        with open(path) as f:
            return np.array(json.load(f))
    
    def set_clims(self, data: akArray, cmin:int = None, cmax:int = None, set_point:int=2) -> tuple:
        def auto_cmin(data:akArray, set_point:int=2) -> int:
            return ak.sort(ak.flatten(data),ascending=True)[set_point]
        def auto_cmax(data:akArray, set_point:int=2) -> int:
            return ak.sort(ak.flatten(data),ascending=True)[-set_point]

        cmin = auto_cmin(data, set_point=set_point) if cmin is None else cmin
        cmax = auto_cmax(data, set_point=set_point) if cmax is None else cmax
        return cmin, cmax

    @figuration(width=9,height=6,font_size=12)
    def cut_eff_plot(self, save_path:str=None):
        """Bar graph of how much data the cuts removed"""
        self.cut_eff_data.pop('tracks') if 'tracks' in self.cut_eff_data else self.cut_eff_data
        cut = list(self.cut_eff_data.keys())
        passing = list(self.cut_eff_data.values())
        plt.bar(cut, height=passing)
        try:
            final_eff = passing[-1]/passing[0]*100
        except:
            final_eff = 0.00
        plt.title(f"Efficiencies, Final Eff: {final_eff:.2f} \n {self.run_config}")
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'cut_efficiencies.png'))
    
    @figuration(width=9,height=9,font_size=14) #Keep it square!
    def manual_sensor_map(self, pixel_map, name: str, cmax:int=None, cmin:int=None, save_path:str=None, text_color='w'):
        cmin, cmax = self.set_clims(pixel_map, cmin=cmin, cmax=cmax)
        pixels_hist = Hist(
            #x axis = col
            hist.axis.Integer(0, 16, name="col", label="col", flow=False),
            #y axis = row
            hist.axis.Integer(0, 16, name="row", label="row", flow=False),
        )
        for (row, col), weight in np.ndenumerate(pixel_map):
            pixels_hist.fill(row=row, col=col, weight=weight)
            # x = col, y = row
            plt.text(col + 0.5, row + 0.5, f'{int(weight)}', ha='center', va='center', color=text_color , fontsize="xx-small")
        plt.title(f"{name} \n Summary {self.run_config}")
        hep.hist2dplot(pixels_hist, cmax=cmax, cmin=cmin)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'{name.replace(" ", "_")}_Map.png'))

    def hit_map(self, cmin:int=None, cmax:int=None, save_path:str=None):
        self.manual_sensor_map(self.hit_map_data,'Hit Map', cmin=cmin, cmax=cmax, save_path=save_path)

    def cal_mode_map(self, cmin:int=None, cmax:int=None, save_path:str=None):
        self.manual_sensor_map(self.cal_mode_data,'Cal Mode', cmin=cmin, cmax=cmax, save_path=save_path)
    
    def res_map(self, cmin:int=None, cmax:int=None, save_path:str=None):
        res_map_data = ak.where(self.res_map_data<0, ak.zeros_like(self.res_map_data), self.res_map_data)
        self.manual_sensor_map(res_map_data,'Resolution', cmin=cmin, cmax=cmax, save_path=save_path)

    def res_corr_map(self, cmin:int=None, cmax:int=None, save_path:str=None):
        res_corr_map_data = ak.where(self.res_corr_map_data<0, ak.zeros_like(self.res_corr_map_data), self.res_corr_map_data)
        self.manual_sensor_map(res_corr_map_data,'Resolution Corrected', cmin=cmin, cmax=cmax, save_path=save_path)

    @figuration(width=9,height=6,font_size=18)
    def resolution_shape_comparison(self, fit=True, save_path=None):
        """Plots all the resolutions for all pixels for timewalk corrected and uncorrected."""
        hide_noise = self.hit_map_data > 50#100
        res_axis = hist.axis.Regular(*self.hist_bins['res_shape_comp'], name="res", label="Resolution (ps)")
        res_hist = Hist(res_axis)
        res_corr_hist = Hist(res_axis, label="corrected", name="corrected")
        res_hist.fill(self.res_map_data[hide_noise].flatten())
        res_corr_hist.fill((self.res_corr_map_data[hide_noise].flatten()))

        hep.histplot(res_hist, color=COLORS[0], label="Uncorrected")
        hep.histplot(res_corr_hist, color=COLORS[1], label="Corrected")

        if fit:
            plot_gauss_fit(res_hist, label='Uncorrected Fit', color=COLORS[0])
            plot_gauss_fit(res_corr_hist, label='TW Corrected Fit', color=COLORS[1])

        plt.title(f"Resolution Distribution \n {self.run_config}")
        plt.ylabel("Count")
        plt.legend()
        if save_path is not None:
            plt.savefig(os.path.join(os.path.join(save_path, 'resolution_shape_comparison.png'))) 
    
    @figuration(width=9,height=6,font_size=18)
    def resolution_corrected_shape(self, fit=True, fill_between=True, save_path=None, extra_label=None):
        """Plots all the resolutions for all pixels for timewalk corrected and uncorrected."""
        hide_noise = self.hit_map_data > 0#100
        res_axis = hist.axis.Regular(*self.hist_bins['res_shape_comp'], name="res", label="Resolution (ps)")
        res_corr_hist = Hist(res_axis, label="corrected", name="corrected")
        res_corr_hist.fill(self.res_corr_map_data[hide_noise].flatten())
        hep.histplot(res_corr_hist, color=COLORS[1], label="Corrected", yerr=0)
        if fit:
            plot_gauss_fit(res_corr_hist, label='TW Corrected Fit', color=COLORS[0], fill_between=fill_between)
        extra_label = '' if extra_label is None else extra_label
        plt.title(f"Resolution Distribution at {extra_label}, npixels={sum(res_corr_hist.counts())} \n {self.run_config}")
        plt.ylabel("Count")
        plt.legend()
        if save_path is not None:
            plt.savefig(os.path.join(os.path.join(save_path, 'resolution_corr_shape.png'))) 

    @figuration(width=9,height=6,font_size=18)
    def histo1D(self, xVar:str, pix:tuple=None, xcuts:tuple=None, do_fit:bool=True):
        """Plots 1D histogram based off the field name "xVar" or the awkard event array. """
        h = self.get_hist(xVar,pix=pix)
        hep.histplot(h, color=COLORS[0])
        if do_fit:
            plot_gauss_fit(h, color=COLORS[2])
        plt.xlabel(xVar)
        plt.ylabel('counts')
        self.draw_thresh_vline(xVar, cuts=xcuts)
        pix_label = pix_labeler(pix)
        plt.title(f'{xVar}{pix_label} n: {sum(h.values())}\n{self.run_config} ')
        plt.legend()
    
    @figuration(width=9,height=9,font_size=18)
    def heatmap(self, xVar:str, yVar:str, pix:tuple=None, xcuts:tuple=None, ycuts:tuple=None):
        """
        2D histogram of any field selected by "xVar" and "yVar" in loaded awkward events array
        xcuts and ycuts are tuples of the high and low to override the thresholds
        """
        xVals = self.get_field_vals(xVar, pix=pix)
        yVals = self.get_field_vals(yVar, pix=pix)

        hist_2d = Hist(
            hist.axis.Regular(*self.hist_bins[xVar], name=xVar),
            hist.axis.Regular(*self.hist_bins[yVar], name=yVar),
        ).fill(xVals, yVals)

        hep.hist2dplot(hist_2d)

        plt.xlabel(xVar)

        self.draw_thresh_vline(xVar,cuts=xcuts)
        self.draw_thresh_hline(yVar,cuts=ycuts)

        pix_label = pix_labeler(pix)
        plt.title(f'{yVar} vs {xVar} {pix} n: {len(xVals)}:\n{self.run_config}')
        plt.legend(labelcolor='w')

    @figuration()
    def timewalk_scatter(self, pix:tuple):
        model='cubicLM'
        row, col = pix
        a, b, c, d = self.tw_corr_data[0][row][col], self.tw_corr_data[1][row][col], self.tw_corr_data[2][row][col], self.tw_corr_data[3][row][col]
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        y = self.get_field_vals('dt', pix=pix).to_numpy()
        x = self.get_field_vals('tot', pix=pix).to_numpy()
        m = np.median(x)
        s = np.std(x)
        idx = np.abs(x - m) < 2*s
        plt.scatter(x[idx], y[idx], label = f'Fitting Data: {ak.count_nonzero(idx)}', alpha = 0.5, color=COLORS[0])
        plt.scatter(x[~idx], y[~idx], label = f'Rejected Data: {ak.count_nonzero(~idx)}', alpha = 0.5, color=COLORS[1])
        x.sort()
        y = tw.predict(model, x, [a, b, c, d])
        plt.title(f"Row {row} Col {col} \n n: {len(x)}")
        plt.ylabel(r"$\Delta T$")
        plt.xlabel("TOT")
        plt.plot(x, y, color=COLORS[2], label=buildLabel([a, b, c, d], model) + '\n m = {:.2f} \n s = {:.2f}'.format(m,s))
        plt.legend()

    def get_field_vals(self, field:str, pix:tuple=None) -> akArray:
        """Retrieves a values from an awkward array by selecting the field."""
        data = self.events
        if pix is not None:
            field_data = data[field]
            data = ak.flatten(field_data[(data.row==pix[0]) & (data.col==pix[1])])
            return data#[field]
        else:
            return ak.flatten(data[field])
        
    def get_hist(self, xVar:str, pix:tuple=None) -> Hist:
        """Takes field (corresponding to field in the events awkward array) and fills the 1D histogram"""
        xVals = self.get_field_vals(xVar, pix=pix)
        hist_axis = hist.axis.Regular(*self.hist_bins[xVar], name=xVar)
        return Hist(hist_axis).fill(xVals)
    
    def update_thresholds(self, thresh_name:str, thresh_value: int):
        setattr(self.thresholds, thresh_name, thresh_value)

    def _draw_thresh_line(self, field:str, is_vertical:bool, cuts:tuple = None, ):
        if cuts is not None:
            self.update_thresholds(f"{field}_low", cuts[0])
            self.update_thresholds(f"{field}_high", cuts[1])

        for thresh_name, thresh in asdict(self.thresholds).items():
            if 'code' in field and field in thresh_name:
                if is_vertical:
                    plt.axvline(thresh, color=COLORS[2], label=f"{thresh_name}={thresh}")
                else:
                    plt.axhline(thresh, color=COLORS[2], label=f"{thresh_name}={thresh}")
    
    def draw_thresh_vline(self, field: str, cuts:tuple=None):
        self._draw_thresh_line(field, True, cuts=cuts)
    def draw_thresh_hline(self, field: str, cuts:tuple=None):
        self._draw_thresh_line(field, False, cuts=cuts)

    def beam_spot(self, percent, high=None, pixel_excludes:list[tuple]=None) -> np.ndarray:
        if high is None:
            high = self.hit_map_data.max()

        mask = (self.hit_map_data <= high) & (self.hit_map_data > 0)

        filtered_data = self.hit_map_data[mask]
        threshold = np.percentile(filtered_data, 100 - percent)

        mask &= (self.hit_map_data >= threshold)
        if pixel_excludes is not None:
            for pix in pixel_excludes:
                mask[pix[0], pix[1]] = False
        return mask

class TBplotRAW:
    def __init__(self, events:ak.Array, run_config: dict, thresholds:  au.Thresholds):
        self.hist_bins = {
            #bins, start, stop
            'dt':       (500, 5, 25),
            'dt_corr':  (77,-1,1),
            'tot_code': (75,0,600),
            'tot':      (50,0,10),
            'toa_code': (75,0,800),
            'toa':      (50,0,10),
            'clock_timestamp':      (50,-25,25),
            'res_shape_comp': (40, 0, 200),
            'mcp_timestamp': (500,300,1400),
            'cal_code': (40, 20, 300),
            'cal_mode': (30, 150, 220)
        }
        self.events = events
        self.run_config = RunConfig(**run_config)
        self.thresholds = thresholds

    def set_clims(self, data: akArray, cmin:int = None, cmax:int = None, set_point:int=2) -> tuple:
        def auto_cmin(data:akArray, set_point:int=2) -> int:
            return ak.sort(ak.flatten(data),ascending=True)[set_point]
        def auto_cmax(data:akArray, set_point:int=2) -> int:
            return ak.sort(ak.flatten(data),ascending=True)[-set_point]

        cmin = auto_cmin(data, set_point=set_point) if cmin is None else cmin
        cmax = auto_cmax(data, set_point=set_point) if cmax is None else cmax
        return cmin, cmax

    @figuration(width=9,height=9,font_size=14) #Keep it square!
    def manual_sensor_map(self, pixel_map, name: str, cmax:int=None, cmin:int=None, text_color='w'):
        cmin, cmax = self.set_clims(pixel_map, cmin=cmin, cmax=cmax)
        pixels_hist = Hist(
            #x axis = col
            hist.axis.Integer(0, 16, name="col", label="col", flow=False),
            #y axis = row
            hist.axis.Integer(0, 16, name="row", label="row", flow=False),
        )
        for (row, col), weight in np.ndenumerate(pixel_map):
            pixels_hist.fill(row=row, col=col, weight=weight)
            # x = col, y = row
            plt.text(col + 0.5, row + 0.5, f'{int(weight)}', ha='center', va='center', color=text_color , fontsize="xx-small")
        plt.title(f"{name} \n Summary {self.run_config}")
        hep.hist2dplot(pixels_hist, cmax=cmax, cmin=cmin)
  
    def hit_map(self, cmin:int=None, cmax:int=None, save_path:str=None):
        hits = self.calc_hit_map()
        self.manual_sensor_map(hits, f'Hit Map, Total {np.sum(hits)}', cmin=cmin, cmax=cmax)

    def cal_mode_map(self, cal_mode, cmin:int=None, cmax:int=None):
        self.manual_sensor_map(cal_mode,'Cal Mode', cmin=cmin, cmax=cmax)
    
    def resolution_map(self, res_map, cmin:int=None, cmax:int=None, save_path:str=None):
        res_map_data = ak.where(res_map<0, ak.zeros_like(res_map), res_map)
        self.manual_sensor_map(res_map_data,'Resolution Map', cmin=cmin, cmax=cmax)

    def calc_hit_map(self):
        hit_matrix = np.zeros((16,16))
        for row in range(16):
            for col in range(16):
                pix_sel = (self.events.row==row)&(self.events.col==col)
                hit_matrix[row][col] += len(ak.flatten(self.events.row[pix_sel]))
        return hit_matrix
    
    @figuration(width=9,height=6,font_size=18)
    def resolution_shape_comparison(self, res, res_corr, fit=True, save_path=None):
        """Plots all the resolutions for all pixels for timewalk corrected and uncorrected."""
        hide_noise = self.calc_hit_map() > 50#100
        res_axis = hist.axis.Regular(*self.hist_bins['res_shape_comp'], name="res", label="Resolution (ps)")
        res_hist = Hist(res_axis)
        res_corr_hist = Hist(res_axis, label="corrected", name="corrected")
        res_hist.fill(res[hide_noise].flatten())
        res_corr_hist.fill((res_corr[hide_noise].flatten()))

        hep.histplot(res_hist, color=COLORS[0], label="Uncorrected")
        hep.histplot(res_corr_hist, color=COLORS[1], label="Corrected")

        if fit:
            plot_gauss_fit(res_hist, label='Uncorrected Fit', color=COLORS[0])
            plot_gauss_fit(res_corr_hist, label='TW Corrected Fit', color=COLORS[1])

        plt.title(f"Resolution Distribution \n {self.run_config}")
        plt.ylabel("Count")
        plt.legend()

    @figuration(width=9,height=6,font_size=18)
    def histo1D(self, xVar:str, pix:tuple=None, xcuts:tuple=None, do_fit:bool=True):
        """Plots 1D histogram based off the field name "xVar" or the awkard event array. """
        h = self.get_hist(xVar,pix=pix)
        hep.histplot(h, color=COLORS[0])
        if do_fit:
            plot_gauss_fit(h, color=COLORS[2])
        plt.xlabel(xVar)
        plt.ylabel('counts')
        self.draw_thresh_vline(xVar, cuts=xcuts)
        pix_label = pix_labeler(pix)
        plt.title(f'{xVar}{pix_label} n: {sum(h.values())}\n{self.run_config} ')
        plt.legend()

    @figuration(width=9,height=9,font_size=18)
    def heatmap(self, xVar:str, yVar:str, pix:tuple=None, xcuts:tuple=None, ycuts:tuple=None):
        """
        2D histogram of any field selected by "xVar" and "yVar" in loaded awkward events array
        xcuts and ycuts are tuples of the high and low to override the thresholds
        """
        xVals = self.get_field_vals(xVar, pix=pix)
        yVals = self.get_field_vals(yVar, pix=pix)

        hist_2d = Hist(
            hist.axis.Regular(*self.hist_bins[xVar], name=xVar),
            hist.axis.Regular(*self.hist_bins[yVar], name=yVar),
        ).fill(xVals, yVals)

        hep.hist2dplot(hist_2d)

        plt.xlabel(xVar)

        self.draw_thresh_vline(xVar,cuts=xcuts)
        self.draw_thresh_hline(yVar,cuts=ycuts)

        plt.title(f'{yVar} vs {xVar} {pix} n: {len(xVals)}:\n{self.run_config}')
        plt.legend(labelcolor='w')
 
    @figuration()
    def timewalk_scatter(self, pix:tuple):
        model='cubicLM'
        row, col = pix
        a, b, c, d = self.tw_corr_data[0][row][col], self.tw_corr_data[1][row][col], self.tw_corr_data[2][row][col], self.tw_corr_data[3][row][col]
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        y = self.get_field_vals('dt', pix=pix).to_numpy()
        x = self.get_field_vals('tot', pix=pix).to_numpy()
        m = np.median(x)
        s = np.std(x)
        idx = np.abs(x - m) < 2*s
        plt.scatter(x[idx], y[idx], label = f'Fitting Data: {ak.count_nonzero(idx)}', alpha = 0.5, color=COLORS[0])
        plt.scatter(x[~idx], y[~idx], label = f'Rejected Data: {ak.count_nonzero(~idx)}', alpha = 0.5, color=COLORS[1])
        x.sort()
        y = tw.predict(model, x, [a, b, c, d])
        plt.title(f"Row {row} Col {col} \n n: {len(x)}")
        plt.ylabel(r"$\Delta T$")
        plt.xlabel("TOT")
        plt.plot(x, y, color=COLORS[2], label=buildLabel([a, b, c, d], model) + '\n m = {:.2f} \n s = {:.2f}'.format(m,s))
        plt.legend()

    def get_field_vals(self, field:str, pix:tuple=None) -> akArray:
        """Retrieves a values from an awkward array by selecting the field."""
        data = self.events
        if pix is not None:
            field_data = data[field]
            data = ak.flatten(field_data[(data.row==pix[0]) & (data.col==pix[1])])
            return data#[field]
        else:
            return ak.flatten(data[field])
        
    def get_hist(self, xVar:str, pix:tuple=None) -> Hist:
        """Takes field (corresponding to field in the events awkward array) and fills the 1D histogram"""
        xVals = self.get_field_vals(xVar, pix=pix)
        hist_axis = hist.axis.Regular(*self.hist_bins[xVar], name=xVar)
        return Hist(hist_axis).fill(xVals)
    
    def _draw_thresh_line(self, field:str, is_vertical:bool, cuts:tuple = None, ):
        if cuts is not None:
            self.update_thresholds(f"{field}_low", cuts[0])
            self.update_thresholds(f"{field}_high", cuts[1])

        for thresh_name, thresh in asdict(self.thresholds).items():
            if 'code' in field and field in thresh_name:
                if is_vertical:
                    plt.axvline(thresh, color=COLORS[2], label=f"{thresh_name}={thresh}")
                else:
                    plt.axhline(thresh, color=COLORS[2], label=f"{thresh_name}={thresh}")
    
    def update_thresholds(self, thresh_name:str, thresh_value: int):
        setattr(self.thresholds, thresh_name, thresh_value)

    def draw_thresh_vline(self, field: str, cuts:tuple=None):
        self._draw_thresh_line(field, True, cuts=cuts)
    def draw_thresh_hline(self, field: str, cuts:tuple=None):
        self._draw_thresh_line(field, False, cuts=cuts)

    # def beam_spot(self, percent, high=None, pixel_excludes:list[tuple]=None) -> np.ndarray:
    #     if high is None:
    #         high = self.hit_map_data.max()

    #     mask = (self.hit_map_data <= high) & (self.hit_map_data > 0)

    #     filtered_data = self.hit_map_data[mask]
    #     threshold = np.percentile(filtered_data, 100 - percent)

    #     mask &= (self.hit_map_data >= threshold)
    #     if pixel_excludes is not None:
    #         for pix in pixel_excludes:
    #             mask[pix[0], pix[1]] = False
    #     return mask
    


