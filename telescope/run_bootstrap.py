#!/usr/bin/env python3
import os
import awkward as ak
import numpy as np
import json
import pandas as pd
import argparse

import hist
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mplhep as hep
from scipy.optimize import curve_fit
plt.style.use(hep.style.CMS)

# (Built from Daniel's original script)


## Function to derive time walk corrections
# From original script, probably will be replaced by unbinned fit by Joshua

def linear(x, *p0):
    a, b = p0
    return a + b*x

def polynomial(x, *p0):
    a, b, c, d = p0
    return a + b*x + c*x**2 + d*x**3

def gaus(x, *p0):
    N, mu, sigma = p0
    return N*np.exp(-(x-mu)**2/(2.0*sigma**2))

def plotLayerMaps(name, hist0, hist1, hist2, mc = 'viridis'):
    if mc=='viridis':
        mc = plt.get_cmap('viridis')
        cp='w'
    else:
        cp='k'
    maxcmap = max([np.max(hist0), np.max(hist1), np.max(hist2)])
    fig, ax = plt.subplots(1,3,figsize=(33,11))
    ax[2].set_title("Layer i: Module 38")
    ax[2].matshow(hist0, cmap=mc, vmin=0, vmax=maxcmap)
    for i in range(16):
        for j in range(16):
            if int(hist0[i,j]): text = ax[2].text(j, i, int(hist0[i,j]), ha="center", va="center", color=cp, fontsize="xx-small")
    ax[1].set_title("Layer j: Module 36")
    ax[1].matshow(hist1, cmap=mc, vmin=0, vmax=maxcmap)
    for i in range(16):
        for j in range(16):
            if int(hist1[i,j]): text = ax[1].text(j, i, int(hist1[i,j]), ha="center", va="center", color=cp, fontsize="xx-small")
    ax[0].set_title("Layer k: Module 37")
    cax = ax[0].matshow(hist2, cmap=mc, vmin=0, vmax=maxcmap)
    for i in range(16):
        for j in range(16):
            if int(hist2[i,j]): text = ax[0].text(j, i, int(hist2[i,j]), ha="center", va="center", color=cp, fontsize="xx-small")
    ax[0].set_ylabel(r'$Row$')
    ax[0].set_xlabel(r'$Column$')
    ax[1].set_xlabel(r'$Column$')
    ax[2].set_xlabel(r'$Column$')
    fig.colorbar(cax,ax=ax)
    fig.savefig(f"{here}/{outputDir}/{name}.pdf")
    fig.savefig(f"{here}/{outputDir}/{name}.png")


def plotLayerGraphs(name, plot = [], title = [], xlabel = [], ylabel = ['Events']):
    ww = 27 if len(ylabel)==1 else 31
    hh = 8
    fig, ax = plt.subplots(1,3,figsize=(ww,hh))
    for p in range(len(plot)): plot[p].plot1d(ax=ax[p])
    for p in range(len(title)): ax[p].set_title(title[p])
    for p in range(len(xlabel)): ax[p].set_xlabel(xlabel[p])
    if len(ylabel) == 1:
        ax[0].set_ylabel(r'$Events$')
    else:
        for p in range(len(ylabel)): ax[p].set_ylabel(ylabel[p])
    fig.savefig(f"{here}/{outputDir}/{name}.pdf")
    fig.savefig(f"{here}/{outputDir}/{name}.png")

def plotLayer2DGraphs(name, axis = [], plot = [], title = [], xlabel = [], ylabel = []):
    ww = 32
    hh = 8
    fig, ax = plt.subplots(1,3,figsize=(ww,hh))
    for p in range(len(plot)): plot[p].project(axis[0], axis[1]).plot2d(ax=ax[p])
    for p in range(len(title)): ax[p].set_title(title[p])
    for p in range(len(xlabel)): ax[p].set_xlabel(xlabel[p])
    for p in range(len(ylabel)): ax[p].set_ylabel(ylabel[p])
    fig.savefig(f"{here}/{outputDir}/{name}.pdf")
    fig.savefig(f"{here}/{outputDir}/{name}.png")

def filterTriplet(events, pixel_i = [], pixel_j = [], pixel_k = []):
    ipixel_sel = (events.chipid==(38<<2)) & (events.row==pixel_i[0]) & (events.col==pixel_i[1])
    jpixel_sel = (events.chipid==(36<<2)) & (events.row==pixel_j[0]) & (events.col==pixel_j[1])
    kpixel_sel = (events.chipid==(37<<2)) & (events.row==pixel_k[0]) & (events.col==pixel_k[1])
    selection = (ipixel_sel) | (jpixel_sel) | (kpixel_sel)
    selected_events = ak.zip({'event': events.event[ak.num(selection)>0],
                              #'l1counter': events.l1counter[ak.num(selection)>0],
                              #'nhits': events.nhits[ak.num(selection)>0],
                              #'bcid': events.bcid[ak.num(selection)>0],
                              'row': events.row[selection],
                              'col': events.col[selection],
                              'chipid': events.chipid[selection],
                              'tot_code': events.tot_code[selection],
                              'toa_code': events.toa_code[selection],
                              'cal_code': events.cal_code[selection],
                              'calmean': events.calmean[selection],
                              'bin': events.bin[selection],
                              'toa': events.toa[selection],
                              'tot': events.tot[selection]}, depth_limit=1)
    return selected_events

def calc_timewalk_corrections(timewalk_hist, polyfit=False,unbinned=True):
    print(f"Timewalk {row=}, {col=}")
    p0 = [12, -0.15] if not polyfit else [12, -0.15, 0.1, 0.1]
    tw_corr = p0
    tw_corr_0, tw_corr_1, tw_corr_2, tw_corr_3 = [0,0,0,0]

    try:
        bin_centers = timewalk_hist.axes.centers[0]
        if not unbinned:
            tw_corr, var_matrix = curve_fit(
                linear if not polyfit else polynomial,
                bin_centers,
                timewalk_hist.values(),
                check_finite=False,
                sigma=np.ones_like(timewalk_hist.values())*0.1,
                p0=p0)

            tw_corr_0 = tw_corr[0]
            tw_corr_1 = tw_corr[1]
            if polyfit:
                tw_corr_2 = tw_corr[2]
                tw_corr_3 = tw_corr[3]

        else: # poor (memory) man's approach to an unbinned fit, just remove the bins with 0 entries
            try:
                tw_corr, var_matrix = curve_fit(
                    linear if not polyfit else polynomial,
                    bin_centers[timewalk_hist.values()>0],
                    timewalk_hist.values()[timewalk_hist.values()>0],
                    p0=p0
                )
                tw_corr_0 = tw_corr[0]
                tw_corr_1 = tw_corr[1]
                if polyfit:
                    tw_corr_2 = tw_corr[2]
                    tw_corr_3 = tw_corr[3]
            except ValueError:
                print("Fit fails because of only empty bins")
            except TypeError:
                print("Fit fails because of too few filled bins")

        return [tw_corr_0, tw_corr_1, tw_corr_2, tw_corr_3]
    except KeyError:
        print(f"Probably no events for {row=}, {col=}")



here = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--run_start', action='store', default=5707, help="Run to start with")
    argParser.add_argument('--run_stop',  action='store', default=6106, help="Run to finish")
    argParser.add_argument('--ipix', default=[], nargs="+", help="Select pixel in layer i e.g. [1, 12]")
    argParser.add_argument('--jpix', default=[], nargs="+", help="Select pixel in layer j e.g. [3, 12]")
    argParser.add_argument('--kpix', default=[], nargs="+", help="Select pixel in layer k e.g. [3, 12]")
    argParser.add_argument('--polynomial', action='store_true', help="Run a poly fit instead of linear")
    args = argParser.parse_args()

    mc = LinearSegmentedColormap.from_list("my_colormap", [(1,1,1), (192./255, 41./255, 41./255)])
    #mc = plt.get_cmap('viridis')

    ### Running options
    run_start = int(args.run_start) # No smaller than 5707
    run_stop = int(args.run_stop) # No higher than 6106
    selectAllLayersHit = False
    if len(args.ipix)==0 or len(args.jpix)==0 or len(args.kpix)==0:
        print('Configuration is not set to run over a defined pixel triplet')
        print('Setting one hit per layer configuration...')
        selectAllLayersHit = True
        selectTriplet = False
    else:
        selectTriplet = True

    ### Read events
    all_events = []
    read_files = 0
    for i in range(run_start,run_stop): 
        in_file = f"/eos/user/f/fernance/ETL/TestBeam/ETROC_output/{i}_merged.json"
        if os.path.isfile(in_file):
            read_files += 1
            with open(in_file, "r") as f:
                all_events.append(ak.from_json(json.load(f)))
        else:
            print(f'Missing file: {in_file}')
    events_raw = ak.concatenate(all_events)

    print(f'Reading number of files {read_files}')

    ## Define the output
    outputDir = 'results'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    print('Output will be saved in {outputDir}/ path')
    for (dirpath, dirnames, filenames) in os.walk(outputDir):
        os.system(f"cp utils/index.php {dirpath}")

    ### Sanity check
    # Removes bad processed events where the chipid and the row length doesn't match
    num_row = ak.num(events_raw.row)
    num_chipid = ak.num(events_raw.chipid)
    events = events_raw[num_row  == num_chipid]

    ### Define the layers for the hits
    events['x'] = (2*events.row + 1) * 1.3/2 + 0.3  # 300um edge, 1.3mm wide pixels
    events['y'] = (30 - 2*events.col - 1) * 1.3/2 - 0.6  # 300um edge, 1.3mm wide pixels
    events['z'] = -122*(events.chipid==(37<<2)) - 61*(events.chipid==(36<<2))

    ### Initial selection of events
    # Requires that all layers have exactly one hit
    all_layer_hit_candidates = events[ak.all(events.nhits==1, axis=1)]
    # Checking that for every hit in row < 5 we have a columm indicating that the pixel lives there!
    all_layer_hit_candidates_no_noise_selection = (ak.num(all_layer_hit_candidates.col[(all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] < 5)]) >0)
    # Only requires to have at least one hit in acceptance region (layer i)
    candidates_no_noise_selection = (ak.num(events.col[(events.row[events.chipid==(38<<2)] < 5)]) >0)

    if selectAllLayersHit:
        events = events[all_layer_hit_candidates_no_noise_selection]
    else:
        events = events[candidates_no_noise_selection]

    #
    ### Loop over all hits
    #

    hits0 = np.zeros([16, 16])
    hits1 = np.zeros([16, 16])
    hits2 = np.zeros([16, 16])
    calsum0 = np.zeros([16,16])
    calsum1 = np.zeros([16,16])
    calsum2 = np.zeros([16,16])
    #for ev in all_layer_hit_candidates[all_layer_hit_candidates_no_noise_selection]:
    for ev in events:
        for row, col in zip(ev.row[ev.chipid==(38 << 2)], ev.col[ev.chipid==(38 << 2)]):
            hits0[row,col]+=1
            calsum0[row,col]+= ak.sum(ev.cal_code[ev.chipid==(38 << 2)][(ev.row[ev.chipid==(38 << 2)]==row)&(ev.col[ev.chipid==(38 << 2)]==col)])
        for row, col in zip(ev.row[ev.chipid==(36 << 2)], ev.col[ev.chipid==(36 << 2)]):
            hits1[row,col]+=1
            calsum1[row,col]+= ak.sum(ev.cal_code[ev.chipid==(36 << 2)][(ev.row[ev.chipid==(36 << 2)]==row)&(ev.col[ev.chipid==(36 << 2)]==col)])
        for row, col in zip(ev.row[ev.chipid==(37 << 2)], ev.col[ev.chipid==(37 << 2)]):
            hits2[row,col]+=1
            calsum2[row,col]+= ak.sum(ev.cal_code[ev.chipid==(37 << 2)][(ev.row[ev.chipid==(37 << 2)]==row)&(ev.col[ev.chipid==(37 << 2)]==col)])

    calmean0 = np.nan_to_num(calsum0 / hits0) 
    calmean1 = np.nan_to_num(calsum1 / hits1)
    calmean2 = np.nan_to_num(calsum2 / hits2)

    # Hit map
    plotLayerMaps('heatmap_allPixels', hits0, hits1, hits2, mc)

    # Cal mean histogram
    plotLayerMaps('calmean_allPixels', calmean0, calmean1, calmean2, 'viridis')

    
    #
    # Observables definition
    #
    calmean_values = []
    for ev in events:
        calmean_values.append(calmean0[ev.row,ev.col]*(ev.chipid==(38<<2)) + calmean1[ev.row,ev.col]*(ev.chipid==(36<<2)) + calmean2[ev.row,ev.col]*(ev.chipid==(37<<2)))

    events['calmean'] = calmean_values
    events['bin'] = 3.125 / events.calmean # events.cal_code if we would like to use the actual value
    events['toa'] = 12.5 - events.bin * events.toa_code
    events['tot'] = (2*events.tot_code - np.floor(events.tot_code/32))*events.bin


    #
    ### Select one single pixel in layer i
    #
    if selectTriplet:
         row_i = int(args.ipix[0])
         col_i = int(args.ipix[1])
         #all_layer_hit_candidates_pixel_i = (ak.num(all_layer_hit_candidates.col[((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] == row_i)&((all_layer_hit_candidates.col[all_layer_hit_candidates.chipid==(38<<2)] == col_i)))]) >0)
         candidates_pixel_i = (ak.num(events.col[((events.row[events.chipid==(38<<2)] == row_i)&((events.col[events.chipid==(38<<2)] == col_i)))]) == 1)

         hits0 = np.zeros([16, 16])
         hits1 = np.zeros([16, 16])
         hits2 = np.zeros([16, 16])
         #for ev in all_layer_hit_candidates[all_layer_hit_candidates_pixel_i]:
         for ev in events[candidates_pixel_i]:
             for row, col in zip(ev.row[ev.chipid==(38 << 2)], ev.col[ev.chipid==(38 << 2)]):
                 hits0[row,col]+=1
             for row, col in zip(ev.row[ev.chipid==(36 << 2)], ev.col[ev.chipid==(36 << 2)]):
                 hits1[row,col]+=1
             for row, col in zip(ev.row[ev.chipid==(37 << 2)], ev.col[ev.chipid==(37 << 2)]):
                 hits2[row,col]+=1

         plotLayerMaps(f'heatmap_ipixel_row_{row_i}_col_{col_i}', hits0, hits1, hits2, mc)


    #
    ### Select one single pixel in layer j on top of already selected pixel in layer i
    #
    if selectTriplet:
        row_j = int(args.jpix[0])
        col_j = int(args.jpix[1])
        #all_layer_hit_candidates_pixel_j = (ak.num(all_layer_hit_candidates.col[((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(36<<2)] == row_j)&((all_layer_hit_candidates.col[all_layer_hit_candidates.chipid==(36<<2)] == col_j)))]) >0)
        candidates_pixel_j = (ak.num(events.col[((events.row[events.chipid==(36<<2)] == row_j)&((events.col[events.chipid==(36<<2)] == col_j)))]) == 1)

        hits0 = np.zeros([16, 16])
        hits1 = np.zeros([16, 16])
        hits2 = np.zeros([16, 16])
        for ev in events[(candidates_pixel_i)&(candidates_pixel_j)]:
            for row, col in zip(ev.row[ev.chipid==(38 << 2)], ev.col[ev.chipid==(38 << 2)]):
                hits0[row,col]+=1
            for row, col in zip(ev.row[ev.chipid==(36 << 2)], ev.col[ev.chipid==(36 << 2)]):
                hits1[row,col]+=1
            for row, col in zip(ev.row[ev.chipid==(37 << 2)], ev.col[ev.chipid==(37 << 2)]):
                hits2[row,col]+=1

        plotLayerMaps(f'heatmap_ipixel_row_{row_i}_col_{col_i}_jpixel_row_{row_j}_col_{col_j}', hits0, hits1, hits2, mc)

    #
    ### Manual selection of a 3-pixel triplet in layers i + j + k following previous selections
    #
    if selectTriplet:
        row_k = int(args.kpix[0])
        col_k = int(args.kpix[1])
        #all_layer_hit_candidates_pixel_k = (ak.num(all_layer_hit_candidates.col[((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(37<<2)] == row_k)&((all_layer_hit_candidates.col[all_layer_hit_candidates.chipid==(37<<2)] == col_k)))]) >0)
        #all_layer_hit_triplet_events = all_layer_hit_candidates[(all_layer_hit_candidates_pixel_i) & (all_layer_hit_candidates_pixel_j) & (all_layer_hit_candidates_pixel_k)]

        candidates_pixel_k = (ak.num(events.col[((events.row[events.chipid==(37<<2)] == row_k)&((events.col[events.chipid==(37<<2)] == col_k)))]) == 1)
        triplet_events = events[(candidates_pixel_i) & (candidates_pixel_j) & (candidates_pixel_k)]

        hits0 = np.zeros([16, 16])
        hits1 = np.zeros([16, 16])
        hits2 = np.zeros([16, 16])
        for ev in events[(candidates_pixel_i)&(candidates_pixel_j)&(candidates_pixel_k)]:
            for row, col in zip(ev.row[ev.chipid==(38 << 2)], ev.col[ev.chipid==(38 << 2)]):
                hits0[row,col]+=1
            for row, col in zip(ev.row[ev.chipid==(36 << 2)], ev.col[ev.chipid==(36 << 2)]):
                hits1[row,col]+=1
            for row, col in zip(ev.row[ev.chipid==(37 << 2)], ev.col[ev.chipid==(37 << 2)]):
                hits2[row,col]+=1

        plotLayerMaps(f'heatmap_ipixel_row_{row_i}_col_{col_i}_jpixel_row_{row_j}_col_{col_j}_kpixel_row_{row_k}_col_{col_k}', hits0, hits1, hits2, mc)


    # If we select the triplet then these will be our preselected events
    # If not, apply just a dummy noise reduction on events with only one hit in every layer
    if selectTriplet:
        print('-> Analysis will run for pixels:')
        print(f'   - Layer i: (row_i, col_i)')
        print(f'   - Layer j: (row_j, col_j)')
        print(f'   - Layer k: (row_k, col_k)')
        presel_events = filterTriplet(triplet_events, pixel_i = [row_i, col_i], pixel_j = [row_j, col_j], pixel_k = [row_k, col_k])
    else:
        all_layer_hit_candidates_no_noise_selection = (ak.num(all_layer_hit_candidates.col[(all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] < 5)]) >0)
        presel_events = all_layer_hit_candidates[all_layer_hit_candidates_no_noise_selection]

    print(f'Total of events (before selection): {len(presel_events)}')
    print(presel_events[0].to_list())

    ####### Selection of layers with only one hit
    #### The same code should run for the track reconstruction
    applySelection = True
    min_toa_code_i = 500
    min_toa_code_j = 500
    min_toa_code_k = 25
    min_tot_code_i = 90
    min_tot_code_j = 50
    min_tot_code_k = 50

    if applySelection:
        toa_selection_i = (ak.num(presel_events.toa_code[presel_events.chipid==(38<<2)][presel_events.toa_code[presel_events.chipid==(38<<2)] > min_toa_code_i]) > 0)
        toa_selection_j = (ak.num(presel_events.toa_code[presel_events.chipid==(36<<2)][presel_events.toa_code[presel_events.chipid==(36<<2)] > min_toa_code_j]) > 0)
        toa_selection_k = (ak.num(presel_events.toa_code[presel_events.chipid==(37<<2)][presel_events.toa_code[presel_events.chipid==(37<<2)] > min_toa_code_k]) > 0)
        all_layer_hit_candidates_toa_selection = toa_selection_i & toa_selection_j & toa_selection_k
        tot_selection_i = (ak.num(presel_events.tot_code[presel_events.chipid==(38<<2)][presel_events.tot_code[presel_events.chipid==(38<<2)] > min_tot_code_i]) > 0)
        tot_selection_j = (ak.num(presel_events.tot_code[presel_events.chipid==(36<<2)][presel_events.tot_code[presel_events.chipid==(36<<2)] > min_tot_code_j]) > 0)
        tot_selection_k = (ak.num(presel_events.tot_code[presel_events.chipid==(37<<2)][presel_events.tot_code[presel_events.chipid==(37<<2)] > min_tot_code_k]) > 0)
        all_layer_hit_candidates_tot_selection = tot_selection_i & tot_selection_j & tot_selection_k
        ## For cal selection we select that at least 3 hits have cal_code values closer to the mean by 15
        cal_selection = (ak.num(presel_events['calmean'][(abs(presel_events['calmean']-presel_events['cal_code']) < 15)]) > 2)
        sel_events = presel_events[all_layer_hit_candidates_toa_selection & all_layer_hit_candidates_tot_selection & cal_selection]
    else:
        sel_events = presel_events

    print(f'Total of events (after selection): {len(sel_events)}')

    ### Get basic plots (Code is fast enough so no need to do fancy stuff here)
    # 
    # For each module, use the mean of the other two and measure the difference
    # i = 38
    # j = 36
    # k = 37

    # Axis
    time_axis = hist.axis.Regular(100, -15, 15, name="time", label="time")
    toacode_axis = hist.axis.Regular(80, 0, 800, name="toa_code", label="TOA CODE")
    totcode_axis = hist.axis.Regular(80, 0, 600, name="tot_code", label="TOT CODE")
    toa_axis = hist.axis.Regular(80, 0, 12, name="toa", label="TOA")
    tot_axis = hist.axis.Regular(80, 0, 14, name="tot", label="TOT")

    # Histos
    toacode_hist_i = hist.Hist(toacode_axis)
    toacode_hist_j = hist.Hist(toacode_axis)
    toacode_hist_k = hist.Hist(toacode_axis)

    totcode_hist_i = hist.Hist(totcode_axis)
    totcode_hist_j = hist.Hist(totcode_axis)
    totcode_hist_k = hist.Hist(totcode_axis)

    totcode_toacode_hist_i = hist.Hist(totcode_axis, toacode_axis)
    totcode_toacode_hist_j = hist.Hist(totcode_axis, toacode_axis)
    totcode_toacode_hist_k = hist.Hist(totcode_axis, toacode_axis)

    toa_hist_i = hist.Hist(toa_axis)
    toa_hist_j = hist.Hist(toa_axis)
    toa_hist_k = hist.Hist(toa_axis)

    tot_hist_i = hist.Hist(tot_axis)
    tot_hist_j = hist.Hist(tot_axis)
    tot_hist_k = hist.Hist(tot_axis)

    tot_toa_hist_i = hist.Hist(tot_axis, toa_axis)
    tot_toa_hist_j = hist.Hist(tot_axis, toa_axis)
    tot_toa_hist_k = hist.Hist(tot_axis, toa_axis)

    # Fill
    print(sel_events.to_list())

    sel_pixel_i = (sel_events.row==row_i) & (sel_events.col==col_i)
    sel_pixel_j = (sel_events.row==row_j) & (sel_events.col==col_j)
    sel_pixel_k = (sel_events.row==row_k) & (sel_events.col==col_k)

    if False:
    #if selectTriplet:
        toacode_i = sel_events.toa_code[(sel_events.chipid==(38<<2)) & (sel_pixel_i)]
        toacode_j = sel_events.toa_code[(sel_events.chipid==(36<<2)) & (sel_pixel_j)]
        toacode_k = sel_events.toa_code[(sel_events.chipid==(37<<2)) & (sel_pixel_k)]

        totcode_i = sel_events.tot_code[(sel_events.chipid==(38<<2)) & (sel_pixel_i)]
        totcode_j = sel_events.tot_code[(sel_events.chipid==(36<<2)) & (sel_pixel_j)]
        totcode_k = sel_events.tot_code[(sel_events.chipid==(37<<2)) & (sel_pixel_k)]

        toa_i = sel_events.toa[(sel_events.chipid==(38<<2)) & (sel_pixel_i)]
        toa_j = sel_events.toa[(sel_events.chipid==(36<<2)) & (sel_pixel_j)]
        toa_k = sel_events.toa[(sel_events.chipid==(37<<2)) & (sel_pixel_k)]

        tot_i = sel_events.tot[(sel_events.chipid==(38<<2)) & (sel_pixel_i)]
        tot_j = sel_events.tot[(sel_events.chipid==(36<<2)) & (sel_pixel_j)]
        tot_k = sel_events.tot[(sel_events.chipid==(37<<2)) & (sel_pixel_k)]
    else:
        toacode_i = sel_events.toa_code[sel_events.chipid==(38<<2)]
        toacode_j = sel_events.toa_code[sel_events.chipid==(36<<2)]
        toacode_k = sel_events.toa_code[sel_events.chipid==(37<<2)]

        totcode_i = sel_events.tot_code[sel_events.chipid==(38<<2)]
        totcode_j = sel_events.tot_code[sel_events.chipid==(36<<2)]
        totcode_k = sel_events.tot_code[sel_events.chipid==(37<<2)]

        toa_i = sel_events.toa[sel_events.chipid==(38<<2)]
        toa_j = sel_events.toa[sel_events.chipid==(36<<2)]
        toa_k = sel_events.toa[sel_events.chipid==(37<<2)]

        tot_i = sel_events.tot[sel_events.chipid==(38<<2)]
        tot_j = sel_events.tot[sel_events.chipid==(36<<2)]
        tot_k = sel_events.tot[sel_events.chipid==(37<<2)]

    toacode_hist_i.fill(ak.flatten(toacode_i)) 
    toacode_hist_j.fill(ak.flatten(toacode_j)) 
    toacode_hist_k.fill(ak.flatten(toacode_k)) 

    totcode_hist_i.fill(ak.flatten(totcode_i)) 
    totcode_hist_j.fill(ak.flatten(totcode_j)) 
    totcode_hist_k.fill(ak.flatten(totcode_k)) 

    toa_hist_i.fill(ak.flatten(toa_i)) 
    toa_hist_j.fill(ak.flatten(toa_j)) 
    toa_hist_k.fill(ak.flatten(toa_k)) 

    tot_hist_i.fill(ak.flatten(tot_i)) 
    tot_hist_j.fill(ak.flatten(tot_j)) 
    tot_hist_k.fill(ak.flatten(tot_k)) 

    totcode_toacode_hist_i.fill(ak.flatten(totcode_i), ak.flatten(toacode_i))
    totcode_toacode_hist_j.fill(ak.flatten(totcode_j), ak.flatten(toacode_j))
    totcode_toacode_hist_k.fill(ak.flatten(totcode_k), ak.flatten(toacode_k))

    tot_toa_hist_i.fill(ak.flatten(tot_i), ak.flatten(toa_i))
    tot_toa_hist_j.fill(ak.flatten(tot_j), ak.flatten(toa_j))
    tot_toa_hist_k.fill(ak.flatten(tot_k), ak.flatten(toa_k))

    
    # Plotting
    plotLayerGraphs('merged_layers_toacode', plot = [toacode_hist_k, toacode_hist_j, toacode_hist_i], title = ["Third layer (k=37)", "Second layer (j=36)", "First layer (i=38)"], xlabel = [r'$TOA\_CODE_{k}$', r'$TOA\_CODE_{j}$', r'$TOA\_CODE_{i}$'], ylabel = [r'$Events$'])
    plotLayerGraphs('merged_layers_totcode', plot = [totcode_hist_k, totcode_hist_j, totcode_hist_i], title = ["Third layer (k=37)", "Second layer (j=36)", "First layer (i=38)"], xlabel = [r'$TOT\_CODE_{k}$', r'$TOT\_CODE_{j}$', r'$TOT\_CODE_{i}$'], ylabel = [r'$Events$'])
    plotLayerGraphs('merged_layers_toa', plot = [toa_hist_k, toa_hist_j, toa_hist_i], title = ["Third layer (k=37)", "Second layer (j=36)", "First layer (i=38)"], xlabel = [r'$TOA_{k}$', r'$TOA_{j}$', r'$TOA_{i}$'], ylabel = [r'$Events$'])
    plotLayerGraphs('merged_layers_tot', plot = [tot_hist_k, tot_hist_j, tot_hist_i], title = ["Third layer (k=37)", "Second layer (j=36)", "First layer (i=38)"], xlabel = [r'$TOT_{k}$', r'$TOT_{j}$', r'$TOT_{i}$'], ylabel = [r'$Events$'])
    plotLayer2DGraphs('merged_layers_toacode_totcode_2d', axis = ['tot_code', 'toa_code'], plot = [totcode_toacode_hist_k, totcode_toacode_hist_j, totcode_toacode_hist_i], title = ["Third layer (k=37)", "Second layer (j=36)", "First layer (i=38)"], xlabel = [r'$TOT\_CODE_{k}$', r'$TOT\_CODE_{j}$', r'$TOT\_CODE_{i}$'], ylabel = [r'$TOA\_CODE_{k}$', r'$TOA\_CODE_{j}$', r'$TOA\_CODE_{i}$'])
    plotLayer2DGraphs('merged_layers_toa_tot_2d', axis = ['tot', 'toa'], plot = [tot_toa_hist_k, tot_toa_hist_j, tot_toa_hist_i], title = ["Third layer (k=37)", "Second layer (j=36)", "First layer (i=38)"], xlabel = [r'$TOT_{k}$', r'$TOT_{j}$', r'$TOT_{i}$'], ylabel = [r'$TOA_{k}$', r'$TOA_{j}$', r'$TOA_{i}$'])


    ### Time walk corrections (following Mustaza's approach)
    # 
    # For each module, use the mean of the other two and measure the difference
    # i = 38
    # j = 36
    # k = 37

    # DeltaTOA_ijk
    dtoa_i_hist = hist.Hist(time_axis)
    dtoa_j_hist = hist.Hist(time_axis)
    dtoa_k_hist = hist.Hist(time_axis)

    if selectTriplet:
        dtoa_i = (sel_events.toa[(sel_events.chipid==(36<<2)) & (sel_pixel_j)] + sel_events.toa[(sel_events.chipid==(37<<2)) & (sel_pixel_k)])/2.0 - sel_events.toa[(sel_events.chipid==(38<<2)) & (sel_pixel_i)]
        dtoa_j = (sel_events.toa[(sel_events.chipid==(37<<2)) & (sel_pixel_k)] + sel_events.toa[(sel_events.chipid==(38<<2)) & (sel_pixel_i)])/2.0 - sel_events.toa[(sel_events.chipid==(36<<2)) & (sel_pixel_j)]
        dtoa_k = (sel_events.toa[(sel_events.chipid==(36<<2)) & (sel_pixel_j)] + sel_events.toa[(sel_events.chipid==(38<<2)) & (sel_pixel_i)])/2.0 - sel_events.toa[(sel_events.chipid==(37<<2)) & (sel_pixel_k)]
    else:
        dtoa_i = (sel_events.toa[sel_events.chipid==(36<<2)] + sel_events.toa[sel_events.chipid==(37<<2)])/2.0 - sel_events.toa[sel_events.chipid==(38<<2)]
        dtoa_j = (sel_events.toa[sel_events.chipid==(37<<2)] + sel_events.toa[sel_events.chipid==(38<<2)])/2.0 - sel_events.toa[sel_events.chipid==(36<<2)]
        dtoa_k = (sel_events.toa[sel_events.chipid==(36<<2)] + sel_events.toa[sel_events.chipid==(38<<2)])/2.0 - sel_events.toa[sel_events.chipid==(37<<2)]

    dtoa_i_hist.fill(ak.flatten(dtoa_i))
    dtoa_j_hist.fill(ak.flatten(dtoa_j))
    dtoa_k_hist.fill(ak.flatten(dtoa_k))

    fig, ax = plt.subplots(1,3,figsize=(24,8))

    dtoa_i_hist.plot1d(ax=ax[2])
    dtoa_j_hist.plot1d(ax=ax[1])
    dtoa_k_hist.plot1d(ax=ax[0])

    ax[2].set_title("First layer (i=38)")
    ax[1].set_title("Second layer (j=36)")
    ax[0].set_title("Third layer (k=37)")
    ax[0].set_ylabel(r'$Events$')
    ax[0].set_xlabel(r'$\Delta TOA_{k}$')
    ax[1].set_xlabel(r'$\Delta TOA_{j}$')
    ax[2].set_xlabel(r'$\Delta TOA_{i}$')
    fig.savefig(f"{here}/{outputDir}/merged_layers_delta_toa.pdf")
    fig.savefig(f"{here}/{outputDir}/merged_layers_delta_toa.png")
    
    ## <DeltaTOA_ijk> vs ToT
    tot_axis    = hist.axis.Regular(12, 2.0, 6.0, name="tot", label="tot")

    dtoa_tot_i_hist = hist.Hist(time_axis, tot_axis)
    dtoa_tot_j_hist = hist.Hist(time_axis, tot_axis)
    dtoa_tot_k_hist = hist.Hist(time_axis, tot_axis)

    tot_i = sel_events.tot[(sel_events.chipid==(38<<2)) & (sel_pixel_i)]
    tot_j = sel_events.tot[(sel_events.chipid==(36<<2)) & (sel_pixel_j)]
    tot_k = sel_events.tot[(sel_events.chipid==(37<<2)) & (sel_pixel_k)]

    dtoa_tot_i_hist.fill(time=ak.flatten(dtoa_i), tot=ak.flatten(tot_i))
    dtoa_tot_j_hist.fill(time=ak.flatten(dtoa_j), tot=ak.flatten(tot_j))
    dtoa_tot_k_hist.fill(time=ak.flatten(dtoa_k), tot=ak.flatten(tot_k))

    dtoa_tot_i_prof = dtoa_tot_i_hist.profile('time')
    dtoa_tot_j_prof = dtoa_tot_j_hist.profile('time')
    dtoa_tot_k_prof = dtoa_tot_k_hist.profile('time')

    # Fit and corrections
    unbinned = False
    if not unbinned:
        bin_centers_i = dtoa_tot_i_prof.axes.centers[0]
        bin_centers_j = dtoa_tot_j_prof.axes.centers[0]
        bin_centers_k = dtoa_tot_k_prof.axes.centers[0]
        p0 = [12, -0.15] if not args.polynomial else [12, -0.15, 0.1, 0.1]
        tw_corr_i = p0
        tw_corr_j = p0
        tw_corr_k = p0
        tw_corr_i, var_matrix_i = curve_fit(linear if not args.polynomial else polynomial, bin_centers_i[dtoa_tot_i_prof.values()>0], dtoa_tot_i_prof.values()[dtoa_tot_i_prof.values()>0], p0=p0)
        tw_corr_j, var_matrix_j = curve_fit(linear if not args.polynomial else polynomial, bin_centers_j[dtoa_tot_j_prof.values()>0], dtoa_tot_j_prof.values()[dtoa_tot_j_prof.values()>0], p0=p0)
        tw_corr_k, var_matrix_k = curve_fit(linear if not args.polynomial else polynomial, bin_centers_k[dtoa_tot_k_prof.values()<0], dtoa_tot_k_prof.values()[dtoa_tot_k_prof.values()<0], p0=p0)
        [tw_corr_0_i, tw_corr_1_i] = tw_corr_i[:2]
        [tw_corr_0_j, tw_corr_1_j] = tw_corr_j[:2]
        [tw_corr_0_k, tw_corr_1_k] = tw_corr_k[:2]
        if args.polynomial:
            [tw_corr_2_i, tw_corr_3_i] = tw_corr_i[2:4]
            [tw_corr_2_j, tw_corr_3_j] = tw_corr_j[2:4]
            [tw_corr_2_k, tw_corr_3_k] = tw_corr_k[2:4]
    else:
        # It won't work (in progress)
        tw_corr_0_i, tw_corr_1_i = np.polyfit(ak.flatten(tot_i), ak.flatten(dtoa_i), deg=1)
        tw_corr_0_j, tw_corr_1_j = np.polyfit(ak.flatten(tot_j), ak.flatten(dtoa_j), deg=1)
        tw_corr_0_k, tw_corr_1_k = np.polyfit(ak.flatten(tot_k), ak.flatten(dtoa_k), deg=1)


    fig, ax = plt.subplots(1,3,figsize=(27,8))


    if not unbinned:
        ax[0].plot(bin_centers_k, linear(bin_centers_k, *tw_corr_k) if not args.polynomial else polynomial(bin_centers_k, *tw_corr_k), color='red', label='Linear Fit\n a: {:.2f} \n b: {:.2f}'.format(tw_corr_k[0],tw_corr_k[1]))
        ax[1].plot(bin_centers_j, linear(bin_centers_j, *tw_corr_j) if not args.polynomial else polynomial(bin_centers_j, *tw_corr_j), color='red', label='Linear Fit\n a: {:.2f} \n b: {:.2f}'.format(tw_corr_j[0],tw_corr_j[1]))
        ax[2].plot(bin_centers_i, linear(bin_centers_i, *tw_corr_i) if not args.polynomial else polynomial(bin_centers_i, *tw_corr_i), color='red', label='Linear Fit\n a: {:.2f} \n b: {:.2f}'.format(tw_corr_i[0],tw_corr_i[1]))
        ax[0].errorbar(bin_centers_k, dtoa_tot_k_prof.values(), dtoa_tot_k_prof.variances(), fmt='o')
        ax[1].errorbar(bin_centers_j, dtoa_tot_j_prof.values(), dtoa_tot_j_prof.variances(), fmt='o')
        ax[2].errorbar(bin_centers_i, dtoa_tot_i_prof.values(), dtoa_tot_i_prof.variances(), fmt='o')
    else:
        ax[0].scatter(tot_k, dtoa_k)
        ax[1].scatter(tot_j, dtoa_j)
        ax[2].scatter(tot_i, dtoa_i)
        totseq = np.linspace(2.2, 5.8, num=100)
        ax[0].plot(totseq, tw_corr_0_k + tw_corr_1_k*totseq, color='red', label='Linear Fit\n a: {:.2f} \n b: {:.2f}'.format(tw_corr_0_k,tw_corr_0_k))
        ax[1].plot(totseq, tw_corr_0_j + tw_corr_1_j*totseq, color='red', label='Linear Fit\n a: {:.2f} \n b: {:.2f}'.format(tw_corr_0_j,tw_corr_0_j))
        ax[2].plot(totseq, tw_corr_0_i + tw_corr_1_i*totseq, color='red', label='Linear Fit\n a: {:.2f} \n b: {:.2f}'.format(tw_corr_0_i,tw_corr_0_i))
    

    #ax[0].set_ylim(1.5*min(dtoa_tot_k_prof.values()), 0.5*max(dtoa_tot_k_prof.values()))
    #ax[1].set_ylim(0.5*min(dtoa_tot_j_prof.values()), 1.5*max(dtoa_tot_j_prof.values()))
    #ax[2].set_ylim(0.5*min(dtoa_tot_i_prof.values()), 1.5*max(dtoa_tot_i_prof.values()))
    ax[0].set_ylim(-10, -7)
    ax[1].set_ylim(3.5, 6)
    ax[2].set_ylim(3.5, 6)

    #dtoa_tot_k_prof.plot1d(ax=ax[0])
    #dtoa_tot_j_prof.plot1d(ax=ax[1])
    #dtoa_tot_i_prof.plot1d(ax=ax[2])

    ax[2].set_title("First layer (i=38)")
    ax[1].set_title("Second layer (j=36)")
    ax[0].set_title("Third layer (k=37)")
    ax[0].set_xlabel(r'$TOT_{k}$')
    ax[1].set_xlabel(r'$TOT_{j}$')
    ax[2].set_xlabel(r'$TOT_{i}$')
    ax[0].set_ylabel(r'$<\Delta TOA_{k}>$')
    ax[1].set_ylabel(r'$<\Delta TOA_{j}>$')
    ax[2].set_ylabel(r'$<\Delta TOA_{i}>$')
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    fig.savefig(f"{here}/{outputDir}/merged_layers_tot_delta_toa_corrections.pdf")
    fig.savefig(f"{here}/{outputDir}/merged_layers_tot_delta_toa_corrections.png")

    ### Resulting time corrected TOA: TOA = TOA + f(TOA)
    corr_toa_i_hist = hist.Hist(time_axis)
    corr_toa_j_hist = hist.Hist(time_axis)
    corr_toa_k_hist = hist.Hist(time_axis)

    corr_i = linear(tot_i, *tw_corr_i) if not args.polynomial else polynomial(tot_i, *tw_corr_i)
    corr_j = linear(tot_j, *tw_corr_j) if not args.polynomial else polynomial(tot_j, *tw_corr_j)
    corr_k = linear(tot_k, *tw_corr_k) if not args.polynomial else polynomial(tot_k, *tw_corr_k)
    corr_toa_i = toa_i + corr_i
    corr_toa_j = toa_j + corr_j
    corr_toa_k = toa_k + corr_k

    corr_toa_i_hist.fill(ak.flatten(corr_toa_i))
    corr_toa_j_hist.fill(ak.flatten(corr_toa_j))
    corr_toa_k_hist.fill(ak.flatten(corr_toa_k))

    fig, ax = plt.subplots(1,3,figsize=(27,8))
    corr_toa_i_hist.plot1d(ax=ax[2])
    corr_toa_j_hist.plot1d(ax=ax[1])
    corr_toa_k_hist.plot1d(ax=ax[0])
    ax[2].set_title("First layer (i=38)")
    ax[1].set_title("Second layer (j=36)")
    ax[0].set_title("Third layer (k=37)")
    ax[0].set_ylabel('Events')
    ax[0].set_xlabel(r'$TOA_{k} + corr(TOT_k)$')
    ax[1].set_xlabel(r'$TOA_{j} + corr(TOT_j)$')
    ax[2].set_xlabel(r'$TOA_{i} + corr(TOT_i)$')
    fig.savefig(f"{here}/{outputDir}/merged_layers_toa_corrected.pdf")
    fig.savefig(f"{here}/{outputDir}/merged_layers_toa_corrected.png")


    ### Pairwise DeltaT
    time_axis = hist.axis.Regular(100, -8, 8, name="time", label="time")

    Tij_hist = hist.Hist(time_axis)
    Tjk_hist = hist.Hist(time_axis)
    Tki_hist = hist.Hist(time_axis)

    Tij = corr_toa_i - corr_toa_j
    Tjk = corr_toa_j - corr_toa_k
    Tki = corr_toa_k - corr_toa_i

    Tij_hist.fill(ak.flatten(Tij))
    Tjk_hist.fill(ak.flatten(Tjk))
    Tki_hist.fill(ak.flatten(Tki))

    bin_centers_ij = Tij_hist.axes.centers[0]
    bin_centers_jk = Tjk_hist.axes.centers[0]
    bin_centers_ki = Tki_hist.axes.centers[0]

    res_coeff_ij, var_matrix_ij = curve_fit(gaus, bin_centers_ij, Tij_hist.values(), p0=[500,5,60])
    res_coeff_jk, var_matrix_jk = curve_fit(gaus, bin_centers_jk, Tjk_hist.values(), p0=[500,5,60])
    res_coeff_ki, var_matrix_ki = curve_fit(gaus, bin_centers_ki, Tki_hist.values(), p0=[500,-5,60])

    fig, ax = plt.subplots(1,3,figsize=(24,8))
    ax[0].plot(bin_centers_ij, gaus(bin_centers_ij, *res_coeff_ij), color='red', label=r'Fit mean: {:.2f}, $\sigma: {:.0f} \pm {:.1f}$ (ps)'.format(res_coeff_ij[1],abs(res_coeff_ij[2])*1000, np.sqrt(var_matrix_ij[2,2])*1000))
    ax[1].plot(bin_centers_jk, gaus(bin_centers_jk, *res_coeff_jk), color='red', label=r'Fit mean: {:.2f}, $\sigma: {:.0f} \pm {:.1f}$ (ps)'.format(res_coeff_jk[1],abs(res_coeff_jk[2])*1000, np.sqrt(var_matrix_jk[2,2])*1000))
    ax[2].plot(bin_centers_ki, gaus(bin_centers_ki, *res_coeff_ki), color='red', label=r'Fit mean: {:.2f}, $\sigma: {:.0f} \pm {:.1f}$ (ps)'.format(res_coeff_ki[1],abs(res_coeff_ki[2])*1000, np.sqrt(var_matrix_ki[2,2])*1000))
    Tij_hist.plot1d(ax=ax[0])
    Tjk_hist.plot1d(ax=ax[1])
    Tki_hist.plot1d(ax=ax[2])
    ax[0].set_title("Pairwise T_{ij} (i=38, j=36)")
    ax[1].set_title("Pairwise T_{jk} (j=36, k=37)")
    ax[2].set_title("Pairwise T_{ki} (k=37, i=38)")
    ax[0].set_ylabel('Events')
    ax[0].set_xlabel(r'$TOA_{i} - TOA_{j}$')
    ax[1].set_xlabel(r'$TOA_{j} - TOA_{k}$')
    ax[2].set_xlabel(r'$TOA_{k} - TOA_{i}$')
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    fig.savefig(f"{here}/{outputDir}/merged_layers_Tijk.pdf")
    fig.savefig(f"{here}/{outputDir}/merged_layers_Tijk.png")

    print("Resolution results:")
    print(f"sigma_ij (ps) = {abs(res_coeff_ij[2])*1000}")
    print(f"sigma_jk (ps) = {abs(res_coeff_jk[2])*1000}")
    print(f"sigma_ki (ps) = {abs(res_coeff_ki[2])*1000}")
    sigma_i = (1/2.0**0.5)*( (abs(res_coeff_ij[2])*1000)**2 + (abs(res_coeff_ki[2])*1000)**2 - (abs(res_coeff_jk[2])*1000)**2)**0.5
    sigma_j = (1/2.0**0.5)*( (abs(res_coeff_ij[2])*1000)**2 + (abs(res_coeff_jk[2])*1000)**2 - (abs(res_coeff_ki[2])*1000)**2)**0.5
    sigma_k = (1/2.0**0.5)*( (abs(res_coeff_ki[2])*1000)**2 + (abs(res_coeff_jk[2])*1000)**2 - (abs(res_coeff_ij[2])*1000)**2)**0.5
    print(f"sigma_i (ps) = {sigma_i}")
    print(f"sigma_j (ps) = {sigma_j}")
    print(f"sigma_k (ps) = {sigma_k}")


    ############################################### Some code used to test the aligment (not used for now)
    # do some simple alignment
    single_pixel = (ak.num(all_layer_hit_candidates.col[((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] ==0)&((all_layer_hit_candidates.col[all_layer_hit_candidates.chipid==(38<<2)] ==10)))]) >0)
    sel_events = all_layer_hit_candidates[single_pixel]
    x_ref = np.mean(ak.flatten(sel_events.x[sel_events.chipid==(38<<2)]))
    y_ref = np.mean(ak.flatten(sel_events.y[sel_events.chipid==(38<<2)]))

    x_corr_2 = x_ref - ak.mean(sel_events.x[sel_events.chipid==(36<<2)])
    y_corr_2 = y_ref - ak.mean(sel_events.y[sel_events.chipid==(36<<2)])

    x_corr_3 = x_ref - ak.mean(sel_events.x[sel_events.chipid==(37<<2)])
    y_corr_3 = y_ref - ak.mean(sel_events.y[sel_events.chipid==(37<<2)])


    # apply alignment to different pixel
    single_pixel = (ak.num(all_layer_hit_candidates.col[((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] ==1)&((all_layer_hit_candidates.col[all_layer_hit_candidates.chipid==(38<<2)] ==12)))]) >0)
    sel_events = all_layer_hit_candidates[single_pixel]


    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    xs = sel_events.x
    xs = xs + sel_events.chipid==(38<<2)*0
    xs = xs + sel_events.chipid==(36<<2)*x_corr_2
    xs = xs + sel_events.chipid==(37<<2)*x_corr_3

    ys = sel_events.y
    ys = ys + sel_events.chipid==(38<<2)*0
    ys = ys + sel_events.chipid==(36<<2)*y_corr_2
    ys = ys + sel_events.chipid==(37<<2)*y_corr_3

    xs = ak.flatten(xs)
    ys = ak.flatten(ys)
    zs = ak.flatten(sel_events.z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    plt.Polygon([(0.3,0.6), (0.3, 1.9), (1.6,1.9), (1.6,0.6)], fill=True, closed=True, edgecolor='black')

    ax.scatter(xs[0:3], ys[0:3], zs[0:3], c='r')
    ax.scatter(xs[3:6]+0.5, ys[3:6], zs[3:6], c='b')
    ax.scatter(xs[6:9], ys[6:9]+0.5, zs[6:9], c='g')
    ax.scatter(xs[9:12]+0.5, ys[9:12]+0.5, zs[9:12], c='orange')

    ax.set_xlim(0,21.4)
    ax.set_ylim(0,21.4)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    fig.savefig(f"{here}/{outputDir}/hits_aligned.pdf")
    fig.savefig(f"{here}/{outputDir}/hits_aligned.png")

    #plt.show()


    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')

    ## For each set of style and range settings, plot n random points in the box
    ## defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    #xs = ak.flatten(sel_events.x)
    #ys = ak.flatten(sel_events.y)
    #zs = ak.flatten(sel_events.z)
    #ax.scatter(xs[:10], ys[:10], zs[:10])

    #ax.set_xlabel('x (mm)')
    #ax.set_ylabel('y (mm)')
    #ax.set_zlabel('z (mm)')

    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    xs = ak.flatten(sel_events.x)
    ys = ak.flatten(sel_events.y)
    zs = ak.flatten(sel_events.z)
    ax.scatter(xs[:10], ys[:10], zs[:10])

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    fig.savefig(f"{here}/results/hits_original.pdf")
    fig.savefig(f"{here}/results//hits_original.png")
    #plt.show()
