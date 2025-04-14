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

class TrackReconstructor:

    def __init__(self, layers):

        # Layers structure should be [[chipid0, z0], [chipid1, z1] ... ]
        # Convention: Layer 0 is the closest to the beam source
        self.layers = layers
        self.size = 16*1.3+0.3


    def reconstructTracks(self, events):

        ### Check that align run before
        # +
        ### Code to reconstruct tracks
        # For now just filtering events with one hit in each layer
        # Will need to return data with the same structure but only containing the hits of the tracks
        track_events = events[ak.all(events.nhits==1, axis=1)]
        return track_events


    def alignLayers(self, events):
        
        ## Misaligned (x,y,z)
        events['x'] = (2*events.row + 1) * 1.3/2 + 0.3  # 300um edge, 1.3mm wide pixels
        events['y'] = (30 - 2*events.col - 1) * 1.3/2 - 0.6  # 300um edge, 1.3mm wide pixels
        events['z'] = self.layers[2][1]*(events.chipid==(self.layers[2][0])) + self.layers[1][1]*(events.chipid==(self.layers[1][0]))

        ## Aligned dummy: For now just copy, no alignment applied
        events['x_corr'] = events['x']
        events['y_corr'] = events['y']
        return events


    def display(self, events):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x_surface = np.linspace(0.0, self.size, 50)
        y_surface = np.linspace(0.0, self.size, 50)
        x_surface, y_surface = np.meshgrid(x_surface, y_surface)
        color = ['blue', 'orange', 'green']
        #for _,l in enumerate(self.layers):
        #    z_surface = (l[1])
        #    ax.plot_surface(x_surface, y_surface, z_surface, alpha = 0.1, color = color[_])
        for ev in events:
            for l,layer in enumerate(self.layers):
                for x,y in zip(ev.x[ev['chipid']==layer[0]], ev.y[ev['chipid']==layer[0]]):
                    ax.scatter(x, y, layer[1], color = color[l], alpha = 0.01)
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("y coordinate")
        ax.set_zlabel("z coordinate")
        ax.set_xlim(0.0, self.size)
        ax.set_ylim(0.0, self.size)
        ax.set_zlim(self.layers[-1][1], 0.0)
        fig.savefig("eventdisplay.png", dpi=1200)


## Test code

if __name__=="__main__":

    ### Read events
    all_events = []
    read_files = 0
    for i in range(5707, 6106):
        in_file = f"/eos/user/f/fernance/ETL/TestBeam/ETROC_output/{i}_merged.json"
        if os.path.isfile(in_file):
            read_files += 1
            with open(in_file, "r") as f:
                all_events.append(ak.from_json(json.load(f)))
        else:
            print(f'Missing file: {in_file}')
    events_raw = ak.concatenate(all_events)
    events = events_raw[ak.num(events_raw.row)==ak.num(events_raw.chipid)]

    ## Reconstruct tracks
    trackReconstructor = TrackReconstructor(layers = [[152, 0.0], [144, -61.], [148, -122.]])
    aligned_events = trackReconstructor.alignLayers(events)
    tracks = trackReconstructor.reconstructTracks(aligned_events)
    trackReconstructor.display(tracks)

    ## Save data (in the same dataformat)
    print(tracks)


