import os
import json
import pickle
import awkward as ak
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    # example from https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def ensure_dir_exists(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)



def save_json(final_hist, output_data_path, filename):
    """
    Saves jsons with numpy types
    output_data_path
        |_____filename
    """
    ensure_dir_exists(output_data_path)

    full_path = os.path.join(output_data_path, filename)
    base_path, _ = os.path.splitext(full_path) #if file extension given get rid of it
    #add it to json based on same structure as files!
    #for heatmaps and ie the 16x16 matrices
    with open(base_path+'.json', 'w') as f:
        #if this fials try adaptiing numpy encoder and following the isinstance example!
        json.dump(final_hist, f, cls=NumpyEncoder)  #using the NpEncoder from previous example  