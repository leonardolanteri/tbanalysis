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


def save_data(final_hist, output_data_path, filename, make_json=False):
    """
    final_hist is the hist object but projected to 1d
    output_data_path
        |_____filename
    example:
    """
    ensure_dir_exists(output_data_path)

    full_path = os.path.join(output_data_path, filename)
    base_path, _ = os.path.splitext(full_path) #if file extension given get rid of it
    #add it to json based on same structure as files!
    if make_json:
        #for heatmaps and ie the 16x16 matrices
        with open(base_path+'.json', 'w') as f:
            #if this fials try adaptiing numpy encoder and following the isinstance example!
            json.dump(final_hist, f, cls=NumpyEncoder)  #using the NpEncoder from previous example       
    else:
        with open(base_path + '.pkl', "wb") as f:
            #pick might be fast with this, according to docs!
            pickle.dump(final_hist, f)


class ParquetManager:
    """
    Saves and Loads large awkard arrays that are analyzed in chunks
    """
    def __init__(self, output_dir, filename):
        self._file_counter = 0
        self.filename = filename
        self.output_dir = output_dir
        if '.' in self.filename:
            raise ValueError("Please remove any . or the file extension")
        self.final_output_file = None


    def add_events(self, events: ak.Array):
        ak.to_parquet(events, os.path.join(self.output_dir,f"{self.filename}_{self._file_counter}.parquet"))
        self._file_counter += 1

    def save(self):
        """
        When it saves it concatenates all the parquets from each chunk in run_chunker to one parquet!
        """
        file_group = []
        for j in range(self._file_counter):
            partial_file = os.path.join(self.output_dir,f"{self.filename}_{j}.parquet")
            file_group.append(ak.from_parquet(partial_file))
            os.remove(partial_file)

        self.combined_collection = ak.concatenate(file_group)
        self.final_output_file = os.path.join(self.output_dir, f'{self.filename}.parquet')
        ak.to_parquet(self.combined_collection, self.final_output_file)
        self._file_counter = 0


    def load(self):
        if self.final_output_file is None:
            raise ValueError("Cannot load non saved parquet file")
        if os.path.isfile(self.final_output_file):
            return ak.from_parquet(self.final_output_file)

    def overwrite_with(self, new_events: ak.Array):
        """
        Takes a new set of events and overwrites the currently saved one. Can only be done after save. 
        """
        if self.final_output_file is None:
            if self._file_counter > 0:
                #partially loaded parquet manager, then just save it
                self.save()
            else:
                raise ValueError("Cannot overwrite parquet if no events were added")
        if os.path.isfile(self.final_output_file):
            ak.to_parquet(new_events, self.final_output_file)
        else:
            #if file does not exist, throw an error
            raise FileNotFoundError(f"File could not be found for parquet manager: {self.final_output_file}")     
