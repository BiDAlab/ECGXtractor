import numpy as np
import tqdm
import pandas as pd
import neurokit2 as nk
import os
import json


segment_back = 160
segment_length = 400
consecutive_heartbeats = 6  # -1 for template, 1 for single hb, otherwise specify the length of cons. blocks (es. 10)
best_number = 5  # default 5
base_path = "datasets/ecg-id/preproc_csv"
base_path_segment = "datasets/ecg-id"


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def build_advanced_segment(sn, best_n):
    import operator
    avg_segment = np.mean(sn, axis=0)

    distance = [np.linalg.norm(s - avg_segment) for s in sn]
    idx_distance = list(enumerate(distance))
    top_n = [t[0] for t in sorted(idx_distance, key=operator.itemgetter(1))[:best_n]]

    segments_selected = [s for i, s in enumerate(sn) if i in top_n]
    result = np.mean(segments_selected, axis=0)

    return result


with open("datasets/ecg-id/r_peaks.json", 'r') as fid:
    dict_r_peaks = [json.loads(l) for l in fid][0]

for k, v in tqdm.tqdm(dict_r_peaks.items()):
    patient = k[:2]
    sample = k[3:-4]

    if not os.path.exists(base_path_segment + "/" + patient):
        os.makedirs(base_path_segment + "/" + patient)
    if not os.path.exists(base_path_segment + "/" + patient + "/" + sample):
        os.makedirs(base_path_segment + "/" + patient + "/" + sample)

    filename = base_path + "/" + k
    ecg = pd.read_csv(filename, header=None)
    ecg = ecg.to_numpy()

    segments = [ecg[r - segment_back:r + segment_length - segment_back, :] for r in v[1:-1]]
    segments = [s for s in segments if s.shape == (400, 1)]

    if consecutive_heartbeats < 0:
        cons_segments = [segments]
    else:
        cons_segments = [segments[i:i + consecutive_heartbeats] for i in range(0, len(segments) -
                                                                   consecutive_heartbeats + 1, consecutive_heartbeats)]

    for i, sn in tqdm.tqdm(enumerate(cons_segments)):
        if not consecutive_heartbeats == 1:
            result = build_advanced_segment(sn, best_number)
        else:
            result = sn[0]
        df = pd.DataFrame(result)
        word = 'template' if consecutive_heartbeats < 0 else 'hb' if consecutive_heartbeats == 1 else 'ss'
        df.to_csv(base_path_segment + "/" + patient + "/" + sample + "/" + word + "_" +
                  "{:03n}".format(i) + ".csv", index=False, header=False)