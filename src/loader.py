import json
import numpy as np
import os
import random
import tqdm
import pandas as pd
import tensorflow as tf
import util
import parallel_load


def load_dataset(data_json, path, in_step, fin_step, parallel):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    keys = list(data[0].keys())

    if parallel:  return parallel_load.load_dataset_parallel(data, keys, path, in_step, fin_step)

    data_dict = {k: [] for k in keys}
    missing = 0

    for d in tqdm.tqdm(data):
        found = False
        for p in path:
            if os.path.isfile(os.path.join(p, d['ecg'])):
                ecg = load_ecg(os.path.join(p, d['ecg']), in_step, fin_step)
                found = True
                break

        if found:
            data_dict['ecg'].append(ecg)
            for k in keys:
                if k == 'ecg': continue
                data_dict[k].append(d[k])

        else: missing += 1

    print("Missing data: ", missing)
    return data_dict


def fix_data_norm(data):
    fixed_data = []
    interval = 2
    for i in range(data.shape[1]):
        d = data[:, i]
        data_interval = max(d) - min(d)
        ratio = (interval / data_interval) if data_interval > 0 else 1
        fixed_data.append(d * ratio)
    fixed_data = np.array(fixed_data)
    fixed_data = np.swapaxes(fixed_data, 0, 1)
    return fixed_data


def load_ecg(record, initial=None, final=None):
    ecg = pd.read_csv(record, header=None)
    if initial is not None and final is not None:
        ecg = ecg.iloc[initial:final, :]
    ecg = ecg.to_numpy()
    ecg = fix_data_norm(ecg)
    ecg = np.expand_dims(ecg, axis=-1)  # Create feature dimension
    return ecg


def prepare_dataset(ds_file, params):
    initial_step = params['initial_step'] if 'initial_step' in params.keys() else None
    final_step = params['final_step'] if 'final_step' in params.keys() else None
    lead_i = params.get("lead_i", False)
    ds = load_dataset(ds_file, params['data_path'], initial_step, final_step, params['parallel'])

    extract_features = params.get("extract_features", False)
    if extract_features:
        ds['ecg'] = util.feature_extraction(params['extractor_path'], params['output_layer'], ds['ecg'])
        if lead_i:
            temp = []
            for arr in ds['ecg']:
                arr2 = arr.reshape((25, 12, 2))
                arr2 = arr2[:, 0, :]
                arr2 = arr2.reshape((-1,))
                temp.append(arr2)
            temp = np.array(temp)
            temp = temp.reshape(temp.shape + (1, 1))
            ds['ecg'] = temp

    return ds


def data_generator(batch_size, preproc, data, in_json_keys, ou_json_keys):
    random.seed(2)
    num_keys, cat_keys = division_keys(data)
    num_examples = len(data['ecg'])

    examples = []
    for i in range(len(data['ecg'])):
        examples.append({k: data[k][i] for k in data.keys()})
    batches = util.shuffle_examples_batch(examples, num_examples, batch_size)

    while True:
        for batch in batches:

            preproc_input = []
            for in_j in in_json_keys:
                arr = [b[in_j] for b in batch]
                arr = preproc.process_numerical(arr, in_j) if in_j in num_keys else \
                    preproc.process_categorical(arr, in_j)
                preproc_input.append(arr)

            preproc_output = []
            for ou_j in ou_json_keys:
                arr = [b[ou_j] for b in batch]
                if ou_j in num_keys:
                    arr = preproc.process_numerical(arr, ou_j) if ou_j == "ecg" else np.array(arr).astype(np.float)
                else:
                    arr = preproc.process_categorical(arr, ou_j)
                preproc_output.append(arr)

            if len(preproc_output) == 1:  yield preproc_input, preproc_output[0]
            else: yield preproc_input, preproc_output


def match_sample_generator(batch_size, examples, shuffle=True):
    num_examples = len(examples)
    batches = util.shuffle_examples_batch(examples, num_examples, batch_size, True, shuffle)

    while True:
        for batch in batches:
            x = np.array([b['match_sample'] for b in batch])
            y = tf.keras.utils.to_categorical([b['is_matching'] for b in batch], num_classes=2)
            yield x, y


class Preproc:

    def __init__(self, data):
        num_keys, cat_keys = division_keys(data)

        self.mean={};  self.std={}
        for nk in num_keys:
            self.mean[nk], self.std[nk] = compute_mean_std(data[nk])

        self.classes={}; self.int_to_class={}; self.class_to_int={}
        for ck in cat_keys:
            # if ck == "user": continue
            self.classes[ck] = sorted(set(l for l in data[ck]))
            self.int_to_class[ck] = dict(zip(range(len(self.classes[ck])), self.classes[ck]))
            self.class_to_int[ck] = {c: i for i, c in self.int_to_class[ck].items()}

    def process_numerical(self, x, key):
        x = np.array(x).astype(np.float)
        x = (x - self.mean[key]) / self.std[key]
        return x

    def process_categorical(self, y, key):
        y = [self.class_to_int[key][c] for c in y]
        y = tf.keras.utils.to_categorical(
                y, num_classes=len(self.classes[key]))
        return y


def division_keys(data):
    num_keys = [];  cat_keys = []
    for k in data.keys():
        if k == "user" or k == "usercode":  cat_keys.append(k)
        elif type(data[k][0]) == int or type(data[k][0]) == float:  num_keys.append(k)
        elif isinstance(data[k][0], (list, pd.core.series.Series, np.ndarray)):
            if str(data[k][0].dtype).find('float') >= 0 or str(data[k][0].dtype).find('int') >= 0:  num_keys.append(k)
            else: cat_keys.append(k)
        elif isinstance(data[k][0], str) and data[k][0].replace('.', '', 1).isnumeric():  num_keys.append(k)
        else: cat_keys.append(k)
    return num_keys, cat_keys


def compute_mean_std(x):
    x = np.hstack(x)
    try:
        x = np.array(x).astype(np.float)
        return (np.mean(x).astype(np.float32),
                np.std(x).astype(np.float32))

    except:
        import math
        size = 20000 * 12
        mean_acc = 0
        std_acc = 0
        n_acc = 0
        i = 0
        while i < x.shape[1]:
            x_temp = x[:, i:i+size, :]
            mean_temp = np.mean(x).astype(np.float32)
            std_temp = np.std(x).astype(np.float32)
            n_temp = x_temp.size
            mean_post = (mean_acc * n_acc + mean_temp * n_temp) / (n_acc + n_temp)
            partial_num = n_acc * std_acc**2 + n_temp * std_temp**2 + n_acc * (mean_acc - mean_post)**2 + n_temp * \
                  (mean_temp - mean_post)**2
            std_acc = math.sqrt(partial_num/(n_acc + n_temp))
            mean_acc = mean_post
            n_acc += n_temp
            i += size
        return mean_acc, std_acc
