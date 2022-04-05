import loader
import tqdm
import os
import multiprocessing as mp
import math


results = []


def load_slice_of_data(i, data, keys, path, in_step, fin_step):
    data_dict = {k: [] for k in keys}
    missing = 0

    for d in tqdm.tqdm(data):
        found = False
        for p in path:
            if os.path.isfile(os.path.join(p, d['ecg'])):
                ecg = loader.load_ecg(os.path.join(p, d['ecg']), in_step, fin_step)
                found = True
                break

        if found:
            data_dict['ecg'].append(ecg)
            for k in keys:
                if k == 'ecg': continue
                data_dict[k].append(d[k])

        else:
            missing += 1

    return (i, data_dict, missing)


def collect_result(result):
    global results
    results.append(result)


def load_dataset_parallel(data, keys, path, in_step, fin_step):
    global results
    results = []
    pool = mp.Pool(mp.cpu_count())
    size = math.ceil(len(data)/mp.cpu_count())
    for i in range(mp.cpu_count()):
        pool.apply_async(load_slice_of_data, args=(i, data[i*size:(i+1)*size], keys, path, in_step, fin_step),
                         callback=collect_result)
    pool.close()
    pool.join()

    final_dict = {k: [] for k in keys}
    final_missing = 0
    results.sort(key=lambda x: x[0])
    for i in range(len(results)):
        for k in keys:
            final_dict[k] = final_dict[k] + results[i][1][k]
        final_missing += results[i][2]

    print("Missing data: ", final_missing)
    return final_dict






