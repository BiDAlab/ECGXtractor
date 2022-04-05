import os
import tqdm
import json
import random
random.seed(2)

base_path = "datasets/ecg-id/"


def get_list_of_patients(keyword):
    pats = [name for name in os.listdir(base_path) if not os.path.isfile(os.path.join(base_path, name))]
    pats = [p for p in pats if not p == '74']
    return [p for p in pats if not p == 'preproc_csv']


def get_peaks(pats):
    dict_peaks = {}
    for p in pats:
        dict_peaks[p] = {}
        path = base_path + p
        sessions = [name for name in os.listdir(path) if not os.path.isfile(os.path.join(path, name))]
        for s in sessions:
            path_session = base_path + p + "/" + s
            peaks = [s + '/' + name for name in os.listdir(path_session) if
                     os.path.isfile(os.path.join(path_session, name))]
            dict_peaks[p][s] = peaks
    return dict_peaks


def keep_peaks_sessions(dict_peaks, sessions, verification):
    # session is a string and can be: 'single', 'multi'
    # verification is a boolean
    kept_peaks = {}
    for k, v in dict_peaks.items():
        kept_peaks[k] = {}
        if sessions == 'single':
            single = v['rec_1' if len(v)==1 else 'rec_2']
            single = [s for s in single if not (s.split('/')[1][:8] == 'template' or s.split('/')[1][:2] == 'hb')]
            kept_peaks[k]['single'] = single
        elif sessions == 'multi' and not verification:
            assert(len(v) > 1)
            # shortest_len = min(map(len, v.values()))
            # sess_min_len = [k for k, v in v.items() if len(v) == shortest_len][0]
            train = []
            keys = list(v.keys())
            keys = [k[4:] for k in keys]
            keys = [k if len(k) == 2 else '0' + k for k in keys]
            keys = sorted(keys)
            keys = ['rec_' + (k if not k[0]=='0' else k[1]) for k in keys]
            for s in keys[:-1]:
                new_train = v[s]
                new_train = [nt for nt in new_train if not (nt.split('/')[1][:8] == 'template' or nt.split('/')[1][:2] == 'hb')]
                train += new_train
            kept_peaks[k]['train'] = train
            test = v[keys[-1]]
            test = [t for t in test if
                         not (t.split('/')[1][:8] == 'template' or t.split('/')[1][:2] == 'hb')]
            kept_peaks[k]['test'] = test
        elif sessions == 'multi' and verification:
            assert (len(v) > 1)
            template = v['rec_1']
            template = [t for t in template if t.split('/')[1][:8] == 'template']
            kept_peaks[k]['template'] = template
            sample = v['rec_2']
            sample = [s for s in sample if s.split('/')[1][:8] == 'template']
            kept_peaks[k]['sample'] = sample
    return kept_peaks


def get_data_from_users_for_verification(kept_peaks, user_list):
    data = []
    if 'single' in list(kept_peaks[list(kept_peaks.keys())[0]].keys()):
        for u in user_list:
            data = data + [{'ecg': vv, 'usercode': u} for vv in kept_peaks[u]['single']]
    elif 'template' in list(kept_peaks[list(kept_peaks.keys())[0]].keys()):
        for u in user_list:
            data = data + [{'ecg': kept_peaks[u]['template'][0], 'usercode': u, 'function': 'template'}]
            data = data + [{'ecg': kept_peaks[u]['sample'][0], 'usercode': u, 'function': 'sample'}]
    return data


def get_data_from_users_for_identification(kept_peaks, split_pct):
    train_ds = [];  val_ds = [];  test_ds = []
    if 'single' in list(kept_peaks[list(kept_peaks.keys())[0]].keys()):
        for u in list(kept_peaks.keys()):
            train_ds = train_ds + [{'ecg': vv, 'usercode': u} for vv in kept_peaks[u]['single'][:int(split_pct['train']*len(kept_peaks[u]['single']))]]
            val_ds = val_ds + [{'ecg': vv, 'usercode': u} for vv in kept_peaks[u]['single'][int(split_pct['train']*len(kept_peaks[u]['single'])):int((split_pct['train']+split_pct['val'])*len(kept_peaks[u]['single']))]]
            test_ds = test_ds + [{'ecg': vv, 'usercode': u} for vv in kept_peaks[u]['single'][int((split_pct['train']+split_pct['val'])*len(kept_peaks[u]['single'])):]]
    elif 'train' in list(kept_peaks[list(kept_peaks.keys())[0]].keys()):
        for u in list(kept_peaks.keys()):
            train = kept_peaks[u]['train']
            sessions = list(set([t.split('/')[0] for t in train]))
            for s in sessions:
                train_session = [t for t in train if s == t[:len(s)]]
                train_ds = train_ds + [{'ecg': vv, 'usercode': u} for vv in train_session[:int(split_pct['train']*len(train_session))]]
                val_ds = val_ds + [{'ecg': vv, 'usercode': u} for vv in train_session[int(split_pct['train']*len(train_session)):]]
            test_ds = test_ds + [{'ecg': vv, 'usercode': u} for vv in kept_peaks[u]['test']]
    return train_ds, val_ds, test_ds


def create_ds(kept_peaks, split_pct, mode):
    # split_pct is a dict with keys 'train', 'val', 'test' and the sum of values is 1.
    # mode is a string and can be: 'verification', 'identification'
    if mode == 'verification':
        users = list(kept_peaks.keys())
        random.shuffle(users)
        assert(split_pct['train'] + split_pct['val'] + split_pct['test'] == 1)
        train_users = users[:int(split_pct['train']*len(users))]
        val_users = users[int(split_pct['train']*len(users)):int((split_pct['train']+split_pct['val'])*len(users))]
        test_users = users[int((split_pct['train']+split_pct['val'])*len(users)):]
        training_ds = get_data_from_users_for_verification(kept_peaks, train_users)
        validation_ds = get_data_from_users_for_verification(kept_peaks, val_users)
        testing_ds = get_data_from_users_for_verification(kept_peaks, test_users)
    elif mode == 'identification':
        if 'train' in list(kept_peaks[list(kept_peaks.keys())[0]].keys()):
            assert(split_pct['train'] + split_pct['val'] == 1)
        elif 'single' in list(kept_peaks[list(kept_peaks.keys())[0]].keys()):
            assert (split_pct['train'] + split_pct['val'] + split_pct['test'] == 1)
        training_ds, validation_ds, testing_ds = get_data_from_users_for_identification(kept_peaks, split_pct)

    return training_ds, validation_ds, testing_ds


def write_json(ds, fn):
    with open('datasets/' + fn, 'w') as fid:
        for tr in ds:
            tr['ecg'] = tr['usercode'] + '/' + tr['ecg']
            json.dump(tr, fid)
            fid.write('\n')


def do_everything(keyword, sessions, split_pct, mode, ds_filename):
    patients = get_list_of_patients(keyword)
    peaks = get_peaks(patients)
    kept_peaks = keep_peaks_sessions(peaks, sessions, mode == 'verification')
    train_ds, val_ds, test_ds = create_ds(kept_peaks, split_pct, mode)
    # pass
    for ds, fn in zip([train_ds, val_ds, test_ds], [name + '_' + ds_filename for name in ['train', 'val', 'test']]):
        if len(ds) > 0:
            write_json(ds, fn)


classic_split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
autoencoder_split = {'train': 0.8, 'val': 0.2, 'test': 0}
small_user_set_split = {'train': 0.5, 'val': 0.1, 'test': 0.4}
only_train_split = {'train': 1, 'val': 0, 'test': 0}
only_test_split = {'train': 0, 'val': 0, 'test': 1}


do_everything('', 'single', only_test_split, 'verification', 'ecg-id_single_verification.json')
do_everything('', 'multi', autoencoder_split, 'identification', 'ecg-id_multi_identification.json')