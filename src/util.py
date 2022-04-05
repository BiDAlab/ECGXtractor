import os
import time
import random
import numpy as np
import _pickle as pickle
import pandas as pd
import tensorflow as tf
import loader


def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_filename_for_saving(save_dir, metrics):

    if len(metrics) > 0:
        params = np.array(["val_loss", "val_" + metrics[0], "epoch", "loss", metrics[0]])
    else:
        params = np.array(["val_loss", "epoch", "loss"])
    out_name = ""
    for p in params:
        if p == "epoch":
            out_name = out_name + "{" + p + ":03d}-"
        else:
            out_name = out_name + "{" + p + ":.3f}-"

    out_name = out_name[:-1] + ".hdf5"

    return os.path.join(save_dir, out_name)


def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'rb') as fid:
        preproc = pickle.load(fid)
    return preproc


def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'wb') as fid:
        pickle.dump(preproc, fid)


def save_train_loss(history, filepathname, metrics):
    dict_df = {'epoch': [i+1 for i in range(len(history.history['loss']))],
               'train_loss': history.history['loss'],
               'val_loss': history.history['val_loss']}
    if len(metrics) > 0:
        for m in metrics:
            dict_df['train_' + m] = history.history[m]
            dict_df['val_' + m] = history.history['val_' + m]

    metrics = pd.DataFrame(dict_df)
    metrics.to_csv(filepathname, index=False)


def shuffle_examples_batch(examples, num_examples, batch_size, shuffle_before=True, shuffle=True):
    random.seed(2)
    if shuffle_before and shuffle:
        random.shuffle(examples)
    end = num_examples - batch_size + 1
    batches = [examples[i:i + batch_size]
               for i in range(0, end, batch_size)]
    if (not shuffle_before) and shuffle:
        for b in batches:
            random.shuffle(b)
    return batches


def feature_extraction(model_path, output_layer, ecgs):
    import tensorflow_addons as tfa
    tfa.losses.TripletSemiHardLoss()

    preproc_old = load(os.path.dirname(model_path))
    dependencies = {'loss': tfa.losses.TripletSemiHardLoss()}
    model = tf.keras.models.load_model(model_path, custom_objects=dependencies)
    if not output_layer == "sequential":
        out = model.get_layer(output_layer).output
        model = tf.keras.Model(model.input, out)

    data_preproc = np.array([(e - preproc_old.mean['ecg']) / preproc_old.std['ecg'] for e in ecgs])
    if data_preproc.shape[2] == 1:
        data_preproc = np.pad(data_preproc, ((0, 0), (0, 0), (0, 11), (0, 0)))
    features = model.predict(data_preproc)
    features = features.reshape(features.shape + (1, 1))
    return features


def get_model_finetun_identification(existing_model, lead_i, out_dim):
    model = tf.keras.Sequential(name='ft_model')
    model.add(tf.keras.Input(shape=(50 if lead_i else 50*12, 1, 1)))
    seq = existing_model.get_layer('sequential')
    for l in seq.layers:
        l.trainable = False

    seq.add(tf.keras.layers.Dropout(0.4, name='new_dropout'))
    seq.add(tf.keras.layers.Dense(out_dim, activation='softmax', name='dense_new_2'))
    model.add(seq)

    for layer in model.layers:
        if layer.name == 'sequential':
            for layer2 in layer.layers:
                layer2._name = layer2.name + str('_new')
        layer._name = layer.name + str('_new')

    return model


def helper_pair_already_selected(pairs_list, new_pair):
    for pl in pairs_list:
        if pl.intersection(new_pair) == new_pair:
            return True
    return False


def helper_comb_available(pairs_list, len_user_ecgs):
    tot = 1
    for i in range(2, len_user_ecgs):
        tot += i
    return len(pairs_list) < tot


def create_generic_match_sample(preproc, data, positive, negative_multiplier, seed_=2):
    import tqdm
    import collections

    template_sample = False
    if 'function' in list(data.keys()):
        template_sample = 'template' in list(data['function'])
    usercode = 'usercode' in list(data.keys())
    dict_data_users = collections.Counter(data['user' if not usercode else 'usercode'])

    random.seed(seed_)
    users = list(dict_data_users.keys())
    random.shuffle(users)
    np_users = np.array(data['user' if not usercode else 'usercode'])
    np_ecgs = np.array(data['ecg'])
    if template_sample:
        np_functions = np.array(data['function'])
    examples = []

    for ix_u in tqdm.tqdm(range(len(users))):
        u = users[ix_u]
        user_index = np.where(np_users == u)
        user_ecgs = np_ecgs[user_index]

        if template_sample:
            user_functions = np_functions[user_index]
            user_templates = user_ecgs[np.where(user_functions == 'template')]
            sample_index = [i for i, f in enumerate(user_functions) if 'sample' in f]
            user_samples = user_ecgs[sample_index]

        if template_sample:
            if len(user_templates) == 0:  continue
        else:
            if len(user_ecgs) <= 1:  continue

        if not template_sample or (template_sample and len(user_samples) > 0):
            pairs_list = []
            for i in range(positive):
                first_index = random.randint(0, len(user_ecgs) - 1) if not template_sample else \
                    random.randint(0, len(user_templates) - 1)
                second_index = random.randint(0, len(user_ecgs) - 1) if not template_sample else i % len(user_samples)

                if (not template_sample) and first_index == second_index:
                    second_index = (second_index + 1) % len(user_ecgs)
                if (not template_sample):
                    new_pair = set([first_index, second_index])

                    change_pair = helper_pair_already_selected(pairs_list, new_pair)
                    more_comb_available = helper_comb_available(pairs_list, len(user_ecgs))
                    while change_pair and more_comb_available:
                        prob = random.randint(0, 1)
                        if prob:  first_index = (first_index + 1) % len(user_ecgs)
                        else:  second_index = (second_index + 1) % len(user_ecgs)
                        if first_index == second_index:
                            second_index = (second_index + 1) % len(user_ecgs)
                        new_pair = set([first_index, second_index])
                        change_pair = helper_pair_already_selected(pairs_list, new_pair)
                    pairs_list.append(new_pair)

                first = preproc_reshape_feature_vector(preproc, user_ecgs[first_index] if not template_sample else
                                                       user_templates[first_index])
                second = preproc_reshape_feature_vector(preproc, user_ecgs[second_index] if not template_sample else
                                                        user_samples[second_index])  # random.randint(0, len(user_samples) - 1)])
                examples.append({'match_sample': np.concatenate([first, second], axis=1).reshape((-1, 2, 1)),
                                 'is_matching': 1,
                                 'template': u,
                                 'sample': u})
        i = 0
        while i < (int(positive * negative_multiplier)):
            u2 = users[(ix_u + i + 1) % len(users)]
            if u == u2:
                i += 1
                continue
            user_index2 = np.where(np_users == u2)
            user_ecgs2 = np_ecgs[user_index2]

            if template_sample:
                user_functions2 = np_functions[user_index2]
                sample_index2 = [i for i, f in enumerate(user_functions2) if 'sample' in f]
                user_samples2 = user_ecgs2[sample_index2]

            if template_sample and len(user_samples2) == 0:
                i += 1
                continue
            elif not template_sample and len(user_ecgs2) == 0:
                i += 1
                continue

            first = preproc_reshape_feature_vector(preproc, user_ecgs[random.randint(0, len(user_ecgs) - 1)] if not template_sample else
                                                   user_templates[random.randint(0, len(user_templates) - 1)])
            second = preproc_reshape_feature_vector(preproc, user_ecgs2[random.randint(0, len(user_ecgs2) - 1)] if not template_sample else
                                                    user_samples2[random.randint(0, len(user_samples2) - 1)])
            examples.append({'match_sample': np.concatenate([first, second], axis=1).reshape((-1, 2, 1)),
                             'is_matching': 0,
                             'template': u,
                             'sample': u2})
            i += 1

    return examples


def fix_data(data, max_size=False):
    # assert('function' not in list(data.keys()))
    usercode = 'usercode' in list(data.keys())
    import collections
    dict_data_users = collections.Counter(data['user' if not usercode else 'usercode'])
    min_user_ecgs = min([v for _, v in dict_data_users.items()])
    if max_size:
        max_user_ecgs = max([v for _, v in dict_data_users.items()])

    np.random.seed(2)
    users = list(dict_data_users.keys())
    np_users = np.array(data['user' if not usercode else 'usercode'])
    np_ecgs = np.array(data['ecg'])
    fixed_users = []
    fixed_ecgs = []
    for u in users:
        user_index = np.where(np_users == u)
        user_ecgs = np_ecgs[user_index]
        np.random.shuffle(user_ecgs)
        if max_size:
            added = 0
            while added < max_user_ecgs:
                fixed_ecgs += user_ecgs[:max_user_ecgs - added].tolist()
                fixed_users += np_users[user_index][:max_user_ecgs - added].tolist()
                added += min(len(user_ecgs), max_user_ecgs - added)
        else:
            fixed_ecgs = fixed_ecgs + user_ecgs[:min_user_ecgs].tolist()
            fixed_users = fixed_users + np_users[user_index][:min_user_ecgs].tolist()
    return {'ecg': np.array(fixed_ecgs), 'user' if not usercode else 'usercode': np.array(fixed_users)}


def preproc_reshape_feature_vector(preproc, data):
    if preproc is not None:
        data = (data - preproc.mean['ecg']) / preproc.std['ecg']
    data = data.reshape((-1, 1))
    return data


def add_compile(model, **params):
    from tensorflow.keras import optimizers, losses, metrics
    model.summary()

    optimizer = optimizers.Adam(
        learning_rate=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    mets = []
    if "accuracy" in params['metrics']:  mets.append('accuracy')
    if "auc" in params['metrics']:  mets.append(metrics.AUC())
    if "mean_absolute_error" in params['metrics']: mets.append(metrics.MeanAbsoluteError())

    ls = losses.MeanSquaredError() if params['experiment'].find("autoencoder") >= 0 else\
        losses.CategoricalCrossentropy()

    model.compile(loss=ls,
                  optimizer=optimizer,
                  metrics=mets)

    return model


def get_best_bs(test_len, max_bs):
    temp_bs = max_bs
    while temp_bs > 0:
        if test_len % temp_bs == 0:
            return temp_bs
        else:
            temp_bs -= 1


def write_predictions(filename, y_pred, y_true, template_users, sample_users):
    results_df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true, 'template_users': template_users,
                               'sample_users': sample_users})
    results_df.to_csv(filename)


def get_model(**params):
    import network_authentication_siamese_CNN
    import network_autoencoder

    experiment = params['experiment']
    if experiment.find('verification') >= 0:
        model = network_authentication_siamese_CNN.build_network(**params)

    elif experiment.find('autoencoder') >= 0:
        model = network_autoencoder.build_network(**params)

    return model


def verification_sample_creation(train_, val_, preproc, **params):

    batch_size = params.get("batch_size", 16)
    positive_samples = params.get('positive_samples', 100)
    negative_multiplier = params.get('negative_multiplier', 5)
    train_examples = create_generic_match_sample(preproc, train_, positive_samples, negative_multiplier)
    train_gen = loader.match_sample_generator(batch_size, train_examples)
    train_len = len(train_examples)
    val_examples = create_generic_match_sample(preproc, val_, positive_samples, negative_multiplier)
    val_gen = loader.match_sample_generator(batch_size, val_examples, shuffle=False)
    val_len = len(val_examples)

    return train_gen, val_gen, train_len, val_len
