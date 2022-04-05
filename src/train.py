import argparse
import json
import numpy as np
import tensorflow as tf
import os
import loader
import util
from shutil import copyfile


def train(params):
    np.random.seed(2)
    print("Loading training set...")
    train = loader.prepare_dataset(params['train'], params)
    print("Loading val set...")
    val = loader.prepare_dataset(params['val'], params)

    print("Building preprocessor...")
    preproc = loader.Preproc(train)

    print("Train size: " + str(len(train['ecg'])) + " examples.")
    print("Val size: " + str(len(val['ecg'])) + " examples.")

    save_dir = util.make_save_dir(params['base_save_dir'], params['experiment'])
    copyfile(params['filename'], save_dir + '\\' + params['filename'].replace('\\', '/').split('/')[-1])
    util.save(preproc, save_dir)

    if 'initial_weights' in params.keys() and len(params['initial_weights']) > 0:
        model = tf.keras.models.load_model(params['initial_weights'])
        trainable = params.get("trainable", True)
        if not trainable:
            for l in model.layers:  l.trainable = False
        if params['experiment'].find('identification') >= 0:
            model = util.get_model_finetun_identification(model, params['lead_i'], params['individuals'])
        model = util.add_compile(model, **params)

    else:
        model = util.get_model(**params)

    stopping = tf.keras.callbacks.EarlyStopping(
        patience=params['patience_es'],
        min_delta=params['min_delta_es'],
        monitor=params['monitor'])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=params['monitor'],
        min_delta=params['min_delta_rlr'],
        factor=params['factor'],
        patience=params['patience_rlr'],
        verbose=1,
        min_lr=params['min_lr'])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=util.get_filename_for_saving(save_dir, params['metrics']),
        save_best_only=params['save_best_only'])

    batch_size = params.get("batch_size", 16)

    if params['experiment'].find('verification') >= 0:
        train_gen, val_gen, train_len, val_len = util.verification_sample_creation(train, val, preproc, **params)
    else:
        input_keys_json = params['input_keys_json']
        output_keys_json = params['output_keys_json']
        if params['experiment'].find('identification') >= 0 or params['experiment'].find('finetun') >= 0:
            resample_max = params.get("resample_max", False)
            train = util.fix_data(train, resample_max)
            val = util.fix_data(val, resample_max)
        train_gen = loader.data_generator(batch_size, preproc, train, input_keys_json, output_keys_json)
        train_len = len(train['ecg'])
        val_gen = loader.data_generator(batch_size, preproc, val, input_keys_json, output_keys_json)
        val_len = len(val['ecg'])

    print("Len train examples: ", train_len, " - Len val examples: ", val_len)
    history = model.fit(
        train_gen,
        steps_per_epoch=int(train_len / batch_size),
        epochs=params['max_epochs'],
        validation_data=val_gen,
        validation_steps=int(val_len / batch_size),
        callbacks=[checkpointer, reduce_lr, stopping])

    util.save_train_loss(history, os.path.join(save_dir, params['csv_train_path'] + ".csv"), params['metrics'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    params['filename'] = str(args.config_file)
    train(params)