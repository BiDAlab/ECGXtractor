# ECGXtractor

ECGXtractor is a Python library that allows to investigate the topic of ECG biometric recognition. In particular, it executes the experiments described in the article specified at the end of this page.

## Article

Melzi, Pietro, Ruben Tolosana, and Ruben Vera-Rodriguez. "Ecg biometric recognition: Review, system proposal, and benchmark evaluation." *IEEE Access*, 2023.
[Link](https://ieeexplore.ieee.org/abstract/document/10043674)

## Setup

In the file [requirements.txt](requirements.txt) you can find the list of dependencies required by this library. Run the following commands in Windows to set up a suitable virtual environment (replace \<env\> with the mane of the environment). *pip* and *virtualenv* libraries are required to run the following commands. 

```bash
python -m venv <env>
```
```bash
.\<env>\Scripts\activate
```
```bash
python -m pip install -r requirements.txt
```

## Comparisons and Pretrained Models

In [this](comparisons) folder you can find the list of genuine and impostor comparison pairs evaluated in the different executions of the experiments included in our paper, related to the task of verification.

We also provide [here](https://bidalab.eps.uam.es/static/pietro/saved.rar) the weights of our pretrained models (saved.rar). They are used to perform the experiments described in our paper.

## Data preparation

The following instructions refer to PTB database. Similarly, you can also run experiments with ECG-ID and CYBHI databases.

### ECG signals

Download [here](https://bidalab.eps.uam.es/static/pietro/ptb.rar) ptb.rar and extract it in datasets\ptb. It contains:
* 549 12-Lead ECG signals from PTB database, re-sampled at 500 Hz, and filtered
* the list of healthy subjects included in the PTB database;
* the list of subjects for which multiple ECG signals are contained in PTB;
* for each ECG signal, the list of time instants corresponding to r-peaks.

### ECG segments

Run the file [build_segments.py](src/ptb/build_segments.py) after changing in the code the parameter *consecutive_heartbeats* to:
* -1 to generate a template from each signal;
* 1 to extract all the single heartbeats obtainable from each signal;
* *n* > 1 to generate summary samples from each signal, with *n* the number of consecutive single heartbeats considered for the summary sample. To reproduce experiments of the article, set *n* = 10.

```bash
python src\ptb\build_segments_ptb.py
```

## Files required to run experiments

### Datasets files

In folder [datasets](datasets), we provide the dataset files used in the different experiments involving PTB:
* single-session verification, with the set of 52 healthy subjects: [test_healthy_single_verification](datasets/test_healthy_single_verification.json);
* single-session verification, with the set of 113 subjects: [test_multi_single_verification](datasets/test_multi_single_verification.json);
* multi-session verification, with the set of 113 subjects provided with multiple ECG signals: [test_multi_multi_verification](datasets/test_multi_multi_verification.json);
* single-session identification, with the set of 52 healthy subjects: [train_healthy_single_identification](datasets/train_healthy_single_identification.json), [val_healthy_single_identification](datasets/val_healthy_single_identification.json), [test_healthy_single_identification](datasets/test_healthy_single_identification.json);
* single-session identification, with the set of 113 subjects: [train_multi_single_identification](datasets/train_multi_single_identification.json), [val_multi_single_identification](datasets/val_multi_single_identification.json), [test_multi_single_identification](datasets/test_multi_single_identification.json);
* multi-session identification, with the set of 113 subjects provided with multiple ECG signals: [train_multi_multi_identification](datasets/train_multi_multi_identification.json), [val_multi_multi_identification](datasets/val_multi_multi_identification.json), [test_multi_multi_identification](datasets/test_multi_multi_identification.json).

These files are obtained by running [create_dataset_ptb.py](src/ptb/create_dataset_ptb.py).

**Important:** the datasets files contain the list of files considered in the experiment. The specific samples used to train and evaluate the system will be generated during execution, with functions contained in the source code.

### Settings files

In folder [settings](settings), we provide the files containing the experimental settings considered when training the different models. As Autoencoder and verification networks are trained with the in-house database, which is not available, some fields of the json files are left empty (*i.e.*, *data_path*, *train*, *val*).

In [config_verification](settings/config_verification.json) you can change the following parameters before running experiments for the verification task:
* *lead_i*: true if only Lead I is considered, false if all the 12 Leads are considered.
* *positive_samples*: specify the number of genuine comparisons generated from each subject.
* *negative_multiplier*: the product *positive_samples* * *negative_multiplier* represents the number of impostor comparisons whose enrolment sample belongs to the same subject.
* *data_path* and *test*: by joining them you obtain the path of the dataset file considered for evaluation.

For the identification task, [config_identification](settings/config_identification.json) allows to change the following parameters, in addition to the previously described ones:
* *initial_weights*: specify the weight of the bottom layers of the identification model (not trainable).
* *individuals*: specify the output dimension of the classifier.
* *resample_max*: true to oversample, false to undersample. It is used in case the number of samples belonging to each subject is not equal in the training and validation datasets.

**Important:** some parameters, *i.e.,* *lead_i* and *initial_weight* must be coherent with the dataset and the network architecture selected, otherwise an error message will appear during the execution. 

## Execution

### Evaluation

To evaluate previously trained models for ECG biometric verification (or models trained for identification), run the following command:
```bash
python src\predict.py settings\config_verificaton.json --model_path <path of the saved model>
```

or
```bash
python src\predict.py settings\config_identification.json --model_path <path of the saved model>
```

### Training 

Additionally, if you want to train your model with your database and settings, you can run the following command:
```bash
python src\train.py settings\config_identification.json
```
### Saved models

After each training experiment, the folder *saved* will be created. It will contain the weights of your trained models.

## References

Melzi, Pietro, Ruben Tolosana, and Ruben Vera-Rodriguez. "Ecg biometric recognition: Review, system proposal, and benchmark evaluation." *IEEE Access*, 2023.
[Link](https://ieeexplore.ieee.org/abstract/document/10043674)

Please remember to reference the article on any work made public, whatever the form, based directly or indirectly on any part of ECGXtractor.

For further questions, please send an email to [pietro.melzi@uam.es](mailto:pietro.melzi@uam.es)
