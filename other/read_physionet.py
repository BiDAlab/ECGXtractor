import wfdb
import numpy as np
import pandas as pd

# read a .dat file (fs=1000) and write a .csv file (fs=500)
def get_initial_csv(ecg_name, patient, files_path):  
    # ecg_name without extension
    record = wfdb.rdsamp(files_path + '/initial_physionet/patient'+patient+'/'+ecg_name)
    record = np.asarray(record[0])
    record = record[0:len(record):2, :12]
    df = pd.DataFrame(record)
    df.to_csv(files_path + '/initial_csv/'+patient+ecg_name+'.csv', index=False, header=False)
