import pandas as pd
import numpy as np
import random
from oversampling import oversampling
from sklearn.preprocessing import MinMaxScaler


# Breaking the data set as (112,52,20)'
newPatient_list = np.array([])

def dataGroupingAndPreProcessing(patient_data,time_step,no_of_features,no_of_patients):
    scalar = MinMaxScaler(feature_range=(0, 1))  # Scalar object is using in two places
    
    zeros = np.zeros(no_of_features)
    patient_groups = patient_data.groupby('ID')
    newPatient_list = np.array([])
    decision_tree_data = [] 
    for x in range(no_of_patients):
        patient_records = patient_groups.get_group((x + 1))
        patient_records = np.array(patient_records)
        decision_tree_data = np.append(decision_tree_data, patient_records[0])
        no_of_records = patient_records.shape[0]
        if (time_step - no_of_records) > 0:
            for line in range(abs(time_step - no_of_records)):
                patient_records = np.insert(patient_records, 0, zeros)
        else:
            patient_records = patient_records.flatten()
        patient_records = patient_records[-time_step * no_of_features:]
        patient_records.resize(time_step, no_of_features) 
        patient_records = scalar.fit_transform(patient_records.astype('float32'))
        newPatient_list = np.append(newPatient_list, patient_records)
        newPatient_list = scalar.fit_transform(patient_records.astype('float32'))
        decision_tree_data = decision_tree_data.reshape(no_of_patients, no_of_features)

        newPatient_list = newPatient_list.reshape(no_of_patients, time_step, no_of_features)
        decision_tree_data = decision_tree_data.reshape(no_of_patients, no_of_features)

    return newPatient_list, decision_tree_data