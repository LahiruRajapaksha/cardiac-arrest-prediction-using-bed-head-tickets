import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler


# Breaking the data set as (112,52,20)'
newPatient_list = np.array([])

def dataGroupingAndPadding(patient_data,time_step,no_of_features,no_of_patients):
    scalar = MinMaxScaler(feature_range=(0, 1))  # Scalar object is using in two places
    zeros = np.zeros(no_of_features)
    patient_groups = patient_data.groupby('ID')

    newPatient_list = []
    decision_tree_data = [] 
    for x in range(no_of_patients):
        patient_records = patient_groups.get_group((x + 1))
        patient_records = np.array(patient_records)
        labels = patient_records[:, -1]
        decision_tree_data = np.append(decision_tree_data, patient_records[0])
        no_of_records = patient_records.shape[0]

        if (time_step - no_of_records) > 0:
            # Create a zero-filled array with the same number of features as the existing records (excluding the last column)
            zeros = np.zeros((time_step - no_of_records, patient_records.shape[1]-1))
            # Create an array with the last column (label) repeated for the zero-filled records
            labels_repeated = np.repeat(labels[-1], time_step - no_of_records)
            labels_repeated = np.expand_dims(labels_repeated, axis=1)
            # Create an array with the last column (label) repeated for the zero-filled records
            padded_records = np.concatenate((np.hstack((zeros, labels_repeated)), patient_records))
        else:
            padded_records = patient_records
        
        # Select the last 'time_step' records
        padded_records = padded_records[-time_step:]

        padded_records = scalar.fit_transform(padded_records.astype('float32'))
        
        # Append the padded records for this patient to the list
        newPatient_list.append(padded_records)
  
    decision_tree_data = decision_tree_data.reshape(no_of_patients, no_of_features)
    newPatient_list = np.array(newPatient_list)
    return newPatient_list, decision_tree_data