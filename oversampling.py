import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

#read data from the csv file
patient_full_record_set = pd.read_csv('./DataCollection/FullPreparedDataSet/PreparedDataxlsx4.csv')
patient_first_record_set = pd.read_csv('./DataCollection/FullPreparedDataSet/PreparedDataxlsxFirstRowsNew.csv') 

def oversampling(full_record_set, first_record_set):
    lstm_data = full_record_set.iloc[:, :7]
    lstm_data_label = full_record_set.iloc[:, -1]
    decision_tree_data = first_record_set.iloc[:, 1:19]
    decision_tree_label = first_record_set.iloc[:, -1]
    smote = SMOTE()
    decision_tree_data, decision_tree_label = smote.fit_sample(decision_tree_data, decision_tree_label)
    lstm_data, lstm_data_label = smote.fit_sample(lstm_data, lstm_data_label)
    lstm_data_new = pd.concat([lstm_data, lstm_data_label], axis=1)
    decision_tree_data_new = pd.concat([decision_tree_data, decision_tree_label], axis=1)
    lstm_data_new.to_csv('OversampledDataLSTM', sep='\t', encoding='utf-8', index=False)
    return lstm_data_new, decision_tree_data_new