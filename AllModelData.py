import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer

patient_record = pd.read_csv('/content/sample_data/TrainDead.csv')
# patient_record = pd.read_csv('/content/sample_data/TrainSurvived.csv')

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=patient_record)
metadata.update_column(
    column_name='ID',
    sdtype='id')
metadata.set_sequence_key(column_name='ID')
# print(metadata)#
# sequence = patient_record[patient_record['ID'] == 1]
# sequence

#
# print(patient_record['ID'].unique().size);


synthesizer = PARSynthesizer(
    metadata,
    context_columns=['Age','Gender','Alcoholic','Smoke','FHCD','TriageScore','Outcome'],
    epochs=90,
    verbose=True,
    )

synthesizer.fit(patient_record)

#
synthetic_data = synthesizer.sample(num_sequences=74)
synthetic_data.to_csv("/content/sample_data/FS74.csv")


-------------------------------------------------------------------
# Load the synthetic data
synthetic_data = pd.read_csv("/content/sample_data/FD59.csv")

# Modify the ID values
synthetic_data['ID'] = synthetic_data['ID'] + 90 - 0

# Save the modified synthetic data
synthetic_data.to_csv("/content/sample_data/FD59_modified.csv", index=False)
-------------------------------------------------------------------

from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot

quality_report = evaluate_quality(
    patient_record,
    synthetic_data,
    metadata)


fig = get_column_plot(
    real_data=patient_record,
    synthetic_data=synthetic_data,
    column_name='Outcome',
    metadata=metadata
)
    
fig.show()
-------------------------------------------------------------------
pip install table_evaluator
-------------------------------------------------------------------
from table_evaluator import load_data, TableEvaluator

real, fake = load_data('/content/sample_data/TrainDead.csv', "/content/sample_data/FD59.csv")
cat_cols = ['Outcome']
table_evaluator = TableEvaluator(real, fake, cat_cols=cat_cols)
table_evaluator.visual_evaluation()
-------------------------------------------------------------------
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
-------------------------------------------------------------------
#caridac-arrest prediction model

import pandas as pd
import numpy as np
import random
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import argparse
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_curve

epocs = 100
learningRate = 0.01
batchSize = 10
lstmNodes = 2

 ## Dataset can be found in https://zenodo.org/record/7603772
 ## Experiment data set
patients_training_data = pd.read_csv('/content/sample_data/ExperimentNo1Data.csv')


patients_test_data = pd.read_csv('/content/sample_data/Test10%.csv') 
patients_validation_data = pd.read_csv('/content/sample_data/Validate10%.csv') 

time_step = int(patients_training_data.groupby('ID').count().mean()[0])
no_of_patients = patients_training_data['ID'].max()
no_of_test_patients = patients_test_data['ID'].max()
no_of_validation_patients = patients_validation_data['ID'].max()

training_LSTM_data, training_decisionTree_data = dataGroupingAndPadding(patients_training_data,time_step,20,no_of_patients)
testing_LSTM_data, testing_decisionTree_data = dataGroupingAndPadding(patients_test_data, time_step, 20,no_of_test_patients)
validation_LSTM_data, validation_decisionTree_data = dataGroupingAndPadding(patients_validation_data,time_step,20,no_of_validation_patients)
print(validation_LSTM_data.shape)

# Shuffle the data set in the same order
training_LSTM_data = training_LSTM_data.tolist()
training_decisionTree_data = training_decisionTree_data.tolist()
shuffleDataSet = list(zip(training_LSTM_data, training_decisionTree_data))
random.shuffle(shuffleDataSet)
training_LSTM_data, training_decisionTree_data = zip(*shuffleDataSet)
training_LSTM_data = np.asarray(training_LSTM_data)
training_decisionTree_data = np.asarray(training_decisionTree_data)

# Prepare data set for LSTM model
lstm_patient_training_data = training_LSTM_data[:, :, 1:7]
lstm_patient_training_data_label = training_LSTM_data[:, :, -1].reshape(no_of_patients, time_step, 1)

# Test data set for LSTM model
lstm_patient_test_data = testing_LSTM_data[:, :, 1:7]
lstm_patient_test_data_label =testing_LSTM_data[:, :, -1].reshape(no_of_test_patients, time_step, 1)

# Validation data set for LSTM model
lstm_validation_data =validation_LSTM_data[:, :, 1:7]
lstm_validation_data_label =validation_LSTM_data[:, :, -1].reshape(no_of_validation_patients,time_step,1)


# Prepare data set for Decision Tree
d_tree_training_data = training_decisionTree_data[:, 7:19]
d_tree_training_data_label = np.reshape(training_decisionTree_data[:, -1], (no_of_patients, 1))

# Defining the model
model = Sequential()
model.add(LSTM(lstmNodes, return_sequences=True, input_shape=(time_step, 6)))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(Adam(learning_rate=learningRate), loss='binary_crossentropy', metrics=['acc'])  # lr0.005 0r 0.001
model.summary()

# Fit the model LSTM
history = model.fit(lstm_patient_training_data, lstm_patient_training_data_label,
                    shuffle=False, validation_data=(lstm_validation_data,lstm_validation_data_label), batch_size=batchSize, epochs=epocs)
# acc = np.round(history.history['acc'][epocs-1], 2)
# val_acc = np.round(history.history['val_acc'][epocs-1], 2)

# print('**********Summary*************')
# print(f'Accuracy {np.round(acc, 3)} ValidationAccuracy {np.round(val_acc, 3)}')
# modelName = f'acc:{acc} val_acc:{val_acc} epocs:{epocs} nodes:{lstmNodes}'
# model.save('./ModelSaves/FinalTrainedModels/'+modelName+'.h5')

# Getting the latent vector space from lstm layers
outputs = []
for layer in model.layers:
    keras_function = K.function([model.input], [layer.output])
    outputs.append((keras_function([lstm_patient_training_data])))
latentVectorSpace = (outputs[0][0])[:, (time_step-1), -1]
latentVectorSpace = np.reshape(latentVectorSpace, (no_of_patients, 1))

#Append latent vector with other data
d_tree_training_data_combined = np.append(d_tree_training_data, latentVectorSpace, axis=1)

test_loss, test_accuracy = model.evaluate(lstm_patient_test_data, lstm_patient_test_data_label)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")







# # Feature selection for decision tree model
# feature_cols = ['Age', 'Gender', 'GCS', 'Na', 'K', 'Cl', 'Urea', 'Creatinine', 'Alcoholic', 'Smoke',
#                 'FHCD', 'TriageScore', 'LSTM']

# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(d_tree_training_data_combined, d_tree_training_data_label,
#                                                     test_size=0.3, random_state=1)  # 70% training and 30% test
# clf = DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred)) 

# # Plotting accuracy metrices
# plot_lstm(history)
# plot_decision_tree()

# ## Prediction
# lstm_probabilities_prediction_ = model.predict_proba(lstm_patient_training_data)
# lstm_proba_prediction_label = np.reshape(lstm_probabilities_prediction_, (no_of_patients * time_step))
# lstm_true_label = np.reshape(lstm_patient_training_data_label, (no_of_patients * time_step))  
# lstm_class_label_prediction = model.predict_classes(lstm_patient_training_data)

# print("==========LSTM Confusion Matrix==========")
# print(metrics.confusion_matrix(lstm_true_label, lstm_class_label_prediction))
# print(lstm_class_label_prediction)
# lstm_falsePositive, lstm_truPositive, thresholds = roc_curve(lstm_true_label, lstm_proba_prediction_label)
# lstm_precision, lstm_recall, lstm_recall_thresholds = precision_recall_curve(lstm_true_label, lstm_proba_prediction_label)
# lstm_auc_score = roc_auc_score(lstm_true_label, lstm_proba_prediction_label)
# lstm_f1_score = f1_score(lstm_true_label, lstm_class_label_prediction)

# # Decision tree model evaluation (Finaldecision tree model that combines LSTM data)
# print("==========Decision Tree Confusion Matrix======")
# print(metrics.confusion_matrix(y_test, y_pred))
# d_tree_false_positive, d_tree_true_positive, d_tree_treshold = roc_curve(y_test, y_pred)
# d_tree_precision, d_tree_recall, d_tree_recall_threshold = precision_recall_curve(y_test, y_pred)
# d_tree_auc_score = roc_auc_score(y_test, y_pred)
# d_tree_f1_score = f1_score(y_test,y_pred)



