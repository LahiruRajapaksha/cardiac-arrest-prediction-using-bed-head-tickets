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
from datPreProcessingAndGrouping import dataGroupingAndPreProcessing
from modelPlots import plot_decision_tree, plot_lstm


# Command line arguments
parser =argparse.ArgumentParser(description='Early Warning Score for cardiac patients')
parser.add_argument('--epocs', type=int, help='number of iteration', default=5)
parser.add_argument('--lrate', type=float, help='learning rate', default=0.001)
parser.add_argument('--model-name', type=str, help='name of the model')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lstm-nodes', type=int, default=2)

args = parser.parse_args()
epocs = args.epocs
learningRate = args.lrate
# modelName =args.model_name
batchSize = args.batch_size
lstmNodes = args.lstm_nodes


NO_OF_FEATURES = 20

patients_data = pd.read_csv('./cardiacPatientData.csv') ## Dataset can be found in https://zenodo.org/record/7603772
time_step = int(patients_data.groupby('ID').count().mean()[0])
no_of_patients = patients_data['ID'].max()
newPatientList, decisionTreeData = dataGroupingAndPreProcessing(patients_data,time_step,NO_OF_FEATURES,no_of_patients)

# Shuffle the data set in the same order
newPatientList = newPatientList.tolist()
decisionTreeData = decisionTreeData.tolist()
shuffleDataSet = list(zip(newPatientList, decisionTreeData))
random.shuffle(shuffleDataSet)
newPatientList, decisionTreeData = zip(*shuffleDataSet)
newPatientList = np.asarray(newPatientList)
decisionTreeData = np.asarray(decisionTreeData)

# Prepare data set for LSTM model
lstm_patient_training_data = newPatientList[:, :, 1:7]
lstm_patient_training_data_label = newPatientList[:, :, -1]

# Reshaping the label data
lstm_patient_training_data_label = lstm_patient_training_data_label.reshape(no_of_patients, time_step, 1)

# Prepare data set for Decision Tree
d_tree_training_data = decisionTreeData[:, 7:19]
d_tree_training_data_label = np.reshape(decisionTreeData[:, -1], (no_of_patients, 1))

# Defining the model
model = Sequential()
model.add(LSTM(lstmNodes, return_sequences=True, input_shape=(time_step, 6)))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(Adam(lr=learningRate), loss='binary_crossentropy', metrics=['acc'])  # lr0.005 0r 0.001
model.summary()

# Fit the model LSTM
history = model.fit(lstm_patient_training_data, lstm_patient_training_data_label,
                    shuffle=False, validation_split=0.3, batch_size=batchSize, epochs=epocs)
acc = np.round(history.history['acc'][epocs-1], 2)
val_acc = np.round(history.history['val_acc'][epocs-1], 2)

print('**********Summary*************')
print(f'Accuracy {np.round(acc, 3)} ValidationAccuracy {np.round(val_acc, 3)}')
modelName = f'acc:{acc} val_acc:{val_acc} epocs:{epocs} nodes:{lstmNodes}'
model.save('./ModelSaves/FinalTrainedModels/'+modelName+'.h5')

# Getting the latent vector space from lstm layers
outputs = []
for layer in model.layers:
    keras_function = K.function([model.input], [layer.output])
    outputs.append((keras_function([lstm_patient_training_data, 1])))
latentVectorSpace = (outputs[0][0])[:, (time_step-1), -1]
latentVectorSpace = np.reshape(latentVectorSpace, (no_of_patients, 1))
d_tree_training_data_combined = np.append(d_tree_training_data, latentVectorSpace, axis=1)#Append latent vector with other data

# Feature selection for decision tree model
feature_cols = ['Age', 'Gender', 'GCS', 'Na', 'K', 'Cl', 'Urea', 'Creatinine', 'Alcoholic', 'Smoke',
                'FHCD', 'TriageScore', 'LSTM']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(d_tree_training_data_combined, d_tree_training_data_label,
                                                    test_size=0.3, random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred)) 

# Plotting accuracy metrices
plot_lstm(history)
plot_decision_tree()

## Prediction
lstm_probabilities_prediction_ = model.predict_proba(lstm_patient_training_data)
lstm_proba_prediction_label = np.reshape(lstm_probabilities_prediction_, (no_of_patients * time_step))
lstm_true_label = np.reshape(lstm_patient_training_data_label, (no_of_patients * time_step))  
lstm_class_label_prediction = model.predict_classes(lstm_patient_training_data)

print("==========LSTM Confusion Matrix==========")
print(metrics.confusion_matrix(lstm_true_label, lstm_class_label_prediction))
print(lstm_class_label_prediction)
lstm_falsePositive, lstm_truPositive, thresholds = roc_curve(lstm_true_label, lstm_proba_prediction_label)
lstm_precision, lstm_recall, lstm_recall_thresholds = precision_recall_curve(lstm_true_label, lstm_proba_prediction_label)
lstm_auc_score = roc_auc_score(lstm_true_label, lstm_proba_prediction_label)
lstm_f1_score = f1_score(lstm_true_label, lstm_class_label_prediction)

# Decision tree model evaluation (Finaldecision tree model that combines LSTM data)
print("==========Decision Tree Confusion Matrix======")
print(metrics.confusion_matrix(y_test, y_pred))
d_tree_false_positive, d_tree_true_positive, d_tree_treshold = roc_curve(y_test, y_pred)
d_tree_precision, d_tree_recall, d_tree_recall_threshold = precision_recall_curve(y_test, y_pred)
d_tree_auc_score = roc_auc_score(y_test, y_pred)
d_tree_f1_score = f1_score(y_test,y_pred)



