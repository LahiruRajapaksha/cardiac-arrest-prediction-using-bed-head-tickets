import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


 ## Experiment data set. Import each experiment data set separately
patients_training_data = pd.read_csv('/content/sample_data/ExperimentNo1Data.csv').to_numpy()
patients_test_data = pd.read_csv('/content/sample_data/Test10%.csv').to_numpy()

x_train = patients_training_data[:,:-2]
y_train = patients_training_data[:,-1]

x_test = patients_test_data[:,:-2]
y_test = patients_test_data[:,-1]

def accuracy_metrics(trueLabel, predlabel,name):
    print('********Measures of******** ',name)
    con = metrics.confusion_matrix(trueLabel, predlabel)
    total = sum(sum(con))
    print('accuracy =', (con[0, 0] + con[1, 1]) / total)
    print('sensitivity =', con[0, 0] / (con[0, 0] + con[1, 0]))
    print('specificity =', con[1, 1] / (con[1, 1] + con[0, 1]))
    print('ppv =', con[0, 0] / (con[0, 0] + con[0, 1]))
    print('npv =', con[1, 1] / (con[0, 1] + con[1, 1]))
    print('fscore =', metrics.f1_score(trueLabel, predlabel))

#Logistic Regression Model
lin_model = LogisticRegression(solver='lbfgs')
lin_model.fit(x_train, y_train)
lmodel_pred =lin_model.predict(x_test)
print(metrics.confusion_matrix(y_test,lmodel_pred))
print("Linear Model Accuracy: ", lin_model.score(x_test, y_test))
accuracy_metrics(y_test, lmodel_pred,'LogisticRegression')

#Support Vector Machine Model
svm_model = SVC(gamma='auto')
svm_model.fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)
print("Support Vector Machine Model Accuracy: ", svm_model.score(x_test, y_test))
accuracy_metrics(y_test, svm_pred,'SVM Tree')

#Decision Tree Model
tree_model = DecisionTreeClassifier()
tree_model.fit(x_train, y_train)
tree_pred = tree_model.predict(x_test)
print("Decision Tree Model Accuracy: ", tree_model.score(x_test, y_test))
accuracy_metrics(y_test, tree_pred,'Decision Tree')

#Random Forest Model
forest_model = RandomForestClassifier(n_estimators=100)
forest_model.fit(x_train, y_train)
forest_pred = forest_model.predict(x_test)
print("Random Forest Model Accuracy: ", forest_model.score(x_test, y_test))
accuracy_metrics(y_test, forest_pred,'Random Forest')






