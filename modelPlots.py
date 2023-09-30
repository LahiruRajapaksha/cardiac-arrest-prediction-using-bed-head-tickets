# Plot accuracy, validation accuracy loss graphs for models

import matplotlib.pyplot as plt
from six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus

# LSTM model
def plot_lstm(hist):  # LSTM plotting
    plt.figure(figsize=(13, 8))
    plt.plot(hist.history['loss'], color='blue')
    plt.plot(hist.history['val_loss'], color='orange')
    plt.plot(hist.history['acc'], color='red')
    plt.plot(hist.history['val_acc'], color='green')
    plt.title('model loss during training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss', 'acc', ' val_acc'], loc='upper left')
    plt.show()


# Decision tree plotting
def plot_decision_tree():
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('cardiacPatient.png')
    Image(graph.create_png())

