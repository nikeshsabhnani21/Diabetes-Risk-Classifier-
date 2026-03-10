import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


target_names = ["No Diabetes", "Diabetes"]

def evaluation(y_test, y_pred):
    """
    Produces classification report for each model
    """
    return classification_report(y_test, y_pred, target_names=target_names)

def heatmaps(y_test, y_pred):
    """
    Produces heatmaps for each model
    """
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    result = plt.show()

    return result
