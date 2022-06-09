import os

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def evaluate(model, X_test, y_test, classnames):
    print("Accuracy of our model on testing data : " , round(model.evaluate(X_test,y_test)[1]*100,5), "%")

    # predicting on test data.
    pred_test = model.predict(X_test)

    y_pred = np.argmax(pred_test, axis=1)
    y_test = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (12, 10))
    cm = pd.DataFrame(cm , index = [i for i in classnames] , columns = [i for i in classnames])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='', vmin=0, vmax=1)
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.show()

    print(classification_report(y_test, y_pred))

    return y_pred