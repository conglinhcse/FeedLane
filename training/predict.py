import os

from sklearn.preprocessing import LabelEncoder
import numpy as np

def predict(model, X, classnames):
    y_pred = model.predict(X)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(classnames)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred)     
    return y_pred