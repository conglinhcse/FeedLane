import os, sys
import glob
import cv2

PYTHON_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PYTHON_PATH)

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataAugment():
    def __init__(self):
        pass

class Dataset():
    def __init__(self, img_shape, data_path="/content/FeedLane/data/subdata"):
        self.IMG_SHAPE = img_shape
        self.DATA_PATH = data_path
        self.LABEL = ["empty","full","minimal","normal"]
        self.DATA_TYPE = ['train', 'test']

    def data_generator(self, data_path=None, data_type=None):
        assert data_type in ["train", "test"], "data_type must be in ['train', 'test']"
        if data_path != None:
          self.DATA_PATH = data_path

        X, Y = [], []

        for label in self.LABEL:
          path = f"{self.DATA_PATH}/{label}"
          ls_img_path = os.listdir(path)
          for ele in ls_img_path:
            img = cv2.imread(f"{path}/{ele}")
            img = cv2.resize(img, self.IMG_SHAPE[:2])
            img = img / 255.0
            X.append(img)
            Y.append(label)
        
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

        scaler = StandardScaler()

        if data_type == 'train':
            # splitting data
            X_train, X_val, y_train, y_val = train_test_split(X, Y, random_state=0, shuffle=True, test_size=0.2)

            # scaling our data with sklearn's Standard scaler
            # X_train = scaler.fit_transform(X_train)
            # X_val = scaler.transform(X_val)

            # making our data compatible to model.
            # X_train = np.expand_dims(X_train, axis=2)
            # X_val = np.expand_dims(X_val, axis=2)

            # print(f"X_train test shape : {X_train.shape}")
            # print(f"y_train test shape : {y_train.shape}")
            # print(f"X_val test shape : {X_val.shape}")
            # print(f"y_val test shape : {y_val.shape}")

            return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)
        else:
            # scaling our data with sklearn's Standard scaler
            X = scaler.fit_transform(X)

            # making our data compatible to model.
            X = np.expand_dims(X, axis=2)

            print(f"X test shape : {X.shape}")
            print(f"Y test shape : {Y.shape}")

            return X, Y

if __name__=="__main__":
  data = Dataset()
  data.data_generator("train") 