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
    def __init__(self, config):
        self.IMG_SIZE = config['img_size']
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
        self.TRAINING_SET = config['training_set']
        self.TESTING_SET = config['testing_set']
        self.CLASS_NAMES = config["class_names"]
        self.DATA_TYPE = config["data_type"]

    def data_generator(self, data_type=None):
        assert data_type in ["train", "test"], "data_type must be in ['train', 'test']"

        X, Y = [], []

        if data_type == 'train':
            DATA_PATH = self.TRAINING_SET
        else:
            DATA_PATH = self.TESTING_SET

        for label in self.CLASS_NAMES:
          path = f"{DATA_PATH}/{label}"
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