import glob
import argparse
import os, sys

PYTHON_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PYTHON_PATH)
from util.util import *
from dataset.dataset import Dataset
from training.evaluate import evaluate
from training.predict import predict
from training.visualize import visualize
from training.pretrain import pretrain

# Math
import numpy as np
# Tensorflow - Keras
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Reshape, RepeatVector, Conv1D, Input, BatchNormalization, AveragePooling2D, Dense, Dropout, Conv2D
from tensorflow.python.keras.layers.core import Flatten

class CNNModel:
    def __init__(self, config):
        self.config = parse_configs(config)
        self.IMG_SIZE = self.config['img_size']
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
        self.MODEL_NAME = self.config['model_name']
        self.CLASS_NAMES = self.config["class_names"]
        self.NUM_CLASSES = self.config["num_classes"]
        self.DATA_TYPE = self.config["data_type"]
        self.WEIGHT = self.config["weight"]
        self.DATA_PATH = self.config["data_path"]
        self.dataset = Dataset(self.IMG_SHAPE)

    def cnnmodel(self):
        inpt = Input(
            shape=self.IMG_SHAPE,
            name="inpt",
        )

        vgg16 = VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=inpt,
            pooling=None,
            classes=self.NUM_CLASSES,
            classifier_activation="softmax",)

        vgg16.trainable = False

        outpt = vgg16.output

        outpt = Conv2D(
            filters=256,
            kernel_size=1,
            input_shape=outpt.shape[1:],
            strides=(1, 1),
            padding="same",
            activation='relu'
        )(outpt)

        outpt = Conv2D(
            filters=128,
            kernel_size=1,
            input_shape=outpt.shape[1:],
            strides=(1, 1),
            padding="same",
            activation='relu'
        )(outpt)

        outpt = Reshape(
            target_shape=((outpt.shape[1]*outpt.shape[2]*outpt.shape[3],)),
            name="reshape_3",
        )(outpt)

        outpt = Dense(units=128, activation='relu', name='fc_1')(outpt)
        outpt = Dropout(0.25)(outpt)
        outpt = Dense(units=64, activation='relu', name='fc_2')(outpt)
        outpt = Dropout(0.25)(outpt)
        outpt = Dense(units=32, activation='relu', name='fc_3')(outpt)
        outpt = Dense(units=4, activation='softmax', name='output')(outpt)

        return inpt, outpt

    def train(self, X_train, X_val, y_train, y_val, ngpus=0):
        train_log = pretrain(self.cnnmodel, self.config, X_train, X_val, y_train, y_val, ngpus=0)
        return train_log

    def evaluate(self, X_test, y_test, weight=None):
        model = self.cnnmodel()
        if weight != None:
            model.load_weights(weight)
        else:
            model.load_weights(self.WEIGHT)
        model.summary()
        y_pred = evaluate(model, X_test, y_test, self.CLASS_NAMES)

    def predict(self, X):
        model = self.cnnmodel()
        model.load_weights(self.WEIGHT)
        model.summary()
        y_pred = predict(model, X, self.CLASS_NAMES)   
        return y_pred

    def run(self, mode, ngpus):
        assert mode in ["train", "test"], "mode must be in ['train', 'test']"

        if mode == 'train':
            X_train, X_val, y_train, y_val = self.dataset.data_generator(data_path=self.DATA_PATH, data_type='train')
            train_log = self.train(X_train, X_val, y_train, y_val,ngpus)
            # visualize(train_log)
            return train_log['model'], train_log['history']
        else: 
            X, Y = self.dataset.data_generator(data_path=self.DATA_PATH, data_type='test')
            # evaluate(X,Y)
            return None, None

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--mode', type=str, default=None)
    parse.add_argument('--ngpus', type=int, default=0)
    opt = parse.parse_args()

    CONFIG = "/content/FeedLane/config/config.json"

    cnnmodel = CNNModel(CONFIG)
    model, history = cnnmodel.run(opt.mode, opt.ngpus)
