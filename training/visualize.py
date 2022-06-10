import os

# Visualize
import matplotlib.pyplot as plt
import numpy as np

def visualize(train_log):
    model = train_log['model']
    history = train_log['history']
    epoch = train_log['epoch']

    epochs = [i for i in range(epoch)]
    fig , ax = plt.subplots(1,3)
    train_auc = history.history['auc']
    train_acc = history.history['categorical_accuracy']
    train_loss = history.history['loss']
    val_auc = history.history['val_auc']
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    fig.set_size_inches(20,6)
    ax[0].plot(epochs , train_loss , label = 'Training Loss')
    ax[0].plot(epochs , val_loss , label = 'Validating Loss')
    ax[0].set_title('Training & Validating Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
    ax[1].plot(epochs , val_acc , label = 'Validating Accuracy')
    ax[1].set_title('Training & Validating Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")

    ax[2].plot(epochs , train_auc , label = 'Training AUC')
    ax[2].plot(epochs , val_auc , label = 'Validating AUC')
    ax[2].set_title('Training & Validating AUC')
    ax[2].legend()
    ax[2].set_xlabel("Epochs")
    plt.show()

import matplotlib.pyplot as plt


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()