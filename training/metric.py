import os

import tensorflow as tf
import numpy as np

class FreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, n_epochs=10, start_cnn=5, end_cnn=22):
        super().__init__()
        # n_epochs=0 means that some layers in model are not unfreezed
        self.n_epochs = n_epochs - 1
        self.start_cnn = start_cnn
        self.end_cnn = end_cnn

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.n_epochs:
            for id in range(self.start_cnn, self.end_cnn):
                l = self.model.get_layer(index=id)
                l.trainable = True
        from tensorflow.keras import backend as K

        trainable_count = int(np.sum([K.count_params(p) for p in list(self.model.trainable_weights)]))
        non_trainable_count = int(np.sum([K.count_params(p) for p in list(self.model.non_trainable_weights)]))

        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))
