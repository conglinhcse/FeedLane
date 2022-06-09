import os

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

def train(cnnmodel, config, X_train, X_val, y_train, y_val, ngpus=0):
    train_log = dict()

    epoch = config["epoch"]
    batch_size = config["batch_size"]        

    if ngpus > 0:
        batch_size *= ngpus
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_memory_growth(physical_devices[1], True)

        device_type = 'GPU'
        devices = tf.config.experimental.list_physical_devices(device_type)
        devices_names = [d.name.split('e:')[1] for d in devices]
        strategy = tf.distribute.MirroredStrategy(devices=devices_names[:ngpus])

        with strategy.scope():
            model, weight_path, log_path = cnnmodel(ngpus=ngpus)
    else:
        model, weight_path, log_path = cnnmodel() 

    lr_reduce = ReduceLROnPlateau(monitor='val_auc', factor=0.6, patience=10, verbose=1, mode='max', min_lr=5e-5, min_delta=2e-4)
    checkpoint = ModelCheckpoint(weight_path, monitor= 'val_auc', mode= 'max', save_weight_only = True, save_best_only = True, verbose= 1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    tbcb = TensorBoard(log_dir=log_path)
    
    model.summary()
    history = model.fit(X_train, y_train, batch_size = batch_size, epochs=epoch, validation_data=(X_val, y_val), callbacks=[lr_reduce,es,tbcb,checkpoint])

    train_log['model'] = model
    train_log['history'] = history
    train_log['epoch'] = es.stopped_epoch
    train_log['batch_size'] = batch_size

    return train_log