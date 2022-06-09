import os

from training.metric import FreezeCallback

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

def pretrain(cnnmodel, config, X_train, X_val, y_train, y_val, ngpus=0):
        train_log = dict()

        WEIGHT = config["weight"]
        LOG = config["log"]

        learning_rate= config["lr"]
        epoch = config["epoch"]
        batch_size = config["batch_size"]
        loss = config["loss"]
        metrics = config["metrics"]
        
        if config["optimizer"].lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif config["optimizer"].lower() == "sgd":
            momentum = config["momentum"]
            nesterov = config["nesterov"]
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
        elif config["optimizer"].lower() == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        lr_reduce = ReduceLROnPlateau(monitor='val_auc', factor=0.6, patience=5, verbose=1, mode='max', min_lr=5e-5, min_delta=5e-4)
        checkpoint = ModelCheckpoint(WEIGHT, monitor= 'val_auc', mode= 'max', save_best_only = True, verbose= 1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        fzcb = FreezeCallback(n_epochs=10)
        tbcb = TensorBoard(log_dir=LOG)

        if ngpus > 0:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
            config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
            config = tf.config.experimental.set_memory_growth(physical_devices[1], True)

            learning_rate *= ngpus
            batch_size *= ngpus

            device_type = 'GPU'
            devices = tf.config.experimental.list_physical_devices(device_type)
            devices_names = [d.name.split('e:')[1] for d in devices]
            strategy = tf.distribute.MirroredStrategy(devices=devices_names[:ngpus])

            with strategy.scope():
                inpt, outpt = cnnmodel()
                model = tf.keras.Model(inputs=inpt, outputs=outpt, name=config['model_name'])
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        else:
            inpt, outpt = cnnmodel()
            model = tf.keras.Model(inputs=inpt, outputs=outpt, name=config['model_name'])
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        model.summary()
        history = model.fit(X_train, y_train, batch_size = batch_size, epochs=epoch, validation_data=(X_val, y_val), callbacks=[lr_reduce,checkpoint,es,fzcb,tbcb])

        model.save(WEIGHT)

        train_log['model'] = model
        train_log['history'] = history
        train_log['epoch'] = es.stopped_epoch + 1
        train_log['learning_rate'] = learning_rate
        train_log['batch_size'] = batch_size

        return train_log