import torch
import logging
import math
import numpy as np

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../../'))

from fedscale.core.client import Client
import tensorflow as tf 

class Customized_Client(Client):
    """Inherit default client to use tensorflow engine"""
    def __init__(self, conf):
        pass

    def train(self, client_data, model, conf):

        clientId = conf.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        train_len = len(client_data)
        
        def gen():
            while True:
                for x, y in client_data:
                    nx, ny = tf.convert_to_tensor(x.swapaxes(1, 3).numpy()), tf.convert_to_tensor(y.numpy()) 
                    yield nx, ny

        tf_client_data = tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.float32, tf.int64),
            output_shapes=(tf.TensorShape([conf.batch_size, 32, 32, 3]), 
                 tf.TensorShape([conf.batch_size]))
        )

        optimizer = tf.keras.optimizers.SGD(learning_rate=conf.learning_rate, momentum=0.0, 
                    nesterov=False, name='SGD')
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])
        
        history = model.fit(tf_client_data, epochs=1, steps_per_epoch=conf.local_steps, verbose=0)

        # Report the training results
        results = {'clientId': clientId, 
                    'moving_loss': sum(history.history['loss'])/(len(history.history['loss'])+1e-4),
                    'trained_size': conf.local_steps*train_len, 'success': True, 'utility': 1}

        logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")


        results['update_weight'] = {layer.name:layer.get_weights() for layer in model.layers}
        results['wall_duration'] = 0

        return results


    def test(self, conf):
        pass