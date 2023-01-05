import logging
import tensorflow as tf
from overrides import overrides
from fedscale.cloud.execution.client_base import ClientBase
import numpy as np


class TensorflowClient(ClientBase):
    """Inherit default client to use tensorflow engine"""

    def __init__(self, args):
        self.args = args

    def convert_np_to_tf_dataset(self, dataset):
        def gen():
            while True:
                for x, y in dataset:
                    # Convert torch tensor to tf tensor
                    nx, ny = tf.convert_to_tensor(x.swapaxes(1, 3).numpy()), \
                             tf.one_hot(tf.convert_to_tensor(y.numpy()), self.args.num_classes)
                    yield nx, ny

        # Sample a batch to get tensor properties
        temp_x, temp_y = next(gen())
        x_shape, y_shape = temp_x.shape.as_list(), temp_y.shape.as_list()
        x_shape[0], y_shape[0] = None, None

        return tf.data.Dataset.from_generator(
            gen,
            output_shapes=(tf.TensorShape(x_shape), tf.TensorShape(y_shape)),
            output_types=(temp_x.dtype, temp_y.dtype),
        )

    @overrides
    def train(self, client_data, model, conf):
        client_id = conf.client_id
        logging.info(f"Start to train (CLIENT: {client_id}) ...")
        tf_dataset = self.convert_np_to_tf_dataset(client_data).take(conf.local_steps)
        history = model.fit(tf_dataset, batch_size=conf.batch_size, verbose=1)

        # Report the training results
        results = {'client_id': client_id,
                   'moving_loss': sum(history.history['loss']) / (len(history.history['loss']) + 1e-4),
                   'trained_size': history.history['row_count'], 'success': True, 'utility': 1}

        logging.info(f"Training of (CLIENT: {client_id}) completes, {results}")

        results['update_weight'] = [np.asarray(layer.get_weights()) for layer in model.layers if layer.trainable]
        results['wall_duration'] = 0

        return results

    @overrides
    def test(self, client_data, model, conf):
        results = model.evaluate(self.convert_np_to_tf_dataset(client_data).take(100), batch_size=conf.batch_size,
                                 return_dict=True)
        for key, value in results.items():
            if key != 'row_count':
                results[key] = results['row_count'] * value
        results['test_len'] = results['row_count']
        return results
