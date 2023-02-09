import logging
import tensorflow as tf
from overrides import overrides
from fedscale.cloud.execution.client_base import ClientBase
import numpy as np

from fedscale.cloud.internal.tensorflow_model_adapter import TensorflowModelAdapter


class TensorflowClient(ClientBase):
    """Implements a TensorFlow-based client for training and evaluation."""

    def __init__(self, args):
        """
        Initializes a tf client.
        :param args: Job args
        """
        self.args = args

    def _convert_np_to_tf_dataset(self, dataset):
        """
        Converts the iterable numpy dataset to a tensorflow Dataset.
        :param dataset: numpy dataset
        :return: tf.data.Dataset
        """
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
        """
        Perform a training task.
        :param client_data: client training dataset
        :param model: the framework-specific model
        :param conf: job config
        :return: training results
        """
        client_id = conf.client_id
        logging.info(f"Start to train (CLIENT: {client_id}) ...")
        tf_dataset = self._convert_np_to_tf_dataset(client_data).take(conf.local_steps)
        history = model.fit(tf_dataset, batch_size=conf.batch_size, verbose=0)

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
        """
        Perform a testing task.
        :param client_data: client evaluation dataset
        :param model: the framework-specific model
        :param conf: job config
        :return: testing results
        """
        tf_dataset = self._convert_np_to_tf_dataset(client_data)
        results = model.evaluate(tf_dataset, batch_size=conf.batch_size,
                                 steps=len(client_data.dataset)/conf.batch_size, return_dict=True, verbose=0)
        for key, value in results.items():
            if key != 'row_count':
                results[key] = results['row_count'] * value
        # transform results to unified form
        results['top_1'] = results['accuracy']
        results['test_loss'] = results['loss']
        results['test_len'] = results['row_count']
        return results

    @overrides
    def get_model_adapter(self, model) -> TensorflowModelAdapter:
        """
        Return framework-specific model adapter.
        :param model: the model
        :return: a model adapter containing the model
        """
        return TensorflowModelAdapter(model)
