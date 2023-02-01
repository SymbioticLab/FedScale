import os
import tensorflow as tf
IMG_SIZE = 28


def build_simple_linear(args):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(10, name='dense_2')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=1e-2,
            momentum=0.9,
            weight_decay=4e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    return model

class TFLiteModel(tf.Module):

    def __init__(self, tf_model:tf.keras.Sequential):
        self.model = tf_model

    # The `train` function takes a batch of input images and labels.
    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32),
        tf.TensorSpec([None, 10], tf.float32),
    ])
    def train(self, data, label):
        with tf.GradientTape() as tape:
            prediction = self.model(data)
            loss = self.model.loss(label, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        result = {"loss": loss}
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32),
        tf.TensorSpec([None, 10], tf.float32)
    ])
    def test(self, data, label):
        predict = self.model(data)
        loss = self.model.loss(label, predict)
        label_vec = tf.argmax(label, 1, output_type=tf.int32)
        top1 = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(predict, 1, output_type=tf.int32), label_vec), tf.int32))
        top5 = tf.reduce_sum(
            tf.cast(tf.equal(tf.nn.top_k(predict, 5).indices, label_vec[:, None]), tf.int32))
        return {
            "loss": loss,
            "top1": top1,
            "top5": top5
        }

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.string)
    ])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value()
                           for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }


def convert_and_save(tf_model: tf.keras.Sequential, saved_model_dir='cache', save=False):
    """Converts and saves the TFLite Transfer Learning model.

    Args:
      saved_model_dir: A directory path to save a converted model.
    """

    tf.saved_model.save(
        tf_model,
        saved_model_dir,
        signatures={
            'train': tf_model.train.get_concrete_function(),
            'test': tf_model.test.get_concrete_function(),
            'save': tf_model.save.get_concrete_function(),
        })

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    if save:
        model_file_path = os.path.join('model.tflite')
        with open(model_file_path, 'wb') as model_file:
            model_file.write(tflite_model)
    
    return tflite_model
