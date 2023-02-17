import os
import tensorflow as tf
IMG_SIZE = 224
NUM_FEATURES = 28224 # MobileNetV3Small
# NUM_FEATURES = 100352 # ResNet50
NUM_CLASSES = 10


def build_simple_linear(args):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(args.num_classes, name='dense_2'),
        tf.keras.layers.Softmax(),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            weight_decay=4e-5),
        loss=tf.keras.losses.CategoricalCrossentropy())
    return model, None


def build_mobilenetv3(args):
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=args.input_shape,
        classes=args.num_classes,
        weights=None)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            weight_decay=4e-5),
        loss=tf.keras.losses.CategoricalCrossentropy())
    return model, None


def build_mobilenetv3_finetune(args):
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=args.input_shape,
        include_top=False)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(args.num_classes, name='dense_2'),
        tf.keras.layers.Softmax(),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            weight_decay=4e-5),
        loss=tf.keras.losses.CategoricalCrossentropy())
    return model, base


def build_resnet50(args):
    model = tf.keras.applications.resnet.ResNet50(
        input_shape=args.input_shape,
        classes=args.num_classes,
        weights=None
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            weight_decay=4e-5),
        loss=tf.keras.losses.CategoricalCrossentropy())
    return model, None


def build_resnet50_finetune(args):
    base = tf.keras.applications.resnet.ResNet50(
        include_top=False,
        input_shape=args.input_shape,
        classes=args.num_classes
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(args.num_classes, name='dense_2'),
        tf.keras.layers.Softmax(),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            weight_decay=4e-5),
        loss=tf.keras.losses.CategoricalCrossentropy())
    return model, base


_models = {
    'linear': build_simple_linear,
    'mobilenetv3': build_mobilenetv3,
    'mobilenetv3_finetune': build_mobilenetv3_finetune,
    'resnet50': build_resnet50,
    'resnet50_finetune': build_resnet50_finetune
}


def get_tflite_model(name: str, args):
    if name not in _models:
        raise ValueError(f"Unsupported model: {name}")
    return _models[name](args)


class TFLiteModel(tf.Module):
    """TF model class."""

    def __init__(self, model: tf.Module):
        """Initializes a transfer learning model instance.

        Args:
            model (tf.Module): customized model to be trained.
        """
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32),
    ])
    def train(self, data: tf.Tensor, label: tf.Tensor) -> dict:
        """Runs one training step with the given features and labels.

        Args:
            data (tf.Tensor): A tensor of features sampled from the training set.
            label (tf.Tensor): A tensor of class labels for the given batch.

        Returns:
            dict: Map of the training loss.
        """
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
        tf.TensorSpec([None, NUM_CLASSES], tf.float32)
    ])
    def test(self, data, label):
        """Invokes a test on the given feature.

        Args:
            data (tf.Tensor): A tensor of features sampled from the testing set.
            label (tf.Tensor): A tensor of class labels for the given batch.

        Returns:
            dict: Map of the testing result, including loss, top1 and top5 accuracy.
        """
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
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)
    ])
    def infer(self, data: tf.Tensor) -> dict:
        """Invokes an inference on the given feature.

        Args:
            data (tf.Tensor): A tensor of image feature batch to invoke an inference on.


        Returns:
            dict: Map of the softmax output.
        """
        output = self.model(data)
        return {'output': output}

    @tf.function(input_signature=[tf.TensorSpec([], tf.string)])
    def save(self, checkpoint_path):
        """Saves the trainable weights to the given checkpoint file.

        Args:
            checkpoint_path (tf.string): A file path to save the model.

        Returns:
            dict: Map of the checkpoint file path.
        """
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value()
                           for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        """Restores the serialized trainable weights from the given checkpoint file.
        
        Args:
            checkpoint_path (tf.string): A path to a saved checkpoint file.

        Returns:
            dict: Map of restored weight and bias.
        """
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name,
                dt=var.dtype, name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors


class TFLiteModelFinetune(TFLiteModel):
    """TF Transfer Learning model class."""

    def __init__(self, model: tf.Module, base: tf.Module):
        """Initializes a transfer learning model instance.

        Args:
            model (tf.Module): customized model to be trained.
            base (tf.Module): frozen base model to extract feature.
        """
        self.model = model
        self.base = base

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32),
    ])
    def load(self, data: tf.Tensor) -> dict:
        """Generates and loads bottleneck features from the given image batch.

        Args:
            data (tf.Tensor): A tensor of image feature batch to generate the bottleneck from.

        Returns:
            dict: Map of the bottleneck.
        """
        bottleneck = tf.keras.layers.Flatten()(self.base(data, training=False))
        return {'bottleneck': bottleneck}

    @tf.function(input_signature=[
        tf.TensorSpec([None, NUM_FEATURES], tf.float32),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32),
    ])
    def train(self, data: tf.Tensor, label: tf.Tensor) -> dict:
        """Runs one training step with the given bottleneck features and labels.

        Args:
            data (tf.Tensor): A tensor of bottleneck features generated from the base model.
            label (tf.Tensor): A tensor of class labels for the given batch.

        Returns:
            dict: Map of the training loss.
        """
        with tf.GradientTape() as tape:
            prediction = self.model(data)
            loss = self.model.loss(label, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        result = {"loss": loss}
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([None, NUM_FEATURES], tf.float32),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32)
    ])
    def test(self, data: tf.Tensor, label: tf.Tensor) -> dict:
        """Invokes a test on the given feature.

        Args:
            data (tf.Tensor): A tensor of bottleneck features generated from the base model.
            label (tf.Tensor): A tensor of class labels for the given batch.

        Returns:
            dict: Map of the testing result, including loss, top1 and top5 accuracy.
        """
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
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)
    ])
    def infer(self, data: tf.Tensor) -> dict:
        """Invokes an inference on the given feature.

        Args:
            feature (tf.Tensor): A tensor of image feature batch to invoke an inference on.


        Returns:
            dict: Map of the softmax output.
        """
        bottleneck = tf.reshape(
            self.base(data, training=False), (-1, NUM_FEATURES))
        output = self.model(bottleneck)
        return {'output': output}


def convert_and_save(tf_model: tf.Module, tf_base: tf.Module, saved_model_dir='cache'):
    """Convert and save the tensorflow model.

    Args:
        tf_model (tf.Module): tensorflow model to be trained.
        tf_base (tf.Module): tensorflow model act as feature extractor.
        saved_model_dir (str, optional): directory to save the tensorflow frozen model. Defaults to 'cache'.

    Returns:
        bytes: TFLite model in bytes format.
        tf.Module: TFLite model in python object format.
    """
    if tf_base is None:
        tflite_model = TFLiteModel(tf_model)
        signatures = {
            'train': tflite_model.train.get_concrete_function(),
            'test': tflite_model.test.get_concrete_function(),
            'infer': tflite_model.infer.get_concrete_function(),
            'save': tflite_model.save.get_concrete_function(),
            'restore': tflite_model.restore.get_concrete_function(),
        }
    else:
        tflite_model = TFLiteModelFinetune(tf_model, tf_base)
        signatures = {
            'train': tflite_model.train.get_concrete_function(),
            'test': tflite_model.test.get_concrete_function(),
            'infer': tflite_model.infer.get_concrete_function(),
            'load': tflite_model.load.get_concrete_function(),
            'save': tflite_model.save.get_concrete_function(),
            'restore': tflite_model.restore.get_concrete_function(),
        }

    tf.saved_model.save(
        tflite_model,
        saved_model_dir,
        signatures=signatures)

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model_bytes = converter.convert()

    # model_file_path = os.path.join('model.tflite')
    # with open(model_file_path, 'wb') as model_file:
    #     model_file.write(tflite_model_bytes)

    return tflite_model_bytes, tflite_model
