import tensorflow as tf


class RowCount(tf.keras.metrics.Metric):
    def __init__(self, name='row_count', **kwargs):
        super(RowCount, self).__init__(**kwargs)
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.count.assign_add(tf.reduce_sum(tf.cast(tf.shape(y_true)[0], self.dtype)))

    def reset_state(self):
        self.count.assign(0)

    def result(self):
        return self.count


def build_resnet50(args):
    model = tf.keras.applications.resnet.ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=args.input_shape,
        pooling=None,
        classes=args.num_classes
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9,
                                        nesterov=False, name='SGD')
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5"), RowCount()])
    return model


def build_mobilenet_v3_small(args):
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=args.input_shape,
        alpha=1.0,
        minimalistic=False,
        input_tensor=None,
        weights=None,
        classes=args.num_classes,
        pooling=None,
        dropout_rate=0.2,
        include_preprocessing=True,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9,
                                        nesterov=False, name='SGD')
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5"), RowCount()])
    return model


_models = {
    'resnet50': build_resnet50,
    'mobilenetv3_small': build_mobilenet_v3_small
}


def get_tensorflow_model(name: str, args):
    if name not in _models:
        raise ValueError(f"Unsupported model: {name}")
    return _models[name](args)
