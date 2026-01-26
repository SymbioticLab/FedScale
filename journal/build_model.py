# build_model.py

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Dropout, Flatten
from keras.regularizers import l2

class Model():

    def __init__(self, loss, optimizer, classes=10):
        self.loss = loss
        self.optimizer = optimizer
        self.num_classes = classes

    def fl_paper_model(self, train_shape):
        model = Sequential()
        
        l2_reg = 0.001

        # 1
        model.add(Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
            input_shape=train_shape,
            # kernel_regularizer=l2(l2_reg),
        ))
        model.add(Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
            # kernel_regularizer=l2(l2_reg),
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        model.add(Dropout(0.2))

        # 2
        model.add(Conv2D(
            filters=64,
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
            # kernel_regularizer=l2(l2_reg),
        ))
        model.add(Conv2D(
            filters=64,
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
            # kernel_regularizer=l2(l2_reg),
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        model.add(Dropout(0.2))

        # 3
        model.add(Flatten())
        model.add(Dense(
            units=512,
            activation='relu',
            # kernel_regularizer=l2(l2_reg),
        ))
        model.add(Dropout(0.2))
        
        # 4
        model.add(Dense(
            units=self.num_classes,
            activation='softmax'
        ))

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        return model