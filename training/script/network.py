# -*- coding: utf-8 -*- 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization


def easy_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def darknet19(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(64, kernel_size=(1, 1),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(128, kernel_size=(1, 1),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(512, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(256, kernel_size=(1, 1),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(512, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(256, kernel_size=(1, 1),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(512, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(1024, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(512, kernel_size=(1, 1),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    # model.add(Conv2D(1024, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    # model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(512, kernel_size=(1, 1),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    # model.add(Conv2D(1024, kernel_size=(3, 3),activation='relu',input_shape=input_shape, name='bb'))
    # model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # for layer in model.layers:
        # print(layer.get_output_at(0).get_shape().as_list())
    return model
