# -*- coding: utf-8 -*- 
import numpy as np
import tensorflow as tf
import random as rn
import os
from keras import backend as K
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras import optimizers


def model1(input_shape, num_classes):
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


def model2(input_shape, num_classes):
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

    for layer in model.layers:
        print(layer.get_output_at(0).get_shape().as_list())
    return model

def main():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(0)
    rn.seed(0)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(0)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    input_shape = (300, 400, 3)
    batch_size = 8
    epochs = 100
    num_classes = 2

    # data_fir = '../datasets/fork_front_sr400/'
    data_dir = '../datasets/'
    log_dir = '../saved/tensorboard/'
    model_dir = '../saved/model/'


    x = []
    y = []
    for f in os.listdir(data_dir + "side_fork_distortion_sr400/"):
        x.append(image.img_to_array(image.load_img(data_dir + "side_fork_distortion_sr400/" + f, target_size=input_shape[:2])))
        y.append(0)
    for f in os.listdir(data_dir + "side_fork_sr400/"):
        x.append(image.img_to_array(image.load_img(data_dir + "side_fork_sr400/" + f, target_size=input_shape[:2])))
        y.append(1)

    x = np.asarray(x)
    x /= 255
    y = np.asarray(y)
    y = keras.utils.to_categorical(y, num_classes)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state= 3)

    print('model load')
    model = model2(input_shape, num_classes)
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,metrics=['accuracy'])
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[TrainValTensorBoard(write_graph=False)])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights(os.path.join(model_dir,'cnn_model_weights_sr400_side_0_dis_1_nor.hdf5'))

    json_string = model.to_json()
    open(os.path.join(model_dir,'cnn_model_weights_sr400_side_0_dis_1_nor.json'), 'w').write(json_string)


class TrainValTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir='../saved/', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, '../saved/tensorboard/train')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, '../saved/tensorboard/val')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

if __name__ == '__main__':
    main()