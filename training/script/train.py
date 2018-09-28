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
from keras import optimizers
import network
import tensorboard_conf


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
    model = network.darknet19(input_shape, num_classes)
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    # tensor board setting
    write_graph = False
    histogram_freq = 0
    clbk = train_val_tesnsorboard.TrainValTensorBoard(write_graph=write_graph, histogram_freq=histogram_freq)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,metrics=['accuracy'])
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights(os.path.join(model_dir,'cnn_model_weights_sr400_side_0_dis_1_nor.hdf5'))

    json_string = model.to_json()
    open(os.path.join(model_dir,'cnn_model_weights_sr400_side_0_dis_1_nor.json'), 'w').write(json_string)




if __name__ == '__main__':
    main()