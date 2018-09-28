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
import configparser
import datetime
import shutil


def main():
    # pre processing
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(0)
    rn.seed(0)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(0)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # config setting
    config_file = '../conf/config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    image_height = int(config.get('image_info', 'image_height'))
    image_width = int(config.get('image_info', 'image_width'))
    image_channel_dim = int(config.get('image_info', 'image_channel_dim'))
    input_shape = (image_height, image_width, image_channel_dim)

    batch_size = int(config.get('train_info', 'batch_size'))
    epochs = int(config.get('train_info', 'epochs'))

    num_classes = int(config.get('label_info', 'num_classes'))

    now = datetime.datetime.now()
    experiment_id = now.strftime('%Y%m%d_%H%M')
    save_dir = config.get('other_info', 'save_dir')
    save_experiment_dir = save_dir + experiment_id + '/'
    data_dir = config.get('other_info', 'data_dir')
    log_dir = save_experiment_dir + '/tensorboard'
    model_dir = save_experiment_dir + '/model/'

    os.mkdir(save_dir + experiment_id)
    os.mkdir(log_dir)
    os.mkdir(model_dir)
    shutil.copyfile(config_file, save_experiment_dir + config_file.split('/')[-1])

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
    write_graph = True # network graph draw
    histogram_freq = 0 # each layer's distribution draw

    clbk = tensorboard_conf.TrainValTensorBoard(log_dir=log_dir, write_graph=write_graph, histogram_freq=histogram_freq)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,metrics=['accuracy'])
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[clbk])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights(os.path.join(model_dir,'cnn_model_weights.hdf5'))

    json_string = model.to_json()
    open(os.path.join(model_dir,'cnn_model_weights.json'), 'w').write(json_string)
    with open(save_experiment_dir + 'result.txt', 'w') as result_f:
        result_f.write('Final Test loss: ' + str(score[0]) + '\n' + 'Final Test accuracy: '+ str(score[1]))

if __name__ == '__main__':
    main()