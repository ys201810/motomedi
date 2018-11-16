# -*- coding: utf-8 -*- 
import numpy as np
import tensorflow as tf
import random as rn
import os
from keras import backend as K
from keras.preprocessing import image
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import keras
import numpy as np
from keras import optimizers
import classification.network as network
import common.tensorboard_conf as tensorboard_conf
import configparser
import datetime
import shutil
from keras.preprocessing.image import ImageDataGenerator
from collections import namedtuple


def setting_conf(conf_file):
    TrainConfig = namedtuple('TrainConfig', 'batch_size epoch_num input_shape class_num ' +
                             'save_experiment_dir save_log_dir save_model_dir train_dir test_dir')
    config = configparser.ConfigParser()
    config.read(conf_file)

    image_height = int(config.get('image_info', 'image_height'))
    image_width = int(config.get('image_info', 'image_width'))
    image_channel_dim = int(config.get('image_info', 'image_channel_dim'))
    input_shape = (image_height, image_width, image_channel_dim)

    batch_size = int(config.get('train_info', 'batch_size'))
    epoch_num = int(config.get('train_info', 'epoch_num'))
    class_num = int(config.get('label_info', 'class_num'))

    now = datetime.datetime.now()
    experiment_id = now.strftime('%Y%m%d_%H%M')

    base_dir = config.get('base_info', 'base_dir')
    save_dir = base_dir + config.get('other_info', 'save_dir')
    save_experiment_dir = save_dir + experiment_id + '/'
    save_log_dir = save_experiment_dir + '/tensorboard'
    save_model_dir = save_experiment_dir + '/model/'
    train_dir = base_dir + config.get('label_info', 'train_path')
    test_dir = base_dir + config.get('label_info', 'test_path')

    t_conf = TrainConfig(
        batch_size,
        epoch_num,
        input_shape,
        class_num,
        save_experiment_dir,
        save_log_dir,
        save_model_dir,
        train_dir,
        test_dir
    )

    return t_conf

def main():
    # pre processing
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(0)
    rn.seed(0)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # GPU memory rate
    # session_conf.gpu_options.per_process_gpu_memory_fraction = 0.4
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.set_random_seed(0)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

    K.set_session(sess)

    # config setting
    conf_file = '../../conf/config.ini'
    t_conf = setting_conf(conf_file)


    if os.path.exists(t_conf.save_experiment_dir):
        shutil.rmtree(t_conf.save_experiment_dir)
    if os.path.exists(t_conf.save_log_dir):
        shutil.rmtree(t_conf.save_log_dir)
    if os.path.exists(t_conf.save_model_dir):
        shutil.rmtree(t_conf.save_model_dir)

    os.mkdir(t_conf.save_experiment_dir)
    os.mkdir(t_conf.save_log_dir)
    os.mkdir(t_conf.save_model_dir)

    # config save
    shutil.copyfile(conf_file, t_conf.save_experiment_dir + conf_file.split('/')[-1])

    # normal_labelnum = int(config.get('label_info', 'normal'))
    # distortion_labelnum = int(config.get('label_info', 'distortion'))

    train_datagen = ImageDataGenerator(rescale=1./255
                                       # width_shift_range=0.2
                                       # shear_range=0.2,
                                       # zoom_range=0.2,
                                       # rotation_range=180,
                                       # horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        t_conf.train_dir,
        target_size=t_conf.input_shape[:2],
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        t_conf.test_dir,
        target_size=t_conf.input_shape[:2],
        batch_size=32,
        class_mode='categorical')

    print('model load')
    model = network.darknet19(t_conf.input_shape, t_conf.class_num)
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    # tensor board setting
    write_graph = True # network graph draw
    histogram_freq = 0 # each layer's distribution draw

    clbk = tensorboard_conf.TrainValTensorBoard(log_dir=t_conf.save_log_dir, write_graph=write_graph, histogram_freq=histogram_freq)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,metrics=['accuracy'])

    history = model.fit_generator(train_generator,
                                  # samples_per_epoch = None,
                                  nb_epoch=t_conf.epoch_num,
                                  validation_data=test_generator,
                                  callbacks=[clbk])

    score = model.evaluate_generator(test_generator)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights(os.path.join(t_conf.save_model_dir,'cnn_model_weights.hdf5'))

    json_string = model.to_json()
    open(os.path.join(t_conf.save_model_dir,'cnn_model_weights.json'), 'w').write(json_string)
    with open(t_conf.save_experiment_dir + 'result.txt', 'w') as result_f:
        result_f.write('Final Test loss: ' + str(score[0]) + '\n' + 'Final Test accuracy: '+ str(score[1]))

if __name__ == '__main__':
    main()