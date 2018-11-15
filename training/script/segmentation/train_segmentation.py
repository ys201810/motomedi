# -*- coding: utf-8 -*- 
from collections import namedtuple
import keras
import tensorflow as tf
import segmentation.dataset as dataset
import segmentation.model as seg_model
import time
from keras.backend import tensorflow_backend as backend
from keras.preprocessing.image import ImageDataGenerator


def main():
    start = time.time()

    SSTrainConfig = namedtuple('SSTrainConfig', 'data_path out_path image_size class_num epochs batch_size')
    t_conf = SSTrainConfig('../../data/CamVid/', '../../saved/segmentation/', (360, 480, 3), 12, 12, 10)
    class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.8))
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

    print('====loading data====')
    ds = dataset.DataSet(class_num=t_conf.class_num, data_shape=t_conf.image_size,
                         train_file='train2.txt', test_file='test2.txt')
    train_data, train_labels = ds.load_data(mode='train',
                                            data_path=t_conf.data_path,
                                            data_shape=t_conf.image_size,
                                            class_num=t_conf.class_num)

    train_data = ds.preprocess_inputs(train_data)
    train_labels = ds.reshape_labels(train_labels)
    print('input data shape...', train_data.shape)
    print('input label shape...', train_labels.shape)

    test_data, test_labels = ds.load_data(mode='test',
                                          data_path=t_conf.data_path,
                                          data_shape=t_conf.image_size,
                                          class_num=t_conf.class_num)

    test_data = ds.preprocess_inputs(test_data)
    test_labels = ds.reshape_labels(test_labels)

    tb_cb = keras.callbacks.TensorBoard(log_dir=t_conf.out_path, histogram_freq=1, write_graph=True, write_images=True)
    print("creating model...")
    model = seg_model.SegNet(input_shape=t_conf.image_size, classes=t_conf.class_num)
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
    model.load_weights(t_conf.out_path + 'seg_9.h5')

    model.fit(train_data, train_labels, initial_epoch=9, batch_size=t_conf.batch_size, epochs=t_conf.epochs,
              verbose=1, class_weight=class_weighting , validation_data=(test_data, test_labels),
              shuffle=True, callbacks=[tb_cb])

    model.save(t_conf.out_path + 'seg_12.h5')

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    backend.clear_session()



if __name__ == '__main__':
    main()