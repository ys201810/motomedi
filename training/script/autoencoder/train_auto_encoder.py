# -*- coding: utf-8 -*- 
from keras.datasets import mnist
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
from collections import namedtuple
from keras.backend import tensorflow_backend as backend
import auto_encorder_network


def main():
    TrainConfig = namedtuple('TrainConfig', 'batch_size nb_epoch encoding_dim image_size ')
    # mnist_conf
    # t_conf = TrainConfig(256, 50, 32, (28, 28, 1))

    # fork_conf
    # t_conf = TrainConfig(256, 100, 320, (99, 196, 1))
    t_conf = TrainConfig(256, 500, 3200, (100, 196, 1))

    autoencoder = auto_encorder_network.simple_auto_encoder(t_conf)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    x_train, x_test = get_fork_data()

    print(x_train.shape, x_test.shape)
    autoencoder.fit(x_train, x_train,
                    nb_epoch=t_conf.nb_epoch,
                    batch_size=t_conf.batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    autoencoder.save_weights('autoencoder_3200_500.h5')
    # autoencoder.load_weights('autoencoder.h5')

    decoded_imgs = autoencoder.predict(x_test)
    draw_result(decoded_imgs, x_test, t_conf)
    backend.clear_session()

def get_fork_data():
    in_dir = '/usr/local/wk/work/VoTT/data/sr400_right/image_bb/'
    # in_dir = '/home/yusuke/work/motomedi/training/script/data/image_bb/'
    image_list = os.listdir(in_dir)
    pixel_list = []
    for i, image in enumerate(image_list):
        if image.find('.DS_store') > 0:
            continue
        img = Image.open(in_dir + image)
        # gray scale
        img = img.convert('L')
        # resize
        resize_img = img.resize((100, 196))

        imgArray = np.asarray(resize_img)
        # regularization
        imgArray = imgArray.astype('float32') / 255.
        pixel_list.append(imgArray)
        # print(i, imgArray.shape)

    pixel_list = np.stack(pixel_list)
    pixel_list = pixel_list.reshape((len(pixel_list), np.prod(imgArray.shape[0:])))
    # pixel_list = pixel_list.reshape((len(pixel_list), 1, 99, 193))
    train_images = pixel_list[:900]
    test_images = pixel_list[900:]

    return train_images, test_images

def get_mnist_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, x_test


def draw_result(decoded_imgs, x_test, t_conf):
    # 何個表示するか
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # オリジナルのテスト画像を表示
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(t_conf.image_size[1], t_conf.image_size[0]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 変換された画像を表示
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(decoded_imgs[i].reshape(t_conf.image_size[1], t_conf.image_size[0]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('figure.png')


if __name__ == '__main__':
    main()