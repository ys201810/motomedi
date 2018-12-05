# -*- coding: utf-8 -*- 
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

def simple_auto_encoder(t_conf):
    encoding_dim = t_conf.encoding_dim
    input_img = Input(shape=(t_conf.image_size[0] * t_conf.image_size[1],))
    # input_img = Input(shape=(t_conf.image_size[0] * t_conf.image_size[1] * t_conf.image_size[2],))

    encoded = Dense(encoding_dim, activation='relu')(input_img)

    decoded = Dense(t_conf.image_size[0] * t_conf.image_size[1], activation='sigmoid')(encoded)
    # decoded = Dense(t_conf.image_size[0] * t_conf.image_size[1] * t_conf.image_size[2], activation='sigmoid')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    return autoencoder


def deep_auto_encoder(t_conf):
    # Input shape = W, H, C
    # input_img = Input(shape=(t_conf.image_size[2], t_conf.image_size[0], t_conf.image_size[1]))
    input_img = Input(shape=(t_conf.image_size[0], t_conf.image_size[1], t_conf.image_size[2]))
    # input_img = Input(shape=(t_conf.image_size[1], t_conf.image_size[2], t_conf.image_size[0]))

    N = 5

    x = Conv2D(filters=16 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv1')(input_img)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')(x)
    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2')(x)
    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv3')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='pool3')(x)

    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv4')(encoded)
    x = UpSampling2D((2, 2), name='upsa1')(x)
    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv5')(x)
    x = UpSampling2D((2, 2), name='upsa2')(x)
    x = Conv2D(filters=16 * N, kernel_size=(3, 3), strides=1, activation='relu', name='conv6')(x)
    x = UpSampling2D((2, 2), name='upsa3')(x)
    # decode's fileters correspond with channel size
    decoded = Conv2D(filters=1, kernel_size=(3, 3), strides=1, activation='sigmoid', padding='same', name='conv7')(x)

    """
    N = 1

    x = Conv2D(filters=16 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv1')(input_img)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')(x)
    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2')(x)
    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='pool3')(x)
    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='pool4')(x)
    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv5')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='pool5')(x)

    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv6')(encoded)
    x = UpSampling2D((2, 2), name='upsa1')(x)
    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv7')(x)
    x = UpSampling2D((2, 2), name='upsa2')(x)
    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv8')(x)
    x = UpSampling2D((2, 2), name='upsa3')(x)
    x = Conv2D(filters=8 * N, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv9')(x)
    x = UpSampling2D((2, 2), name='upsa4')(x)
    x = Conv2D(filters=16 * N, kernel_size=(3, 3), strides=1, activation='relu', name='conv10')(x)
    x = Conv2D(filters=16 * N, kernel_size=(3, 3), strides=1, activation='relu', name='conv11')(x)
    x = Conv2D(filters=16 * N, kernel_size=(3, 3), strides=1, activation='relu', name='conv12')(x)
    x = Conv2D(filters=16 * N, kernel_size=(3, 3), strides=1, activation='relu', name='conv13')(x)
    x = Conv2D(filters=16 * N, kernel_size=(3, 3), strides=1, activation='relu', name='conv14')(x)
    x = Conv2D(filters=16 * N, kernel_size=(3, 3), strides=1, activation='relu', name='conv15')(x)
    x = Conv2D(filters=16 * N, kernel_size=(3, 3), strides=1, activation='relu', name='conv16')(x)
    x = UpSampling2D((2, 2), name='upsa5')(x)
    # decode's fileters correspond with channel size
    decoded = Conv2D(filters=1, kernel_size=(3, 3), strides=1, activation='sigmoid', padding='same', name='conv17')(x)
    """

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # from keras.utils import plot_model
    # plot_model(autoencoder, to_file='model.png')

    return autoencoder
