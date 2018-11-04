# -*- coding: utf-8 -*- 
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
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
    input_img = Input(shape=(1, t_conf.image_size[0], t_conf.image_size[1]))

    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 3, 3, activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # from keras.utils import plot_model
    # plot_model(autoencoder, to_file='model.png')

    return autoencoder