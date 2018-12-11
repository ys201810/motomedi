# -*- coding: utf-8 -*- 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, ZeroPadding2D, Input, Flatten
import numpy as np
from keras.optimizers import SGD
import keras
from keras import backend as K
import tensorflow as tf

def main():
    sess = tf.Session(graph=tf.get_default_graph())

    K.set_session(sess)

    # データ準備
    """
    data1 = np.linspace(1, N, N0)
    data1 = data1.reshape((N, N, 1))
    data2 = np.linspace(1, N, N0)
    data2 = data2.reshape((N, N, 1))
    """
    N = 4
    data1, data2, data3, data4, data5 = np.full((N,N,1), 1), np.full((N,N,1), 2), np.full((N,N,1), 3), \
                                        np.full((N,N,1), 4), np.full((N,N,1), 5)
    data6, data7, data8, data9, data10 = np.full((N,N,1), 252), np.full((N,N,1), 253), np.full((N,N,1), 254), \
                                        np.full((N,N,1), 255), np.full((N,N,1), 256)
    label1, label2, label3, label4, label5 = 0, 0, 0, 0, 0
    label6, label7, label8, label9, label10 = 1, 1, 1, 1,1

    data = np.asarray([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10])
    label = np.asarray([label1, label2, label3, label4, label5, label6, label7, label8, label9, label10])
    data = data
    label = keras.utils.to_categorical(label, 2)


    # モデル設定52
    model = Sequential()
    input_shape = (N, N, 1)
    model.add(Conv2D(filters=2, kernel_size=(2, 2), strides=2, activation='relu', input_shape=input_shape, name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='pool1'))
    model.add(Flatten())
    # model.add(Dense(5, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    # モデル学習
    model.fit(x=data, y=label, batch_size=1, epochs=500, verbose=1)

    # モデル保存
    # print(model.output)

    model.save_weights('./cnn_model_weights.h5')
    json_string = model.to_json()
    open(('./cnn_model_weights.json'), 'w').write(json_string)

    # model.load_weights('cnn_model_weights.h5')

    # print('Input img\n{}'.format(model.layers[0].input))

    M = 2
    test_data = np.asarray([
                            [
                               [1 * M], [2 * M], [3 * M], [4 * M]
                            ],
                            [
                               [5 * M], [6 * M], [7 * M], [8 * M]
                            ],
                            [
                               [9 * M], [10 * M], [11 * M], [12 * M]
                            ],
                            [
                               [13 * M], [14 * M], [15 * M], [16 * M]
                            ]
                            ])

    for i in range(len(model.layers)):
        print('layer no:{}'.format(model.layers[i].name))

        if len(model.layers[i].get_weights()) > 0:
            print('model weight shape{} model bias shape{}'.format(model.layers[i].get_weights()[0].shape, model.layers[i].get_weights()[1].shape))
            print('model weight{}\nmodel bias'.format(model.layers[i].get_weights()[0], model.layers[i].get_weights()[1]))
            # if i == 0:
                # np.save('test2_5.npy', model.layers[i].get_weights()[0])

        getFeature = K.function([model.layers[0].input, K.learning_phase()],[model.layers[i].output])
        feature_out = getFeature([test_data.reshape(1, N, N, 1), 0])[0]
        print('shape:{}\nvalue{}\n'.format(feature_out.shape, feature_out))


    # 予測


    K.clear_session()

if __name__ == '__main__':
    main()