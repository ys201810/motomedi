# -*- coding: utf-8 -*- 
import cv2
import numpy as np
from keras.applications import imagenet_utils
import os


class DataSet:
    def __init__(self, class_num=12, data_shape=(360, 480), train_file='train.txt', test_file='test.txt'):
        self.train_file = train_file
        self.test_file = test_file
        self.data_shape = data_shape[0] * data_shape[1]
        self.class_num = class_num

    def normalized(self, rgb):
        norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

        # opencv's sequence changing
        b = rgb[:, :, 0]
        g = rgb[:, :, 1]
        r = rgb[:, :, 2]

        norm[:, :, 0] = cv2.equalizeHist(b)
        norm[:, :, 1] = cv2.equalizeHist(g)
        norm[:, :, 2] = cv2.equalizeHist(r)

        return norm

    def one_hot_it(self, data_shape, class_num, labels):
        x = np.zeros([data_shape[0], data_shape[1], class_num])
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                x[i, j, labels[i, j]] = 1

        return x

    def load_data(self, mode, data_path, data_shape, class_num):
        data = []
        label = []
        if mode == 'train':
            filename = self.train_file
        else:
            filename = self.test_file

        with open(data_path + filename, 'r') as inf:
            txt = inf.readlines()
            txt = [line.split(' ') for line in txt]

        for i in range(len(txt)):
            # depends on annotation format
            data.append(self.normalized(cv2.imread(txt[i][0])))
            label.append(self.one_hot_it(data_shape=data_shape, class_num=class_num, labels=cv2.imread(txt[i][1].rstrip())[:, :, 0]))
            # data.append(self.normalized(cv2.imread(os.getcwd() + txt[i][0][7:])))
            # label.append(self.one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:, :, 0]))
            print(',', end='')
        print('\n')

        return np.array(data), np.array(label)

    def preprocess_inputs(self, X):
        return imagenet_utils.preprocess_input(X)

    def reshape_labels(self, y):
        return np.reshape(y, (len(y), self.data_shape, self.class_num))