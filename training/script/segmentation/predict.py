# -*- coding: utf-8 -*- 
import numpy as np
import keras
from PIL import Image

from segmentation.model import SegNet

import segmentation.dataset as dataset

height = 360
width = 480
classes = 12
epochs = 100
batch_size = 1
log_filepath='./../../saved/segmentation/'

data_shape = 360*480

def writeImage(image, filename):
    """ label data to colored image """
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0,12):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

def predict(test):
    model = keras.models.load_model('../../saved/segmentation/seg_9.h5')
    probs = model.predict(test, batch_size=1)

    prob = probs[0].reshape((height, width, classes)).argmax(axis=2)
    return prob

def main():
    print("loading data...")
    # ds = dataset.Dataset(test_file='../../data/CamVid/val.txt', classes=classes)
    ds = dataset.DataSet(class_num=classes, data_shape=(360, 480, 3),
                         train_file='train2.txt', test_file='val2.txt')
    test_X, test_y = ds.load_data(mode='test', data_path='../../data/CamVid/', data_shape=(360, 480, 3),class_num=classes) # need to implement, y shape is (None, 360, 480, classes)
    print(test_X.shape, test_y.shape)
    test_X = ds.preprocess_inputs(test_X)
    test_Y = ds.reshape_labels(test_y)
    print(test_X.shape, test_y.shape)
    print(test_X[0].shape, test_y[0].shape)
    np.reshape(test_X[0], (1, 360, 480, 3))
    np.reshape(test_y[0], (1, 360, 480, 12))
    print(test_X[0].shape, test_y[0].shape, np.reshape(test_X[0], (1, 360, 480, 3)).shape, np.reshape(test_y[0], (1, 360, 480, 12)).shape)

    for i in range(101):
        print(i)
        prob = predict(np.reshape(test_X[i], (1, 360, 480, 3)))
        writeImage(prob, 'predict_result/val_' + str(i) + '.png')

if __name__ == '__main__':
    main()
