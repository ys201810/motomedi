# -*- coding: utf-8 -*- 
import network
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import os


def predict(target_image, model_file):
    model_path = model_file
    input_shape = (300, 400, 3)
    num_classes = 2

    model = network.darknet19(input_shape, num_classes)
    model.load_weights(model_path)

    # normal_images = os.listdir(sr400_normal_dir)
    # distortion_images = os.listdir(sr400_distortion_dir)
    # distortion_images = os.listdir((distortion_dir))
    # collect_cnt = 0

    x_list = []
    result = ''
    img = image.img_to_array(image.load_img(target_image, target_size=input_shape[:2]))
    x_list.append(img)

    x_list = np.asarray(x_list)
    x_list /= 255

    features = model.predict(x_list)
    if features[0][0] > features[0][1]:
        result = '歪みあり'
    else:
        result = '歪みなし'

    return result, features

"""
    for i, image_name in enumerate(distortion_images):
        x_list = []
        result = ''
        img_path = distortion_dir + image_name
        img = image.img_to_array(image.load_img(img_path, target_size=input_shape[:2]))
        x_list.append(img)

        x_list = np.asarray(x_list)
        x_list /= 255

        features = model.predict(x_list)
        if features[0][0] > features[0][1]:
            result = 'normal'
        else:
            result = 'distortion'
            collect_cnt += 1

        print(image_name, result, features)
    print(i, collect_cnt, collect_cnt/ i)
"""

if __name__ == '__main__':
    # image_dir = './test_data/distortion/'
    image_dir = './test_data/side_distortion/'
    image_list = os.listdir(image_dir)
    for target_image in image_list:
        img_path = image_dir + target_image
        # model_path = './model/cnn_model_weights_front_test.hdf5'
        model_path = './model/cnn_model_weights_side.hdf5'
        # model_path = './model/cnn_model_weights_sr400_side_0_dis_1_nor.hdf5'
        # model_path = './model/cnn_model_weights_sr400_side.hdf5'
        print(img_path)
        result, features = predict(img_path, model_path)
        print(result)
