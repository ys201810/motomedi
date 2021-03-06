# -*- coding: utf-8 -*- 
import network
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import os


def predict(target_image, model_file, num_class):
    model_path = model_file
    input_shape = (300, 400, 3)
    num_classes = num_class

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
    """
    if features[0][0] > features[0][1]:
        result = '歪みあり'
    else:
        result = '歪みなし'
    """
    if features.shape[-1] > 3:
        first_idx, second_idx, third_idx = np.argmax(features), np.where(features==np.sort(features[-1])[-2])[-1][0], np.where(features==np.sort(features[-1])[-3])[-1][0]
    else:
        first_idx, second_idx , third_idx = np.argmax(features), 0, 0

    return first_idx, second_idx, third_idx,  features

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
    # model_path = './model/cnn_model_weights_side.hdf5'
    model_path = '/home/yusuke/work/motomedi/training/saved/classification/20181116_1326/model/cnn_model_weights.hdf5'

    # image_dir = './test_data/distortion/'
    shashu_base_dir = '/home/yusuke/work/motomedi/datasets/classification/shashu/test/'
    shashu_list = os.listdir(shashu_base_dir)

    for i, shashu in enumerate(shashu_list):
        print(shashu)
        shashu_dir = shashu_base_dir + shashu + '/'
        image_list = os.listdir(shashu_dir)
        result_list = []

        for target_image in image_list:
            img_path = shashu_dir + target_image
            print(img_path)
            result, features = predict(img_path, model_path)
            print(result, features)
            result_list.append(result)

        test_num = len(result_list)
        correct_num = result_list.count(i)
        accuracy = correct_num / test_num
        print('=============================')
        print(str(i) + '番目:' + shashu + ' テスト枚数:' + str(test_num) + ' 正解数:' + str(correct_num) + ' 正解率:' + str(accuracy))
        print('=============================')
