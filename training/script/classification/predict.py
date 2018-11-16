# -*- coding: utf-8 -*- 
import network
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import os
import time
import pickle



def predict(model, target_image, model_file):

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

    return np.argmax(features), features

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

    input_shape = (300, 400, 3)
    num_classes = 19

    model = network.darknet19(input_shape, num_classes)
    model.load_weights(model_path)

    all_true_list = []
    all_pred_list = []

    for i, shashu in enumerate(shashu_list):
        shashuh_true_list = ['bolt','cb1300','gsx_r1000','gsx_r1000r','gsx_s1000','gsx_s1000f','mt_07','mt_09','ninja','sv650','tmax','trecer900','v_strom_1000','v_strom_650','v_strom_650xt','yamaha__sr400','yzf_r1','yzf_r25','zrx1200']
        true = shashuh_true_list.index(shashu)

        shashu_dir = shashu_base_dir + shashu + '/'
        image_list = os.listdir(shashu_dir)
        true_list = []
        pred_list = []

        for j, target_image in enumerate(image_list):
            start = time.time()

            img_path = shashu_dir + target_image
            result, features = predict(model, img_path, model_path)

            true_list.append(true)
            pred_list.append(result)
            all_true_list.append(true)
            all_pred_list.append(result)
            if true == result:
                judge = 'ok'
            else:
                judge = 'ng'

            elapsed_time = time.time() - start

            # print('{0}:  {1}番目の予測結果:{2}  成否:{3}  image_path:{4}  elapsed_time:{5}'.format(shashu, str(j), str(result), judge, img_path, str(elapsed_time)[:5]))
            # print(shashu + ':  ' + str(j) + '番目の予測結果:' + str(result) + '  成否:' + judge + '  image_path:' + img_path + '  elapsed_time' + str(elapsed_time)[:5])

        with open('pred_results/' + str(true) + '_pred_list.pickle', 'wb') as wf:
            pickle.dump(pred_list, wf)

        test_num = len(true_list)
        correct_num = pred_list.count(true)
        accuracy = correct_num / test_num
        print(str(i) + '番目:' + shashu + ' テスト枚数:' + str(test_num) + ' 正解数:' + str(correct_num) + ' 正解率:' + str(accuracy))

    # with open('pred_results/all_true_list.pickle', 'wb') as wf:
    #     pickle.dump(all_true_list, wf)

    # with open('pred_results/all_pred_list.pickle', 'wb') as wf:
    #     pickle.dump(all_pred_list, wf)
