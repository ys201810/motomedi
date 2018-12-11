# -*- coding: utf-8 -*- 
import numpy as np
from PIL import Image
import cv2


def make_base_img(base_image):
    imgArray = np.array([
        [
            [255, 0, 0], [0, 255 ,0], [0, 0, 255], [0, 0, 0]
        ],
        [
            [255, 255, 0], [0, 255, 255], [255, 0, 255], [64, 64, 64]
        ],
        [
            [126, 64, 32], [32, 126, 64], [64, 32, 126], [126, 126, 126]
        ]
    ])

    # image save.
    baseImg = Image.fromarray(np.uint8(imgArray))
    baseImg.save(base_image)

def one_hot_it(data_shape, class_num, labels):
    x = np.zeros([data_shape[0], data_shape[1], class_num])
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            x[i, j, labels[i, j]] = 1

    return x

def main():
    # base_image = '../data/pil_base.png'
    base_image = '../data/0001TP_008190.png'
    # make_base_img(base_image)

    # load from pil and convert to np.array
    # pil_image = Image.open(base_image)
    # np_array_pil = np.asarray(pil_image)
    # print(np_array_pil)

    # load from opencv
    cv2_image = cv2.imread(base_image)
    # print(type(cv2_image), cv2_image)
    label = []
    """
    for i in range(cv2_image.shape[0]): # height
        for j in range(cv2_image.shape[1]): # width
            print(type(cv2_image[i, j]))

    print(cv2_image, '\n ============================\n')
    print(cv2_image[:, :, 0]) # B要素だけを取得している。
    print(cv2_image[:, :, 1]) # G要素だけを取得している。
    print(cv2_image[:, :, 2]) # R要素だけを取得している。
    """
    x = one_hot_it(data_shape=cv2_image.shape[:2], class_num=12, labels=cv2_image[:, :, 0])
    print(x)

    # print(imgArray, imgArray.shape, imgArray.shape[2:])

    # to gray scale[image_num, w * h]
    # reshapeArray = imgArray.reshape((1, np.prod(imgArray.shape[0:])))

    # to rgb scale[image_num, w * h, 3(rgb)]
    # reshapeArray = imgArray.reshape((1, np.prod(imgArray.shape[1:3]), imgArray.shape[2]))
    # pilImg = Image.fromarray(np.uint8(reshapeArray))
    # pilImg.save('4.png')

if __name__ == '__main__':
    main()