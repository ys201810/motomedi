# -*- coding: utf-8 -*- 
import os
import cv2


def main():
    target_dir = '/usr/local/wk/work/VoTT/data/sr400_right/image_bb/'
    image_list = os.listdir(target_dir)

    max_w_size = 0
    max_h_size = 0
    for image in image_list:
        if image.find('DS.store') > 0:
            continue
        img = cv2.imread(target_dir + image)
        h = img.shape[0]
        w = img.shape[1]
        if h > max_h_size:
            max_h_size = h
        if w > max_w_size:
            max_w_size = w
        print(img.shape, w, h, h /w)
    print(max_w_size, max_h_size, max_h_size/ max_w_size)

if __name__ == '__main__':
    main()