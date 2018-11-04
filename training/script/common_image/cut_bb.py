# -*- coding: utf-8 -*- 
import os
import cv2

def main():
    """
    バウンディングボックスのアノテーションがある画像から、バウンディングボックス内だけの画像を取得する。
    アノテーションは、
    xxxx.jpg x1,y1,x2,y2 というファイルを想定。
    :return:
    """
    print('a')
    in_dir = '/usr/local/wk/work/VoTT/data/sr400_right/normal/'
    out_dit = '/usr/local/wk/work/VoTT/data/sr400_right/image_bb/'
    annotation_path = '/usr/local/wk/work/VoTT/data/sr400_right/normal_output/train_right_fork.txt'

    with open(annotation_path, 'r') as anno_f:
        for line in anno_f:
            line = line.rstrip()
            vals = line.split(' ')
            image_name = vals[0].split('/')[-1]
            x_left_upper = int(vals[1].split(',')[0])
            y_left_upper = int(vals[1].split(',')[1])
            x_right_lower = int(vals[1].split(',')[2])
            y_right_lower = int(vals[1].split(',')[3])
            img = cv2.imread(in_dir + image_name)
            bb_img = img[y_left_upper:y_right_lower, x_left_upper:x_right_lower, :]
            cv2.imwrite(out_dit + image_name.split('.')[0] + '_bb.' + image_name.split('.')[1], bb_img)

if __name__ == '__main__':
    main()