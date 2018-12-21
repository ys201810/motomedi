# -*- coding: utf-8 -*-
"""
target_dirに対象のディレクトリを指定して、縦横のサイズと比率をグラフで表示。このままではグラフは表示しないで、画像パス・横幅・縦幅・比率をリストで保存するだけ。
graphが見たければ、graph_flg = 0を0以外に変えて。
"""
import glob
from PIL import Image
import matplotlib.pyplot as plt
from statistics import mean, median,variance,stdev

def main():
    target_dir = '/Volumes/Transcend/open_image_dataset_v4/image/all_taxi_image/'
    image_list = glob.glob(target_dir + '*.jpg')
    print(len(image_list))
    graph_flg = 0

    width_list = []
    height_list = []
    rate_list = []

    with open('image_list.txt', 'a') as image_list_outf:
        image_list_outf.write('image_path width height rate\n')
        for image in image_list:
            img = Image.open(image)
            width_list.append(img.size[0])
            height_list.append(img.size[1])
            rate_list.append(round(img.size[0] / img.size[1], 2))
            image_list_outf.write(' '.join([image, str(img.size[0]), str(img.size[1]), str(round(img.size[0] / img.size[1], 2))]) + '\n')

    if graph_flg == 0:
        exit(1)

    plt.hist(width_list)
    plt.title('width histogram')
    plt.text(1000, 300, 'max:' + str(max(width_list) )+ '\nmin:' + str(min(width_list)) + '\nave:' + str(mean(width_list)))
    plt.show()

    plt.hist(height_list)
    plt.title('height histogram')
    plt.text(1000, 300, 'max:' + str(max(height_list) )+ '\nmin:' + str(min(height_list)) + '\nave:' + str(mean(height_list)))
    plt.show()

    plt.hist(rate_list)
    plt.title('rate histogram')
    plt.text(1000, 300, 'max:' + str(max(rate_list) )+ '\nmin:' + str(min(rate_list)) + '\nave:' + str(mean(rate_list)))
    plt.show()

if __name__ == '__main__':
    main()