# -*- coding: utf-8 -*- 
import glob
from PIL import Image

def main():
    target_dir = '/Volumes/Transcend/open_image_dataset_v4/image/all_taxi_image/'
    out_dir = '/Volumes/Transcend/open_image_dataset_v4/image/all_image_resize_2400_1600/'
    resize_size = (2400, 1600)

    image_list = glob.glob(target_dir + '*.jpg')
    print(str(len(image_list)) + '枚をリサイズ   リサイズサイズ(w, h)：' + str(resize_size))

    for image in image_list:
        img = Image.open(image)
        resize_img = img.resize(resize_size)
        resize_img.save(out_dir + image.split('/')[-1].split('.')[0] + '_2400_1600.jpg')

if __name__ == '__main__':
    main()