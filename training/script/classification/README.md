## what is this directory
This directory is for cnn classification training.  
You can use only darknet19 now. I want to add other networks.

## how to use
1. Make a dataset.
 Prepare train and test data on /path/to/motomedi/datasets/.(if you don't have this repository, please make it.)  
 Make below's format.  
  /path/to/motomedi/datasets/fruit/train/[apple, banana, grape]  
  /path/to/motomedi/datasets/fruit/test/[apple, banana, grape]  
 For Example  

  ```
  /usr/local/wk/motomedi/datasets/fruit/train/apple/apple_1.png
  /usr/local/wk/motomedi/datasets/fruit/train/apple/apple_2.png
  /usr/local/wk/motomedi/datasets/fruit/train/apple/apple_3.png
  /usr/local/wk/motomedi/datasets/fruit/train/banana/banana_1.png
  /usr/local/wk/motomedi/datasets/fruit/train/banana/banana_2.png
  ...
  ```

    (Of course, you can use free name for image files. You don't need to follow example names.)

2. Edit conf file.
 Edit conf file(/path/to/motomedi/training/conf/config.ini)
 Editing points are below.
  - base_dir : set your environment's motomedi path.
  - image_height, image_width : set your image size. (if your original images doesn't match this number, it is ok. in the process, automatically resize using this settings.)
    [用意した画像サイズと、このサイズが違っても問題ないです。処理の中で設定したサイズで勝手にリサイズします。]
  - image_channel_dim : set your image channel dimensions. if you want to use gray scale then 1, rgb then 3.
  - batch_size : set cnn's processing batch size.
  - epoch_num : set cnn's processing epoch number. 1 epoch means using for training all training data.
  - class_num : set your target classification's result number.(this number correspond with number of directories under the datasets/train/ and datasets/test/)
  - train_path, test_path : set 1 procedure's path.
  - save_dir : set saving result and conf and model and log directories path.

3. Do training.
 ```
 export PYTHONPATY=$PYTHONPATY:/path/to/motomedi/training/script
 python train.py
 ```

 please fix from /path/to to your environment's path.

## notes
 This directory is WIP.
 if you want to fix some points then, please make a issue.
