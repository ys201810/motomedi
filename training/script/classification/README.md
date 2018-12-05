## about this directory
This directory is for cnn classification training.  
In this direcroty, you can use only darknet19 now. I want to add other networks in the future.

## how to use
### 1. Make a dataset.
 Prepare train and test directories at /path/to/motomedi/datasets/. (if you don't have this directory, please make it.)  
 You should make like a below's two directories.  
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

    (Of course, you can use free name for image files. You don't need to follow perfectory example names.)

### 2. Edit the conf file.
 Edit conf file(/path/to/motomedi/training/conf/config.ini)
 Editing points are below.
 
| No | variable name | example | remark |
|:-----------:|:------------|:------------|:--------|
| 1 | base_dir | /usr/local/wk/motomedi/  | your environment's motomedi path. |
| 2 | image_height | 300 | your images height size. if this doesn't match your image file height, it is ok. Automatically resize on processing using this config. |
| 3 | image_width | 400 | your images width size. and same as image_height. |
| 4 | image_channel_dim | 3 | your image channel dimensions. |
| 5 | batch_size | 12 | cnn's processing batch size. |
| 6 | epoch_num | 10 | cnn's processing epoch number. 1 epoch means using for training all training data. |
| 7 | class_num | 3 | your target classification's result number.(this number correspond with number of directories under t    he datasets/train/ and datasets/test/) |
| 8 | train_path | /usr/local/wk/motomedi/datasets/fruit/train/ | your train data path. |
| 9 | test_path | /usr/local/wk/motomedi/datasets/fruit/test/ | your test data path. |
| 10 | save_dir | /usr/local/wk/motomedi/training/saved/ | your save path. after processing this path save log, model, result, conf file. |

### 3. Do training.
 ```
 export PYTHONPATY=$PYTHONPATY:/path/to/motomedi/training/script
 python train.py
 ```

 please fix from /path/to to your environment's path.

## notes
 This directory is WIP.
 if you want to fix some points then, please make a issue.
