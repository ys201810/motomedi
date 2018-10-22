# motomedi

This repository have a below's function.
1. training a DL model.
2. confirm the DL inference.(Display confidence all labels)

## envirounment
python 3.5.2
tensorflow 1.6.0
keras 2.1.5

## about DL training
This repository has below's.
 1-1. classification(DarkNet19).  
 2-2. object detection(YOLO9000).  

### how to train
datasetディレクトリの下に、trainとtestを作成し、その下にそれぞれのカテゴリでディレクトリを作って画像を格納。
configデイレクトリ配下のconfig.iniにて設定して
`python train.py`
