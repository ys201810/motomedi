# motomedi

## about this repository
This repository have a below's function.
1. Training a DL model.(script location: 'training/')  
2. Confirming the DL inference result at Web UI.(script location: 'web_if/')  
    Currently this UI can confirm only Classification and Object Detection's results.  

## envirounment
python 3.5.2  
tensorflow 1.6.0  
keras 2.1.5  
CUDA 9.0  
GPU GTX1080Ti  

## about Training a DL model
This repository has below's task training scripts.  

| No | Task name | Network architecture | Script path |
|:-----------|:------------|:------------|:------------|
| 1 | Classification | DarkNet19. | training/script/classification/ |
| 2 | Object Detection | YOLOv3 | training/script/objectdetection/ |
| 3 | Semantic Segmentation | SegNet | training/script/segmentation/ |
| 4 | Auto Encoder | Very simple Network like a U-net | training/script/autoencoder/ |

Note that, now I don't write how to use each training scripts.  
I will update it in each directories.[WIP]  

## about Confirming the DL inference result.
I set up simple UI web apprication at 'web_if/'.  
If you want to check the result using this, then you should execute a below command.  

 ```
  cd web_if/
  python app.py
 ```

Then you can show the simple UI page at http://localhost:5000.  
But this is only an example, so if you want to customize your own model then you have to change the code.  

