# Image Recognition
Klasifikasi tanaman selada dan seledri mengunakan Google Colab

## 1. Pra-pengolahan dataset
Siapkan dataset berupa gambar (.jpg) dan beri label menggunakan [labelImg](https://github.com/tzutalin/labelImg).
Dalam pemberian label harus diingat huruf/kata yang digunakan.

### a. Menulis kelas dataset pada .pbtxt
Kelas-kelas yang akan digunakan ditulis pada Notepad dengan eksistensi .pbtxt

```
item {
      id: 1
      name: 'seledri'
}
item {
     id: 2
     name: 'selada'
}
```
### b. Download model pre-trained
[TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).
Model yang digunakan

### c. Download script generate_tfrecords.py
Script ini berfungsi untuk menkonversi file eksistensi .csv menjadi file .tfrecord

## 2. Atur direktori pada Google Drive

```
TensorFlow
├───scripts
│   └───preprocessing
│     └───generate_tfrecord.py 
└───workspace
    └───training_demo
        ├───annotations
        │   └───label_map.pbtxt 
        ├───exported-models
        ├───images
        │   ├───test
        │   │     └───test images with corresponding XML files
        │   └───train
        │         └───train images with corresponding XML files
        ├───models
        │   └───my_ssd_resnet50_v1_fpn
        │     └───pipeline.config
        └───pre-trained-models
            └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
```
## 3. Setting Hardware pada Google Colab

## 4. Menghubungkan pada Google Drive
```from google.colab import drive
drive.mount('/content/gdrive')
```

## 5. Download TensorFlow Model Garden
```
#cd into the TensorFlow directory in your Google Drive
%cd '/content/gdrive/My Drive/TensorFlow'
#and clone the TensorFlow Model Garden repository
!git clone https://github.com/tensorflow/models.git
#using a older version of repo
%cd '/content/gdrive/MyDrive/TensorFlow/models'
!git checkout -f e04dafd04d69053d3733bb91d47d0d95bc2c8199
```

## 6. Install beberapa library dan tools
```
!apt-get install protobuf-compiler python-lxml python-pil
!pip install Cython pandas tf-slim lvis
```

## 7. Kompile librari Protobuf
```
#cd into 'TensorFlow/models/research'
%cd '/content/gdrive/My Drive/TensorFlow/models/research/'
!protoc object_detection/protos/*.proto --python_out=.
```

## 8. Setting Environment
```
import os
import sys
os.environ['PYTHONPATH']+=":/content/gdrive/My Drive/TensorFlow/models"
sys.path.append("/content/gdrive/My Drive/TensorFlow/models/research")
```

## 9. Build dan install setup.py
```
!python setup.py build
!python setup.py install
```

## 10. Test the Installation
```
#cd into 'TensorFlow/models/research/object_detection/builders/'
%cd '/content/gdrive/My Drive/TensorFlow/models/research/object_detection/builders/'
!python model_builder_tf2_test.py
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
print('Done')
```

## 11. Generate Tfrecords
```
#cd into preprocessing directory
%cd '/content/gdrive/My Drive/TensorFlow/scripts/preprocessing'
#run the cell to generate test.record and train.record
!python generate_tfrecord.py -x '/content/gdrive/My Drive/TensorFlow/workspace/training_demo/images/train' -l '/content/gdrive/My Drive/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt' -o '/content/gdrive/My Drive/TensorFlow/workspace/training_demo/annotations/train.record'
!python generate_tfrecord.py -x '/content/gdrive/My Drive/TensorFlow/workspace/training_demo/images/test' -l '/content/gdrive/My Drive/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt' -o '/content/gdrive/My Drive/TensorFlow/workspace/training_demo/annotations/test.record'
# !python generate_tfrecord.py -x '[path_to_train_folder]' -l '[path_to_annotations_folder]/label_map.pbtxt' -o '[path_to_annotations_folder]/train.record'
# !python generate_tfrecord.py -x '[path_to_test_folder]' -l '[path_to_annotations_folder]/label_map.pbtxt' -o '[path_to_annotations_folder]/test.record'
```

## 12. Copy beberapa file
Copy the “model_main_tf2.py” file from “TensorFlow\models\research\object_detection” and paste it in training_demo folder. We will need this file for training the model.
Copy the “exporter_main_v2.py” file from “TensorFlow\models\research\object_detection” and paste it in training_demo folder. We will need this file to export the trained model.

## 13. Konfigurasi file pipeline
```
Line 3:
num_classes: 3 (#number of classes your model can classify/ number of different labels)
Line 131:
batch_size: 16 (#you can read more about batch_size here)
Line 161:
fine_tune_checkpoint: "pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" (#path to checkpoint of downloaded pre-trained-model)
Line 162:
num_steps: 250000 (#maximum number of steps to train model, note that this specifies the maximum number of steps, you can stop model training on any step you wish)
Line 167:
fine_tune_checkpoint_type: "detection" (#since we are training full detection model, you can read more about model fine-tuning here)
Line 168:
use_bfloat16: false (#Set this to true only if you are training on a TPU)
Line 172:
label_map_path: "annotations/label_map.pbtxt" (#path to your label_map file)
Line 174:
input_path: "annotations/train.record" (#path to train.record)
Line 182:
label_map_path: "annotations/label_map.pbtxt" (#path to your label_map file)
Line 186:
input_path: "annotations/test.record" (#Path to test.record)
```
