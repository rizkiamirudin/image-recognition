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

## 14. Start Tensorboard
```
#cd into training_demo
%cd '/content/gdrive/My Drive/TensorFlow/workspace/training_demo'
#start the Tensorboard
%load_ext tensorboard
%tensorboard --logdir=models/my_ssd_resnet50_v1_fpn
# %load_ext tensorboard
# %tensorboard --logdir=models/[name_of_pre-trained-model_you_downloaded]
```
## 15. Train model
```
!python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
# !python model_main_tf2.py --model_dir=models/[name_of_pre-trained-model_you_downloaded] --pipeline_config_path=models/[name_of_pre-trained-model_you_downloaded]/pipeline.config
```
## 16. Export the Trained Model
```
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_resnet50_v1_fpn/pipeline.config --trained_checkpoint_dir ./models/my_ssd_resnet50_v1_fpn/ --output_directory ./exported-models/my_model
# !python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/[name_of_pre-trained-model you downloaded]/pipeline.config --trained_checkpoint_dir ./models/[name_of_pre-trained-model_you_downloaded]/ --output_directory ./exported-models/my_model
```

## 17. Testing the model (Loading saved_model)
```
#Loading the saved_model(change the path according to your directory names)
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
PATH_TO_SAVED_MODEL="/content/gdrive/My Drive/TensorFlow/workspace/training_demo/exported-models/my_model/saved_model"
print('Loading model...', end='')
# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')
```

## 18. Testing the model (Loading label_map)
```
#Loading the label_map
category_index=label_map_util.create_category_index_from_labelmap("/content/gdrive/My Drive/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt",use_display_name=True)
#category_index=label_map_util.create_category_index_from_labelmap([path_to_label_map],use_display_name=True)
```

## 19. Testing the model (Loading images)
```
#Loading the image
img=['/content/img1.jpg','/content/img2.jpg']
print(img)
#list containing paths of all the images
```

## 20. Running the Inference
```
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
def load_image_into_numpy_array(path):
    return np.array(Image.open(path))
for image_path in img:
print('Running inference for {}... '.format(image_path), end='')
    image_np=load_image_into_numpy_array(image_path)
input_tensor=tf.convert_to_tensor(image_np)
    input_tensor=input_tensor[tf.newaxis, ...]
detections=detect_fn(input_tensor)
num_detections=int(detections.pop('num_detections'))
    detections={key:value[0,:num_detections].numpy()
                   for key,value in detections.items()}
    detections['num_detections']=num_detections
detections['detection_classes']=             detections['detection_classes'].astype(np.int64)
image_np_with_detections=image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=100,     
          min_score_thresh=.5,      
          agnostic_mode=False)
%matplotlib inline
    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
    plt.show()
```
