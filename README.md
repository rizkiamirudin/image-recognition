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
