# Image Recognition
Klasifikasi tanaman selada dan seledri mengunakan Google Colab

## 1. Pra-pengolahan dataset
Siapkan dataset berupa gambar (.jpg) dan beri label menggunakan labelImg. LabelImg dapat didownload pada Link berikut [labelImg](https://github.com/tzutalin/labelImg)
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
