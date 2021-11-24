# image-recognition
Klasifikasi tanaman selada dan seledri mengunakan Google Colab

# 1. Pra-pengolahan dataset
Siapkan dataset berupa gambar (.jpg) dan beri label menggunakan labelImg. LabelImg dapat didownload pada Link https://github.com/tzutalin/labelImg
Dalam pemberian label harus diingat huruf/kata yang digunakan.

# 2. Membuat labelmap.pbtxt
Labelmap memuat kelas dataset yang akan di klasifikasi. Cara membuat labelmap adalah dengan menulis pada Notepad dan menyimpan file dengan ektensi .pbtxt

.. code:: shell
item {
      id: 1
      name: 'apple'
}
item {
     id: 2
     name: 'orange'
}
