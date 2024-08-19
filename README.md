## HandSpeak: Revolusi Interaksi dengan Gesture Tangan
### Sign Language & Computer Control

HandSpeak adalah aplikasi inovatif yang mengubah cara berinteraksi dengan perangkat melalui gesture tangan. Dengan teknologi terkini dan kemampuan real-time, aplikasi ini menawarkan berbagai fitur berikut:

  1. Deteksi Huruf dengan Gesture Tangan
     - Ketik teks dengan cara yang menyenangkan dan intuitif. Dengan gesture tangan yang spesifik, huruf-huruf dapat diketikkan, kata-kata dapat dirangkai, dan kalimat dapat dibentuk. Kata-kata yang telah dirangkai dapat diucapkan menggunakan gTTS.
       

https://github.com/user-attachments/assets/b77b1e0e-7a44-43f5-a050-db7154d9e5dc


  2. Kontrol Volume dengan Gesture Tangan
     - Atur volume perangkat dengan gerakan tangan yang sederhana.
       

https://github.com/user-attachments/assets/f656b48d-bf58-4949-84e6-8b8204efe714


  3. Kontrol Kecerahan dengan Gesture Tangan
     - Sesuaikan kecerahan layar dengan gerakan tangan.
       

https://github.com/user-attachments/assets/cbed63cd-19f3-449c-b632-22c9e1c5fe35


  4. Menggambar dengan Gesture Tangan
     - Jadikan tangan sebagai alat gambar. Dengan HandSpeak, menggambar secara kreatif menggunakan tangan sebagai "pensil" atau "pulpen" menjadi mungkin. Pilih warna dan buat gambar langsung di layar dengan gesture tangan.
       

https://github.com/user-attachments/assets/29304a46-f2b9-446c-85be-9f81c57f9514


## Panduan

### 1. Instalasi

Install Library

```bash
pip install -r requirements.txt
```


### 2. Optimalisasi
Supaya GUI sesuai, ubah resolusi layar menjadi **1366x768** atau yang mendekati
- Cara ubah resolusi layar:
  - Dekstop -> Klik Kanan -> Display settings -> Scroll Kebawah -> Display resolution

Dan untuk hasil deteksi yang maksimal usahakan jarak tangan dengan kamera kurang lebih 2 jengkal

>saat deteksi huruf, jika huruf yang diinginkan sudah ditampilkan di character 
maka lakukan gesture "next" untuk menyimpan character ke sentence. 
jika huruf yang terdeteksi pada kotak tidak berubah selama setengah detik,
maka huruf yang terdeteksi tersebut akan di simpan di character.
misal ingin membuat kata "saya" maka lakukan:    
"s" "next" "a" "next" "y" "next" "a"


>program kadang error entah kenapa saya pun tidak tau, jadi kalo gagal coba lagi aja terus jangan menyerah!!!!
kalo bingung liat vid demo aja dah



Referensi:
https://github.com/Devansh-47/Sign-Language-To-Text-and-Speech-Conversion/blob/master/README.md
