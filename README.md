# Panduan Penggunaan Aplikasi Deteksi Gesture Tangan

## 1. Instalasi

Install Library

```bash
pip install -r requirements.txt
```


## 2. Optimalisasi
Supaya GUI sesuai, ubah resolusi layar menjadi 1366x768 atau yang mendekati
- Cara ubah resolusi layar: Dekstop -> Klik Kanan -> Display settings -> Scroll Kebawah -> Display resolution
Dan untuk hasil deteksi yang maksimal usahakan jarak tangan dengan kamera kurang lebih 2 jengkal

*catatan
saat deteksi huruf, jika huruf yang diinginkan sudah ditampilkan di character 
maka lakukan gesture "next" untuk menyimpan character ke sentence. 
jika huruf yang terdeteksi pada kotak tidak berubah selama setengah detik,
maka huruf yang terdeteksi tersebut akan di simpan di character.

misal ingin membuat kata "saya" maka lakukan:    
"s" "next" "a" "next" "y" "next" "a"

*catatan lagi
program kadang error entah kenapa jadi saya pun tidak tau, jadi kalo gagal coba lagi aja terus jangan menyerah!!!!
kalo bingung liat vid demo aja dah



Referensi:
https://github.com/Devansh-47/Sign-Language-To-Text-and-Speech-Conversion/blob/master/README.md
