from gtts import gTTS
import pygame
import os
import time

def speak_text(text):
    # Buat objek gTTS
    tts = gTTS(text=text, lang='id')
    # Simpan file audio
    audio_file = "output.mp3"
    tts.save(audio_file)
    
    # Inisialisasi pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    # Tunggu hingga pemutaran selesai
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    
    # Tutup pygame mixer dan hapus file audio setelah diputar
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(audio_file)

if __name__ == "__main__":
    while True:
        # Ambil input dari pengguna
        user_input = input("Masukkan teks yang ingin diucapkan (atau ketik 'exit' untuk keluar): ")
        if user_input.lower() == 'exit':
            break
        speak_text(user_input)
