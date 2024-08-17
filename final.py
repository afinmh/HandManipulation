import sys
import cv2
import numpy as np
import time
import joblib
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import screen_brightness_control as sbc
from gtts import gTTS
import pygame
import os

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Muat model dari file
model_filename = 'model\model_complete1.pkl'
knn = joblib.load(model_filename)
gesture_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: 'next'}


def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None
        self.capture_image = None
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.volume_locked = False
        self.brightness_locked = False
        self.drawing = False
        self.erasing = False
        self.key = ''
        self.prev_index_tip_coords = None
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.clear_canvas = True
        self.dele.clicked.connect(self.delete)
        self.clear.clicked.connect(self.clr)
        self.camera.currentIndexChanged.connect(self.ganticamera)
        self.check_cam.stateChanged.connect(self.onoffcam)
        self.check_hand.stateChanged.connect(self.onoffhand)
        self.check_draw.stateChanged.connect(self.onoffdraw)
        self.check_speak.stateChanged.connect(self.onoffspeak)
        self.check_bright.stateChanged.connect(self.onoffbright)
        self.warnaai.currentIndexChanged.connect(self.update_color_from_combobox)
        self.warnaai.setEnabled(False)
        self.check_draw.setVisible(False)
        self.check_speak.setVisible(False)
        self.check_hand.setVisible(False)
        self.check_bright.setVisible(False)

        self.last_gesture = None
        self.last_change_time = time.time()
        self.text = ""
        self.speakk.clicked.connect(self.speak)

    def set_drawing_color(self, color):
        self.drawing_color = color

    def update_color_from_combobox(self):
        color_index = self.warnaai.currentIndex()
    
        if color_index == 0:
            self.set_drawing_color((0, 0, 255))  # Merah
        elif color_index == 1:
            self.set_drawing_color((0, 255, 0))  # Hijau
        elif color_index == 2:
            self.set_drawing_color((255, 0, 0))  # Biru
        elif color_index == 3:
            self.set_drawing_color((0, 255, 255))  # Kuning
        elif color_index == 4:
            self.set_drawing_color((128, 0, 128))  # Ungu
        else:
            self.set_drawing_color((0, 0, 0))  # Warna default (Hitam)

    def onoffcam(self, state):
        if state == Qt.Checked:
            camera_index = self.camera.currentIndex()
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                self.idle.setText("Webcam tidak tersedia")
                self.check_draw.setChecked(False)
                self.check_speak.setChecked(False)
                self.check_hand.setChecked(False)
                self.check_bright.setChecked(False)
            else:
                self.idle.setText("Kamera nonaktif")
                self.check_draw.setVisible(True)
                self.check_speak.setVisible(True)
                self.check_hand.setVisible(True)
                self.check_bright.setVisible(True)  
                self.idle.hide()
                self.key = 'c' 
                self.timer.timeout.connect(self.update_frame)
                self.timer.start(10)    
        else:
            self.closecam()
            self.check_draw.setChecked(False)
            self.check_speak.setChecked(False)
            self.check_hand.setChecked(False)
            self.check_bright.setChecked(False)  
            self.check_draw.setVisible(False)
            self.check_speak.setVisible(False)
            self.check_hand.setVisible(False)
            self.check_bright.setVisible(False)       
            self.idle.setText("Kamera nonaktif")
            self.idle.show()

    def ganticamera(self, index):
        if self.check_cam.isChecked():
            self.check_cam.setChecked(False)
            self.check_cam.setChecked(True)

    def closecam(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.Image = None  # Clear the Image 
        self.key = ' '
        self.displayPicture(2)
        self.displayImage(1)

    def onoffhand(self, state):
        if state == Qt.Checked:
            self.check_speak.setEnabled(False)
            self.check_draw.setEnabled(False)
            self.check_bright.setEnabled(False)
            self.key = 'a'
            self.Image = 'icon/picture\gesture.png' 
            self.displayPicture(2)  
            self.timer.start(10)
        else:
            self.key = 'c' 

    def onoffdraw(self, state):
        if state == Qt.Checked:
            self.check_speak.setEnabled(False)
            self.check_hand.setEnabled(False)
            self.check_bright.setEnabled(False)
            self.warnaai.setEnabled(True)
            self.set_drawing_color((0, 0, 255))
            self.key = 'd'
            self.Image = 'icon/picture\draw.png'
            self.displayPicture(2)  
            self.timer.start(10)
        else:
            self.warnaai.setEnabled(False)
            self.key = 'c'

    def onoffspeak(self, state):
        if state == Qt.Checked:
            self.check_draw.setEnabled(False)
            self.check_hand.setEnabled(False)
            self.check_bright.setEnabled(False)
            self.key = 'v'
            self.Image = 'icon/picture\sun.png'
            self.displayPicture(2)  
            self.timer.start(10)
        else:
            self.key = 'c'

    def onoffbright(self, state):
        if state == Qt.Checked:
            self.check_draw.setEnabled(False)
            self.check_hand.setEnabled(False)
            self.check_speak.setEnabled(False)
            self.key = 'b'
            self.Image = 'icon/picture\sun.png'
            self.displayPicture(2)  
            self.timer.start(10)
        else:
            self.key = 'c'

    def speak(self):
        # Tentukan bahasa berdasarkan pilihan di combobox
        if self.bahasa.currentIndex() == 0:
            lang = 'id'  # Bahasa Indonesia
        elif self.bahasa.currentIndex() == 1:
            lang = 'en'  # Bahasa Inggris

        # Periksa apakah teks kosong
        text = self.sentence.text()
        if not text:
            if lang == 'id':
                text = "kata belum dirangkai"
            elif lang == 'en':
                text = "empty"

        # Buat objek gTTS dengan bahasa yang dipilih
        tts = gTTS(text=text, lang=lang)
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

    def clr(self):
        self.text = ""
        self.sentence.setText("")

    def delete(self):
        if self.text:
            self.text = self.text[:-1]
            current_text = self.sentence.text()
            if current_text:
                new_text = current_text[:-1]
                self.sentence.setText(new_text)

    def update_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if ret and self.key == 'c':
            self.check_speak.setEnabled(True)
            self.check_hand.setEnabled(True)
            self.check_draw.setEnabled(True)
            self.check_bright.setEnabled(True)
            self.Image = 'icon/picture\detail.png'
            self.displayPicture(2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Deteksi tangan
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Hitung koordinat kotak di sekitar tangan
                    h, w, c = frame.shape
                    x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w) - 50
                    y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h) - 50
                    x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w) + 50
                    y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h) + 50

                    # Pastikan koordinat berada dalam batas gambar
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)

                    # Gambarkan kotak di sekitar tangan
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Gambar landmark dan koneksi pada tangan
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Tampilkan gambar asli di windows 1
            self.Image = frame
            self.displayImage(1)

        elif ret and self.key == 'a':
            # Frame asli untuk windows 1
            original_frame = frame.copy()

            # Deteksi gesture pada frame asli
            gesture, landmarks = self.predict_gesture(frame)
            if landmarks:
                gesture_name = gesture_label.get(gesture, "Unknown")
                top_left, bottom_right = self.draw_hand_box(frame, landmarks)

                # Gambar kotak tangan dan teks pada frame asli
                cv2.rectangle(original_frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(original_frame, f'Gesture: {gesture_name}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                mp_draw.draw_landmarks(original_frame, landmarks, mp_hands.HAND_CONNECTIONS)

                if gesture != self.last_gesture:
                    self.last_gesture = gesture
                    self.last_change_time = time.time()
                else:
                    if time.time() - self.last_change_time >= 0.5:
                        self.chara.setText(gesture_name)
                        print(gesture_name)

                    # Append text based on gestures
                    if gesture_name == 'next' and self.previous_gesture != 'next':
                        self.text += self.previous_gesture  
                        self.sentence.setText(self.text)
                    elif gesture_name == 'space':
                        self.text += ' '

                    if gesture_name != 'next':
                        self.previous_gesture = gesture_name

                        # Update gesture sebelumnya
                    self.previous_gesture = gesture_name

            # Update QLabel untuk windows 1
            self.Image = original_frame  # Display original frame with hand box on windows 1
            self.displayImage(1)  # For windows 1 (original frame with hand box)

        elif ret and self.key == 'd':
            if self.clear_canvas:
                self.canvas.fill(0)  # Mengisi canvas dengan warna hitam
                self.clear_canvas = False
        
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    index_tip = handLms.landmark[8]
                    middle_tip = handLms.landmark[12]
                    ring_tip = handLms.landmark[16]

                    index_tip_coords = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                    middle_tip_coords = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))
                    ring_tip_coords = (int(ring_tip.x * frame.shape[1]), int(ring_tip.y * frame.shape[0]))

                    # Menghitung jarak antara jari telunjuk dan jari tengah
                    distance_index_middle = calculate_distance(index_tip_coords, middle_tip_coords)
                    # Menghitung jarak antara jari telunjuk dan jari manis
                    distance_index_ring = calculate_distance(index_tip_coords, ring_tip_coords)

                    # Jika jarak antara telunjuk dan tengah jauh, aktifkan menggambar
                    if distance_index_middle > 40 and not self.erasing:
                        self.drawing = True
                        if self.prev_index_tip_coords:
                            cv2.line(self.canvas, self.prev_index_tip_coords, index_tip_coords, self.drawing_color, 5)
                        self.prev_index_tip_coords = index_tip_coords
                    else:
                        self.drawing = False
                        self.prev_index_tip_coords = None

                    # Jika jari telunjuk, jari tengah, dan jari manis berdekatan, aktifkan penghapus
                    if distance_index_middle < 40 and distance_index_ring < 40:
                        self.erasing = True
                        cv2.circle(self.canvas, index_tip_coords, 50, (0, 0, 0), -1)
                    else:
                        self.erasing = False

            # Gabungkan kanvas dan gambar asli
            frame = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
            self.Image = frame
            self.displayImage(1)

        elif ret and self.key == 'v':
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    thumb_tip = handLms.landmark[4]
                    index_tip = handLms.landmark[8]
                    middle_tip = handLms.landmark[12]
                    ring_tip = handLms.landmark[16]
                    pinky_tip = handLms.landmark[20]

                    thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
                    index_tip_coords = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                    middle_tip_coords = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))
                    ring_tip_coords = (int(ring_tip.x * frame.shape[1]), int(ring_tip.y * frame.shape[0]))
                    pinky_tip_coords = (int(pinky_tip.x * frame.shape[1]), int(pinky_tip.y * frame.shape[0]))

                    cv2.circle(frame, thumb_tip_coords, 10, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, index_tip_coords, 10, (255, 0, 0), cv2.FILLED)
                    cv2.line(frame, thumb_tip_coords, index_tip_coords, (255, 0, 0), 3)

                    # Menghitung jarak antara ujung ibu jari dan ujung jari telunjuk
                    distance = calculate_distance(thumb_tip_coords, index_tip_coords)

                    if not self.volume_locked:
                        vol = min_vol + (max_vol - min_vol) * (distance / 200)  # Sesuaikan skala jarak
                        vol = max(min_vol, min(max_vol, vol))  # Batasi nilai volume dalam rentang yang diizinkan
                        volume.SetMasterVolumeLevel(vol, None)
                
                    thumb_middle_distance = calculate_distance(thumb_tip_coords, middle_tip_coords)
                    thumb_pinky_distance = calculate_distance(thumb_tip_coords, pinky_tip_coords)
                    thumb_ring_distance = calculate_distance(thumb_tip_coords, ring_tip_coords)

                    if thumb_middle_distance < 40:
                        self.volume_locked = True
                        cv2.putText(frame, "Volume Locked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    elif thumb_pinky_distance < 40:
                        self.volume_locked = False
                        cv2.putText(frame, "Volume Unlocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            current_vol = volume.GetMasterVolumeLevel()
            vol_percentage = int((current_vol - min_vol) / (max_vol - min_vol) * 100)
            print(current_vol)
            cv2.putText(frame, f'Volume: {vol_percentage}%', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.rectangle(frame, (50, 200), (50 + vol_percentage, 250), (255, 0, 0), cv2.FILLED)
            cv2.rectangle(frame, (50, 200), (150, 250), (255, 0, 0), 3)     
            self.Image = frame
            self.displayImage(1)

        elif ret and self.key == 'b':
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    thumb_tip = handLms.landmark[4]
                    index_tip = handLms.landmark[8]
                    middle_tip = handLms.landmark[12]
                    ring_tip = handLms.landmark[16]
                    pinky_tip = handLms.landmark[20]

                    thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
                    index_tip_coords = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                    middle_tip_coords = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))
                    ring_tip_coords = (int(ring_tip.x * frame.shape[1]), int(ring_tip.y * frame.shape[0]))
                    pinky_tip_coords = (int(pinky_tip.x * frame.shape[1]), int(pinky_tip.y * frame.shape[0]))

                    cv2.circle(frame, thumb_tip_coords, 10, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, index_tip_coords, 10, (255, 0, 0), cv2.FILLED)
                    cv2.line(frame, thumb_tip_coords, index_tip_coords, (255, 0, 0), 3)

                    # Menghitung jarak antara ujung ibu jari dan ujung jari telunjuk
                    distance = calculate_distance(thumb_tip_coords, index_tip_coords)

                    # Kontrol kecerahan
                    if not self.brightness_locked:
                        brightness = min(100, max(0, int((distance / 200) * 100)))
                        sbc.set_brightness(brightness)

                    thumb_middle_distance = calculate_distance(thumb_tip_coords, middle_tip_coords)
                    thumb_pinky_distance = calculate_distance(thumb_tip_coords, pinky_tip_coords)
                
                    if thumb_middle_distance < 40:
                        self.brightness_locked = True
                        cv2.putText(frame, "Brightness Locked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    elif thumb_pinky_distance < 40:
                        self.brightness_locked = False
                        cv2.putText(frame, "Brightness Unlocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Menampilkan presentasi kecerahan
            current_brightness = sbc.get_brightness(display=0)
            brightness_percentage = int(current_brightness[0]) if isinstance(current_brightness, list) else int(current_brightness)
            cv2.putText(frame, f'Brightness: {brightness_percentage}%', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            cv2.rectangle(frame, (50, 200), (50 + brightness_percentage, 250), (0, 255, 255), cv2.FILLED)
            cv2.rectangle(frame, (50, 200), (150, 250), (0, 255, 255), 3)   

            self.Image = frame
            self.displayImage(1)

        else:
            pass

    def predict_gesture(self, image):
        landmarks, raw_landmarks = self.extract_hand_landmarks(image)
        if landmarks:   
            features = np.array(landmarks).flatten().reshape(1, -1)
            return knn.predict(features)[0], raw_landmarks
        return None, None
    
    def extract_hand_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            normalized_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]
            return normalized_landmarks, landmarks
        return None, None

    def draw_hand_box(self, frame, landmarks, padding=10):
        h, w, _ = frame.shape
        x_min, x_max = w, 0 
        y_min, y_max = h, 0
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
        # Tambahkan padding
        x_min = max(x_min - padding, 0)
        x_max = min(x_max + padding, w)
        y_min = max(y_min - padding, 0)
        y_max = min(y_max + padding, h)
        return (x_min, y_min), (x_max, y_max)

    def displayImage(self, windows):
        if windows == 1 and self.Image is not None:
            if isinstance(self.Image, str):  # Check if Image is a file path
                img = QImage(self.Image)
            else:  # Otherwise, it's an image array
                qformat = QImage.Format_RGB888
                img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
                img = img.rgbSwapped()
            pixmap = QPixmap.fromImage(img)
            self.vid.setPixmap(pixmap)
            self.vid.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.vid.setScaledContents(True)
        else:
            self.vid.clear()

    def displayPicture(self, windows):
        if windows == 2 and self.Image is not None:
            if isinstance(self.Image, str):  # If it's a path
                image_path = self.Image
                img = QImage(image_path)
            else:  # If it's an image array
                qformat = QImage.Format_RGB888
                img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
                img = img.rgbSwapped()
            pixmap = QPixmap.fromImage(img)
            self.picture.setPixmap(pixmap)
            self.picture.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.picture.setScaledContents(True)
        else:
            self.picture.clear()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        event.accept()

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Final Hand')
window.showMaximized()
sys.exit(app.exec_())
