import cv2
import mediapipe as mp
import numpy as np
import joblib
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import screen_brightness_control as sbc
import keyboard
import pyautogui
import time

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Muat model dari file
model_filename = 'D:\Semester Pendek\comvis\model\model_complete.pkl'
knn = joblib.load(model_filename)
gesture_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: 'Next'}

# Inisialisasi pycaw untuk kontrol volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        normalized_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]
        return normalized_landmarks, landmarks
    return None, None

def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def calculate_distances(landmark1, landmark2, frame_width, frame_height):
    x1, y1 = int(landmark1.x * frame_width), int(landmark1.y * frame_height)
    x2, y2 = int(landmark2.x * frame_width), int(landmark2.y * frame_height)
    distance = math.hypot(x2 - x1, y2 - y1)
    return distance

def predict_gesture(image):
    landmarks, raw_landmarks = extract_hand_landmarks(image)
    if landmarks:
        features = np.array(landmarks).flatten().reshape(1, -1)
        return knn.predict(features)[0], raw_landmarks
    return None, None

def draw_hand_box(frame, landmarks, padding=50):
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

# Buka kamera
cap = cv2.VideoCapture(1)
screen_width, screen_height = pyautogui.size()
key = 'a'
drawing = False
erasing = False
prev_index_tip_coords = None
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
clear_canvas = True
last_gesture = None
last_change_time = time.time()
text = ''

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    cv2.putText(frame, f'Text: {text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Prediksi gesture
    if key == 'a':
        gesture, landmarks = predict_gesture(frame)
        if gesture is not None:
            gesture_name = gesture_label.get(gesture, "Unknown")
            top_left, bottom_right = draw_hand_box(frame, landmarks)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, f'Gesture: {gesture_name}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            if gesture != last_gesture:
                last_gesture = gesture
                last_change_time = time.time()
            else:
                # Check if it has been a second since the last change
                if time.time() - last_change_time >= 0.5:
                    idle = gesture_name
                    print(idle)
            if keyboard.is_pressed('n'):
                text += idle
            if keyboard.is_pressed(' '):
                text += ' '

            

        if keyboard.is_pressed('a'):
            key = 'a'
        elif keyboard.is_pressed('b'):
            key = 'b'
        elif keyboard.is_pressed('c'):
            key = 'c'
        elif keyboard.is_pressed('d'):
            key = 'd'
        elif keyboard.is_pressed('v'):
            key = 'v'

    elif key == 'b':
        cv2.putText(frame, 'Gesture: b = Brightness Control', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        brightness_locked = False
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
                if not brightness_locked:
                    brightness = min(100, max(0, int((distance / 200) * 100)))
                    sbc.set_brightness(brightness)
                
                thumb_middle_distance = calculate_distance(thumb_tip_coords, middle_tip_coords)
                thumb_pinky_distance = calculate_distance(thumb_tip_coords, pinky_tip_coords)
                thumb_ring_distance = calculate_distance(thumb_tip_coords, ring_tip_coords)
                
                if thumb_middle_distance < 40:
                    brightness_locked = True
                    cv2.putText(frame, "Brightness Locked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    key = 'a'
                elif thumb_pinky_distance < 40:
                    brightness_locked = False
                    cv2.putText(frame, "Brightness Unlocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Menampilkan presentasi kecerahan
        current_brightness = sbc.get_brightness(display=0)
        brightness_percentage = int(current_brightness[0]) if isinstance(current_brightness, list) else int(current_brightness)
        cv2.putText(frame, f'Brightness: {brightness_percentage}%', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.rectangle(frame, (50, 200), (50 + brightness_percentage, 250), (0, 255, 255), cv2.FILLED)
        cv2.rectangle(frame, (50, 200), (150, 250), (0, 255, 255), 3)
    
    elif key == 'c':
        cv2.putText(frame, 'Gesture: c = Cursor', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        frame_height, frame_width, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Ambil landmark yang ingin Anda gunakan untuk menggerakkan kursor (misalnya, titik 8: ujung jari telunjuk)
                index_finger_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]

                # Konversi koordinat dari nilai relatif (0-1) ke pixel pada layar
                screen_x = int(index_finger_tip.x * screen_width)
                screen_y = int(index_finger_tip.y * screen_height)

                # Pindahkan kursor
                pyautogui.moveTo(screen_x, screen_y)

                # Hitung jarak antara ujung ibu jari dan ujung telunjuk
                distance = calculate_distances(thumb_tip, index_finger_tip, frame_width, frame_height)

                # Jika jarak antara ibu jari dan telunjuk kurang dari ambang batas tertentu, lakukan klik kiri
                if distance < 20:  # Ambang batas dapat disesuaikan
                    pyautogui.click()

                # Gambarkan landmark dan koneksi tangan pada frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if keyboard.is_pressed('a'):
            key = 'a'
    
    elif key == 'd':
        cv2.putText(frame, 'Gesture: d = Drawing', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if clear_canvas:
            canvas.fill(0)  # Mengisi canvas dengan warna hitam
            clear_canvas = False
        
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
                if distance_index_middle > 40 and not erasing:
                    drawing = True
                    if prev_index_tip_coords:
                        cv2.line(canvas, prev_index_tip_coords, index_tip_coords, (255, 0, 0), 5)
                    prev_index_tip_coords = index_tip_coords
                else:
                    drawing = False
                    prev_index_tip_coords = None

                # Jika jari telunjuk, jari tengah, dan jari manis berdekatan, aktifkan penghapus
                if distance_index_middle < 40 and distance_index_ring < 40:
                    erasing = True
                    cv2.circle(canvas, index_tip_coords, 50, (0, 0, 0), -1)
                else:
                    erasing = False

        # Gabungkan kanvas dan gambar asli
        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        if keyboard.is_pressed('a'):
            key = 'a'

    elif key == 'v':
        cv2.putText(frame, 'Gesture: v = Volume Control', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        volume_locked = False
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

                if not volume_locked:
                    vol = min_vol + (max_vol - min_vol) * (distance / 200)  # Sesuaikan skala jarak
                    vol = max(min_vol, min(max_vol, vol))  # Batasi nilai volume dalam rentang yang diizinkan
                    volume.SetMasterVolumeLevel(vol, None)
                
                thumb_middle_distance = calculate_distance(thumb_tip_coords, middle_tip_coords)
                thumb_pinky_distance = calculate_distance(thumb_tip_coords, pinky_tip_coords)
                thumb_ring_distance = calculate_distance(thumb_tip_coords, ring_tip_coords)
                
                if thumb_middle_distance < 40:
                    volume_locked = True
                    cv2.putText(frame, "Volume Locked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    key = 'a'
                elif thumb_pinky_distance < 40:
                    volume_locked = False
                    cv2.putText(frame, "Volume Unlocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        current_vol = volume.GetMasterVolumeLevel()
        vol_percentage = int((current_vol - min_vol) / (max_vol - min_vol) * 100)
        cv2.putText(frame, f'Volume: {vol_percentage}%', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.rectangle(frame, (50, 200), (50 + vol_percentage, 250), (255, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (50, 200), (150, 250), (255, 0, 0), 3)

    # Tampilkan frame dengan prediksi gesture
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
