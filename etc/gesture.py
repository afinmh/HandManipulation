import cv2
import os
import mediapipe as mp
import joblib
import numpy as np

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Pengaturan folder penyimpanan
save_dir = 'uji'
image_counter = 0

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Muat model dari file
model_filename = 'model/model_complete1.pkl'
knn = joblib.load(model_filename)
gesture_label = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 
    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: 'next'
}

# Fungsi prediksi gesture dari landmark tangan
def predict_gesture(landmarks):
    features = np.array(landmarks).flatten().reshape(1, -1)
    return knn.predict(features)[0]

# Inisialisasi kamera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    
    # Ubah frame menjadi RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi tangan
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Buat salinan dari frame asli sebelum menggambar landmark dan kotak
            frame_copy = frame.copy()

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

            # Ekstrak landmark untuk prediksi gesture
            normalized_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
            gesture = predict_gesture(normalized_landmarks)
            gesture_name = gesture_label.get(gesture, "Unknown")

            # Tampilkan gesture yang terdeteksi
            cv2.putText(frame, f'Gesture: {gesture_name}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Simpan gambar dalam kotak jika tombol 's' ditekan
            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Potong gambar dalam kotak dari salinan frame asli
                cropped_image = frame_copy[y_min:y_max, x_min:x_max]

                # Simpan gambar tanpa resize
                image_path = os.path.join(save_dir, f'{gesture_name}_{image_counter}.png')
                cv2.imwrite(image_path, cropped_image)
                image_counter += 1
                print(f'Gambar disimpan: {image_path}')
    
    # Tampilkan frame
    cv2.imshow('Gesture Detection', frame)
    
    # Keluar dengan menekan ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Lepaskan resources
cap.release()
cv2.destroyAllWindows()
