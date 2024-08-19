import cv2
import mediapipe as mp
import os

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Buat folder untuk menyimpan gambar jika belum ada
save_dir = 'uji'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Ukuran gambar yang disimpan
output_size = (512, 512)

# Buka kamera
cap = cv2.VideoCapture(1)
image_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Konversi frame ke RGB
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

            # Gambar landmark dan koneksi pada tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            # Simpan gambar dalam kotak jika tombol 's' ditekan
            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Potong gambar dalam kotak dari salinan frame asli
                cropped_image = frame_copy[y_min:y_max, x_min:x_max]

                # Resize gambar ke ukuran yang diinginkan
                resized_image = cv2.resize(cropped_image, output_size)

                # Simpan gambar
                image_path = os.path.join(save_dir, f'a{image_counter}.png')
                cv2.imwrite(image_path, cropped_image)
                image_counter += 1
                print(f'Gambar disimpan: {image_path}')

    # Tampilkan frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()