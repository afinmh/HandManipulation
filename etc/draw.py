import cv2
import mediapipe as mp
import math
import numpy as np

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Fungsi untuk menghitung jarak antara dua titik
def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

# Buka kamera
cap = cv2.VideoCapture(0)

drawing = False
erasing = False
prev_index_tip_coords = None
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
clear_canvas = True

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        break

    if clear_canvas:
        # Setiap kali clear_canvas True, kita set ulang canvas menjadi warna hitam
        canvas.fill(0)  # Mengisi canvas dengan warna hitam
        clear_canvas = False
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            index_tip = handLms.landmark[8]
            middle_tip = handLms.landmark[12]
            ring_tip = handLms.landmark[16]

            index_tip_coords = (int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0]))
            middle_tip_coords = (int(middle_tip.x * img.shape[1]), int(middle_tip.y * img.shape[0]))
            ring_tip_coords = (int(ring_tip.x * img.shape[1]), int(ring_tip.y * img.shape[0]))

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
    img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

    cv2.imshow("Hand Drawing", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
