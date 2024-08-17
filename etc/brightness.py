import cv2
import mediapipe as mp
import math
import screen_brightness_control as sbc

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Fungsi untuk menghitung jarak antara dua titik
def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

# Buka kamera
cap = cv2.VideoCapture(0)

brightness_locked = False

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            thumb_tip = handLms.landmark[4]
            index_tip = handLms.landmark[8]
            middle_tip = handLms.landmark[12]
            ring_tip = handLms.landmark[16]
            pinky_tip = handLms.landmark[20]

            thumb_tip_coords = (int(thumb_tip.x * img.shape[1]), int(thumb_tip.y * img.shape[0]))
            index_tip_coords = (int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0]))

            cv2.circle(img, thumb_tip_coords, 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, index_tip_coords, 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, thumb_tip_coords, index_tip_coords, (255, 0, 0), 3)

            # Menghitung jarak antara ujung ibu jari dan ujung jari telunjuk
            distance = calculate_distance(thumb_tip_coords, index_tip_coords)

            # Kontrol kecerahan
            if not brightness_locked:
                brightness = min(100, max(0, int((distance / 200) * 100)))
                sbc.set_brightness(brightness)

    # Menampilkan presentasi kecerahan
    current_brightness = sbc.get_brightness(display=0)
    brightness_percentage = int(current_brightness[0]) if isinstance(current_brightness, list) else int(current_brightness)
    cv2.putText(img, f'Brightness: {brightness_percentage}%', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.rectangle(img, (50, 200), (50 + brightness_percentage, 250), (0, 255, 255), cv2.FILLED)
    cv2.rectangle(img, (50, 200), (150, 250), (0, 255, 255), 3)

    # Cetak ukuran frame
    print(f"Ukuran frame: {img.shape[1]} x {img.shape[0]}")

    cv2.imshow("Hand Gesture Brightness Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
