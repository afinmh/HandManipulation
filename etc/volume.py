import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Inisialisasi Pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Fungsi untuk menghitung jarak antara dua titik
def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

# Buka kamera
cap = cv2.VideoCapture(0)

volume_locked = False
zoom_locked = False
zoom_scale = 1.0
zoom_factor = 0.02  # Kecepatan zoom

while True:
    success, img = cap.read()
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
            middle_tip_coords = (int(middle_tip.x * img.shape[1]), int(middle_tip.y * img.shape[0]))
            ring_tip_coords = (int(ring_tip.x * img.shape[1]), int(ring_tip.y * img.shape[0]))
            pinky_tip_coords = (int(pinky_tip.x * img.shape[1]), int(pinky_tip.y * img.shape[0]))

            cv2.circle(img, thumb_tip_coords, 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, index_tip_coords, 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, thumb_tip_coords, index_tip_coords, (255, 0, 0), 3)

            # Menghitung jarak antara ujung ibu jari dan ujung jari telunjuk
            distance = calculate_distance(thumb_tip_coords, index_tip_coords)

            # Deteksi gesture untuk mengunci/membuka kunci volume
            thumb_middle_distance = calculate_distance(thumb_tip_coords, middle_tip_coords)
            thumb_pinky_distance = calculate_distance(thumb_tip_coords, pinky_tip_coords)
            thumb_ring_distance = calculate_distance(thumb_tip_coords, ring_tip_coords)

            if thumb_middle_distance < 40:
                volume_locked = True
                cv2.putText(img, "Volume Locked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            elif thumb_pinky_distance < 40:
                volume_locked = False
                cv2.putText(img, "Volume Unlocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if thumb_ring_distance < 40:
                if not zoom_locked:
                    zoom_locked = True
                    cv2.putText(img, "Zoom Locked", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    zoom_locked = False
                    cv2.putText(img, "Zoom Unlocked", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if not volume_locked:
                vol = min_vol + (max_vol - min_vol) * (distance / 200)  # Sesuaikan skala jarak
                vol = max(min_vol, min(max_vol, vol))  # Batasi nilai volume dalam rentang yang diizinkan
                volume.SetMasterVolumeLevel(vol, None)

            # # Kontrol zoom in dan zoom out
            # if not zoom_locked:
            #     zoom_scale += (distance - 200) * zoom_factor / 100  # Sesuaikan skala jarak
            #     zoom_scale = max(0.5, min(2.0, zoom_scale))  # Batasi skala zoom antara 0.5 dan 2.0

    # Terapkan efek zoom
    height, width, _ = img.shape
    new_height, new_width = int(height * zoom_scale), int(width * zoom_scale)
    img_resized = cv2.resize(img, (new_width, new_height))

    # Crop untuk menjaga ukuran asli
    if zoom_scale > 1:
        start_row = (new_height - height) // 2
        start_col = (new_width - width) // 2
        img_cropped = img_resized[start_row:start_row + height, start_col:start_col + width]
    else:
        img_cropped = img_resized

    # Menampilkan presentasi volume sesuai dengan volume sistem
    current_vol = volume.GetMasterVolumeLevel()
    vol_percentage = int((current_vol - min_vol) / (max_vol - min_vol) * 100)
    cv2.putText(img_cropped, f'Volume: {vol_percentage}%', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.rectangle(img_cropped, (50, 200), (50 + vol_percentage, 250), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(img_cropped, (50, 200), (150, 250), (255, 0, 0), 3)

    cv2.imshow("Hand Gesture Volume Control & Zoom", img_cropped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
