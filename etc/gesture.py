import cv2
import threading

def show_camera(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Tekan 'Esc' untuk menutup jendela secara manual
                break
        else:
            print("Tidak bisa mengambil frame dari kamera.")
    cap.release()
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)  # Langsung membuka kamera saat program dimulai
    camera_open = cap.isOpened()

    if camera_open:
        print("Kamera dibuka. Ketik 'close' untuk menutup kamera atau 'exit' untuk keluar.")
        threading.Thread(target=show_camera, args=(cap,)).start()
    else:
        print("Gagal membuka kamera.")
        return

    while True:
        command = input("Ketik 'close' untuk menutup kamera, 'open' untuk membuka kamera kembali, atau 'exit' untuk keluar: ").strip().lower()
        
        if command == 'close':
            if camera_open:
                cap.release()
                cv2.destroyAllWindows()
                camera_open = False
                print("Kamera ditutup.")
            else:
                print("Kamera sudah ditutup.")
        
        elif command == 'open':
            if not camera_open:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    camera_open = True
                    print("Kamera dibuka kembali.")
                    threading.Thread(target=show_camera, args=(cap,)).start()
                else:
                    print("Gagal membuka kamera.")
            else:
                print("Kamera sudah dibuka.")
        
        elif command == 'exit':
            if camera_open:
                cap.release()
            cv2.destroyAllWindows()
            print("Program selesai.")
            break
        
        else:
            print("Perintah tidak dikenali. Ketik 'open', 'close', atau 'exit'.")

if __name__ == "__main__":
    main()
