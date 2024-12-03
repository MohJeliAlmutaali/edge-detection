import cv2
import numpy as np
from matplotlib import pyplot as plt

# Fungsi utama untuk deteksi tepi
def deteksi_tepi(gambar_input, metode='Canny'):
    # Baca gambar
    gambar = cv2.imread(gambar_input, cv2.IMREAD_COLOR)
    if gambar is None:
        raise ValueError("Gambar tidak ditemukan. Periksa jalur file gambar.")

    # Konversi gambar ke skala abu-abu
    gambar_gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # Deteksi tepi menggunakan metode yang dipilih
    if metode == 'Canny':
        tepi = cv2.Canny(gambar_gray, 100, 200)
    elif metode == 'Sobel':
        grad_x = cv2.Sobel(gambar_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gambar_gray, cv2.CV_64F, 0, 1, ksize=3)
        tepi = cv2.magnitude(grad_x, grad_y)
        tepi = np.uint8(tepi)
    else:
        raise ValueError("Metode deteksi tepi tidak dikenali. Pilih 'Canny' atau 'Sobel'.")

    # Tampilkan gambar asli dan hasil deteksi tepi
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB))
    plt.title("Gambar Asli")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(tepi, cmap='gray')
    plt.title("Deteksi Tepi")
    plt.axis("off")

    plt.show()

    # Simpan hasil
    cv2.imwrite("hasil_deteksi_tepi.png", tepi)
    print("Gambar hasil deteksi tepi telah disimpan sebagai 'hasil_deteksi_tepi.png'.")

# Jalankan fungsi utama
if __name__ == "__main__":
    # Ganti 'gambar.jpg' dengan jalur file gambar Anda
    deteksi_tepi('gambar.jpg', metode='Canny')
