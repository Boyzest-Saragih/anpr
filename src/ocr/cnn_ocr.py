import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve()
    .parents[2]
    / "models"
    / "cnn_model.keras"
)


# mapping kelas angka dan charr
class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]

def predict_plate_number(list_potongan_karakter):
    model = tf.keras.models.load_model(str(MODEL_PATH))
    hasil_plat_nomor = ""
    list_confidence = []

    # 3. Iterasi dan prediksi setiap potongan karakter
    for img_karakter in list_potongan_karakter:
        # Pastikan ukuran img_karakter adalah (32, 32)
        # Tambah dimensi batch dan channel agar sesuai input CNN: (1, 32, 32, 1)
        img_input = np.expand_dims(img_karakter, axis=0)      # Menjadi (1, 32, 32)
        img_input = np.expand_dims(img_input, axis=-1)     # Menjadi (1, 32, 32, 1)
        
        # Normalisasi intensitas piksel ke rentang 0-1 (jika belum dilakukan di preprocessing)
        if img_input.max() > 1.0:
            img_input = img_input / 255.0
            
        # Prediksi menggunakan model
        prediksi = model.predict(img_input, verbose=0)
        
        # Ambil indeks kelas dengan probabilitas tertinggi
        indeks_terbaik = np.argmax(prediksi)
        karakter_tebakan = class_names[indeks_terbaik]
        
        confidence = prediksi[0][indeks_terbaik] * 100

        # Konversi indeks menjadi karakter string
        karakter_tebakan = class_names[indeks_terbaik]
        
        # Gabungkan karakter ke string hasil akhir
        hasil_plat_nomor += karakter_tebakan
        list_confidence.append(round(confidence, 2))
        
    avg_confidence = round(np.mean(list_confidence), 2) if list_confidence else 0.0
    return hasil_plat_nomor,avg_confidence