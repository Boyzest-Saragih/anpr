from ultralytics import YOLO
from ..preprocessing.threshold import thresh_plate
from ..preprocessing.morphology import morph_closing
from pathlib import Path
import cv2

MODEL_PATH = (
    Path(__file__).resolve()
    .parents[2]
    / "models"
    / "char_plate.pt"
)

model = YOLO(str(MODEL_PATH))

def find_char_candidates_yolo(plate_img):
    char_candidates = []

    if len(plate_img.shape) == 2:
        img_for_predict = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)
    else:
        img_for_predict = plate_img

    results = model.predict(img_for_predict, verbose=False)

    result = results[0]

    # Ambil SEMUA box karakter yang berhasil dideteksi YOLO
    for box in result.boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        w = xmax - xmin
        h = ymax - ymin
        char_candidates.append((int(xmin), int(ymin), int(w), int(h)))

    # Sort karakter dari kiri ke kanan berdasarkan koordinat X (xmin)
    char_candidates = sorted(char_candidates, key=lambda b: b[0])

    return char_candidates


def extract_characters(plate_bgr):
    binary_plate = thresh_plate(plate_bgr)

    results = yolo_char_plate.predict(plate_bgr, verbose=False)
    result = results[0]

    char_images = []
    char_boxes = []

    for box in result.boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        char_boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))

    char_boxes = sorted(char_boxes, key=lambda b: b[0])

    # Potong dari gambar binary_plate, bukan dari gambar warna
    for xmin, ymin, xmax, ymax in char_boxes:
        # Tambahkan sedikit margin/padding
        h_p, w_p = binary_plate.shape
        y1, y2 = max(0, ymin), min(h_p, ymax)
        x1, x2 = max(0, xmin), min(w_p, xmax)

        char_crop = binary_plate[y1:y2, x1:x2]

        # Apply morphological closing
        closed_char = morph_closing(char_crop)
        char_images.append(closed_char)

    return char_images