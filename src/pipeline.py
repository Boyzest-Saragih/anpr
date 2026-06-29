from .detection import yolo_plate
from .detection import yolo_char_plate
from .preprocessing import resize
from .ocr import cnn_ocr

def recognize_plate(image):
    # 1. Deteksi plat
    plate, box = yolo_plate.crop_plate_yolo(image)

    if plate is None:
        raise ValueError("Plat kendaraan tidak ditemukan.")

    # 2. Deteksi karakter
    character_images = yolo_char_plate.extract_characters(plate)

    if len(character_images) == 0:
        raise ValueError("Karakter pada plat tidak ditemukan.")

    # 3. Resize karakter
    processed_characters = []

    for char in character_images:
        resized = resize.resize_with_padding(char)
        processed_characters.append(resized)

    plate_number, confidence = cnn_ocr.predict_plate_number(processed_characters)

    return {
        "plate_image": plate,
        "characters": processed_characters,
        "plate_number": plate_number,
        "confidence": confidence 
    }