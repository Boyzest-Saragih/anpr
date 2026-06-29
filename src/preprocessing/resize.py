import cv2
import numpy as np

def resize_with_padding(img, target_size=28):
    h, w = img.shape

    # hitung scaling
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize tanpa distorsi
    resized = cv2.resize(img, (new_w, new_h))

    # buat canvas hitam
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)

    # hitung posisi tengah
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    # tempel ke tengah
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas