import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blur_bilateral(image_gray):
    return cv2.bilateralFilter(image_gray, 9, 75, 75)

def automatic_canny_thresholds(image):
    median = np.median(image)

    # Atur threshold berdasarkan median
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    return lower, upper

def crop_plate(img_input):

    # 2. Deteksi Tepi Adaptif
    lower, upper = automatic_canny_thresholds(img_input)
    edged = cv2.Canny(img_input, lower, upper)

    # 3. Temukan Kontur
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

    results = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500: # Abaikan kontur yang terlalu kecil (noise)
            continue

        # Pendekatan poligon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(approx)

        # 5. Filter Rasio Aspek Plat (Kunci Utama)
        # Plat nomor Indonesia (motor) biasanya sekitar 25cm x 10cm, rasio ~2.5:1
        # Kita beri toleransi, misalnya 2.0:1 hingga 4.0:1
        aspect_ratio = float(w) / h

        # Jika kontur memiliki 4-8 sudut (toleransi untuk plat miring/terpotong)
        if 4 <= len(approx) <= 8:
            if 2.0 < aspect_ratio < 4.5:
                candidate = img_input[y:y+h, x:x+w]
                results.append(((x, y, w, h), candidate))

    if results:
        best_candidate_loc, best_candidate_img = results[0]
        return best_candidate_img, results

    return None, []

def localize_plate(blur_bilateral):
    plate_img, all_candidates = crop_plate(blur_bilateral)


    img_detected = cv2.cvtColor(blur_bilateral, cv2.COLOR_BGR2RGB).copy()
    if all_candidates:
        for (x, y, w, h), _ in all_candidates:
            cv2.rectangle(img_detected, (x, y), (x+w, y+h), (0, 255, 0), 3)
        (x, y, w, h), _ = all_candidates[0]
        cv2.rectangle(img_detected, (x, y), (x+w, y+h), (255, 0, 0), 5)

    return img_detected, plate_img


def plate_img(plate_img):
    if plate_img is not None:
        plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        return plate_img_rgb
    else:
        return "Tidak ditemukan plat nomor yang valid."


def thresholding(plate_img):
    thresh_binary = cv2.adaptiveThreshold(plate_img, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
    return thresh_binary

def morphological_operations(thresh_binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Closing dulu untuk menutup celah/lubang di dalam karakter
    closing = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)

    # Baru dilate untuk menebalkan
    final_result = cv2.dilate(closing, kernel, iterations=1)
    return final_result

def find_contours(morph_result):
    contours, _ = cv2.findContours(morph_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_characters(contours, plate_morph):
    char_candidates = []

    plate_h, plate_w = plate_morph.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        ratio = h / w
        area = w * h

        if (
            50 < h < 120 and
            10 < w < 100 and
            1 < ratio < 5 and
            500 < area < 5000 and
            y < plate_h * 0.8
        ):
            char_candidates.append((x, y, w, h))

    char_candidates = sorted(char_candidates, key=lambda b: b[0])
    return char_candidates

def detected_characters(char_candidates, final_result):
    img_with_boxes = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
    for (x, y, w, h) in char_candidates:
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img_with_boxes

def extract_characters(char_candidates, final_result):
    characters = []

    for (x, y, w, h,*_) in char_candidates:
        char = final_result[y:y+h, x:x+w]
        characters.append(char)

    return characters

def extract_characters_with_pd(char_candidates, final_result):
    characters = []

    for (x, y, w, h) in char_candidates:
        pad = 2

        y1 = max(y - pad, 0)
        y2 = min(y + h + pad, final_result.shape[0])
        x1 = max(x - pad, 0)
        x2 = min(x + w + pad, final_result.shape[1])

        char = final_result[y1:y2, x1:x2]

        characters.append(char)
    return characters

def resize_characters(characters):
    resized_characters = []
    for char in characters:
        resized = cv2.resize(char, (28, 28), interpolation=cv2.INTER_AREA)
        resized_characters.append(resized)
    return resized_characters


def resize_with_padding(img, target_size=28):
    h, w = img.shape

    # hitung scaling
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize tanpa distorsi
    resized = cv2.resize(img, (new_w, new_h))

    # buat canvas hytam
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)

    # hitung posisi mid
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    # tempel ke mid
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def visualize_characters(characters):

    final_characters = []

    for i, char in enumerate(characters):
        plt.subplot(1, len(characters), i+1)
        char = resize_with_padding(char)
        final_characters.append(char)
    
    return final_characters
