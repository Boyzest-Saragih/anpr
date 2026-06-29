import cv2

def morph_closing(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Closing dulu untuk menutup celah/lubang di dalam karakter
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Baru dilate untuk menebalkan
    final_result = cv2.dilate(closing, kernel, iterations=1)

    return final_result