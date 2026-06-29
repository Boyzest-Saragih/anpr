import cv2

def gaussianFilter(img_gray):
    return cv2.GaussianBlur(img_gray, (3, 3), 0)

def medianFilter(img_gray):
    return cv2.medianBlur(img_gray, 5)

def bilateralFilter(img_gray):
    return cv2.bilateralFilter(img_gray, 9, 75, 75)
