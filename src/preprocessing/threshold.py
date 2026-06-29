import cv2

def thresh_plate(plate):

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = cv2.countNonZero(thresh) / float(thresh.shape[0] * thresh.shape[1])

    if white_ratio > 0.5:
        thresh = cv2.bitwise_not(thresh)

    return thresh