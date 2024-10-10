import cv2


def subtract(arr1, arr2):
    return [a - b for a, b in zip(arr1, arr2)]


def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[1] if len(contours) == 2 else contours[0] \
        if len(contours) == 1 else None
