import cv2
import numpy as np
import easyocr
import pytesseract
import os
from textblob import TextBlob


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds-eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


# Initialize EasyOCR
reader = easyocr.Reader(['en'])

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set path to the photos folder
photos_folder = 'photos'

# Set path to the output file
output_file = 'output.txt'

# Clear the output file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('')

# Loop through all files in the photos folder
for file_name in os.listdir(photos_folder):
    # Check if the file is an image
    if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
        # Load the image
        try:
            image = cv2.imread(os.path.join(photos_folder, file_name))
            print(f'Processing {file_name}')
        except:
            print(f'{file_name} could not be read')
            continue

        # Perform image preprocessing
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        image = resize(image, height=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # Find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None

        # Loop over the contours
        for c in cnts:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # If the approximated contour has four points, then assume
            # that we have found the screen
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is not None:
            # Apply the four point transform to obtain a top-down
            # view of the original image
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

            # Convert the warped image to grayscale, then threshold it
            # to give it that 'black and white' paper effect
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

            # Perform OCR with Tesseract
            tesseract_text = pytesseract.image_to_string(warped)

            # Perform OCR with EasyOCR
            easyocr_result = reader.readtext(warped)
            easyocr_text = ' '.join([res[1] for res in easyocr_result])

            # Determine the better OCR result
            if len(easyocr_text) > len(tesseract_text):
                best_text = easyocr_text
            else:
                best_text = tesseract_text
            text = best_text
        # If the image is already flat, perform normal processing
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            # Apply de-noising to the image
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            # Apply thresholding to the image
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            # Perform OCR with Tesseract
            tesseract_text = pytesseract.image_to_string(orig)

            # Perform OCR with EasyOCR
            easyocr_result = reader.readtext(orig)
            easyocr_text = ' '.join([res[1] for res in easyocr_result])

            # Determine the better OCR result
            if len(easyocr_text) < len(tesseract_text) and (len(easyocr_text) != 0):
                best_text = easyocr_text
            else:
                best_text = tesseract_text
            text = best_text

        text = str(TextBlob(text).correct())
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f'--- {file_name} ---\n')
            f.write(text)
            f.write('\n\n')

os.system(f'start {output_file}')
