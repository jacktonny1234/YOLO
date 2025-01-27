import cv2
import pytesseract
import numpy as np
import os
import matplotlib.pyplot as plt

# Set up Tesseract (No need to specify the path, it should work directly)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Commented out
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_car_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and filter for potential license plates
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate = None

    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour is a rectangle
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)

            # Aspect ratio condition (to filter license plates)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5:  # Adjust this according to your dataset
                license_plate = orig_image[y:y+h, x:x+w]
                cv2.rectangle(orig_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break


    if license_plate is not None:
        # Use Tesseract to extract text from the license plate image
        plate_number = pytesseract.image_to_string(license_plate, config='--psm 8')
        return orig_image, plate_number.strip()
    else:
        return orig_image, "License Plate Not Detected"

# Example usage with Kaggle dataset
image_path = 'Data/img.png'  # Replace with your image path
output_image, detected_plate = detect_car_plate(image_path)

# Show the output image with detected license plate bounding box
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()

print(f"Detected License Plate Number: {detected_plate}")