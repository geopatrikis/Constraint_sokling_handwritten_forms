import os
import cv2


def extract_characters(img_name):
    img = cv2.imread(img_name)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 5)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        if 30 < w < 100 and 40 < h < 100:
            cv2.rectangle(img, (x+3, y+3), (x + w-3, y + h-3), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# extract file name without extension
if __name__ == '__main__':
    #extract_characters('C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields/0001_DATE.jpg')
    dir_path = 'C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields'
    # Define the allowed image extensions
    img_extensions = [".jpg", ".jpeg", ".png"]

    # Initialize an empty list to store the image paths

    # Iterate over all files and subdirectories in the directory
    for root, dirs, files in os.walk(dir_path):
        # Iterate over all files in the current directory
        for file in files:
            # Check if the file has an image extension
            if any(file.endswith(ext) for ext in img_extensions):
                # Add the full path of the image to the list
                img_path = os.path.join(root, file)
                extract_characters(img_path)