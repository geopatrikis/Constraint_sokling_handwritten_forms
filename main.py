import os

import cv2

correctly: int = 0

def take_fields_and_save_them(image_mame):
    global correctly
    file_name = os.path.splitext(os.path.basename(image_mame))[0]
    # split file name from the end and take the last element
    image_number = file_name.split("_")[-1]
    # print image number
    print(image_number)
    # Load the image
    img = cv2.imread(image_mame)
    img_cropped = img[240:, :]
    scale_factor = 2
    resized_img = cv2.resize(img_cropped,
                             (int(img_cropped.shape[1] / scale_factor), int(img_cropped.shape[0] / scale_factor)))
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to create a binary image
    thresh_value = 220  # Threshold value to be used
    max_value = 255  # Maximum value for pixels above the threshold
    binary_img = cv2.adaptiveThreshold(gray_img, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25,
                                       10)
    # Apply morphological operations to remove noise and connect nearby letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Loop through the contours
    i = 0
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out small and large contours that are unlikely to be letters
        if 300 < w < 1200 and 18 < h < 200:
            # Draw a rectangle around the contour
            roi = resized_img[max((y - 2),0):y + h + 2, max((x - 2),0):x + w + 2]
            filename = ""
            # Save the ROI as a separate image
            if i == 0:
                filename = "C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields/" + image_number + "_NAT.jpg"
            elif i == 1:
                filename = "C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields/" + image_number + "_RF.jpg"
            elif i == 2:
                filename = "C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields/" + image_number + "_DATE.jpg"
            elif i == 3:
                filename = "C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields/" + image_number + "_LN.jpg"
            elif i == 4:
                filename = "C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields/" + image_number + "_FN.jpg"
            if i <= 4 and filename != "" and roi is not None:
                try:
                    cv2.imwrite(filename, roi)
                except cv2.error as e:
                    print('Error: Unable to write the image to a file:', e)
            else:
                break
            i += 1
    # Display the image with bounding boxes
    if i == 5:
        print("Done for image: " + image_number)
        correctly += 1
    else:
        print("Error------------------------" + image_number)
    cv2.destroyAllWindows()


# extract file name without extension
if __name__ == '__main__':
    dir_path = "C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/24022023"
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
                take_fields_and_save_them(img_path)
    print(str(correctly))