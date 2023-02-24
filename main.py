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


people = [
    ['MICHELLE', 'RODRIGUEZ', '12/02/1950', 'RFMR12025033445', 'DE1202507171'],
    ['SIERRA', 'YU', '29/03/1950', 'RFSY29035047838', 'IT2903504950'],
    ['LAURA', 'MENDOZA', '16/10/1952', 'RFLM16105293143', 'BR1610522746'],
    ['ANGELA', 'FITZGERALD', '04/07/1956', 'RFAF04075656622', 'BE0407563203'],
    ['DAVID', 'LOPEZ', '17/01/1963', 'RFDL17016329220', 'BR1701638007'],
    ['HECTOR', 'COOK', '09/02/1964', 'RFHC09026495520', 'IT0902647355'],
    ['TIMOTHY', 'GREEN', '19/08/1968', 'RFTG19086827450', 'BR1908688556'],
    ['ELIZABETH', 'RIVERA', '08/07/1970', 'RFER08077010123', 'BR0807708065'],
    ['DEREK', 'CHURCH', '13/01/1972', 'RFDC13017297192', 'BL1301728110'],
    ['JEREMIAH', 'TUCKER', '11/10/1972', 'RFJT11107251133', 'BR1110727764'],
    ['LEAH', 'FERGUSON', '05/01/1976', 'RFLF05017691772', 'US0501769434'],
    ['KENNETH', 'STONE', '19/01/1977', 'RFKS19017711799', 'DE1901777829'],
    ['EDWIN', 'ALLEN', '22/09/1977', 'RFEA22097775609', 'DE2209776709'],
    ['ANGEL', 'SCHWARTZ', '18/05/1982', 'RFAS18058263146', 'BR1805822377'],
    ['JARED', 'MURRAY', '26/01/1985', 'RFJM26018598155', 'BR2601853435'],
    ['MELISSA', 'MENDOZA', '11/06/1987', 'RFMM11068724346', 'BL1106877773'],
    ['NICOLE', 'CANTU', '19/03/1989', 'RFNC19038962968', 'DE1903891767'],
    ['JAMES', 'HARRISON', '17/02/1990', 'RFJH17029026206', 'BE1702909215'],
    ['JEFFREY', 'MOONEY', '19/04/1990', 'RFJM19049013684', 'BE1904901793'],
    ['LORI', 'NUNEZ', '28/06/1997', 'RFLN28069748554', 'BR2806974343']
]



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