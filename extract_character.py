import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

from Lenet import LeNet


def extract_characters(img_name,model):
    img = cv2.imread(img_name)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 5)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        threshold_value = 200
        black_found = False
        character = img[max((y + 3), 0):y + h - 3, max((x + 3), 0):x + w - 3]
        # Iterate over all pixels and check if their intensity values are above the threshold
        for row in range(character.shape[0]):
            for col in range(character.shape[1]):
                pixel_intensity = character[row, col]
                if (pixel_intensity < threshold_value).any():
                    black_found = True
                    break
            if black_found:
                break
        # Get the bounding rectangle of the contour
        if 30 < w < 100 and 40 < h < 100 and black_found:
            show_one_prediction(character,model)
            cv2.rectangle(img, (x + 3, y + 3), (x + w - 3, y + h - 3), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_one_prediction(img, model, i=0):
    gray_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_roi = cv2.resize(gray_roi, (28, 28))/255
    # Turn off gradients to speed up this part
    img_tensor = torch.from_numpy(resized_roi).unsqueeze(0).float()
    # Define a transform to normalize the data
    # Apply the transform to the tensor
    img_tensor_normalized = (img_tensor - 0.5) / -0.5
    print(img_tensor_normalized)
    with torch.no_grad():
        logps = model(img_tensor_normalized)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print(("pred:",probab.index(max(probab))))
    cv2.imshow("actual",img)
    cv2.waitKey(0)

def predict_letter(img,model):
    print(img.shape)
    cv2.imshow('image', img)
    gray_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_roi = cv2.resize(gray_roi, (28, 28))
    normalized_roi = resized_roi / 255.0
    #pil_image = Image.fromarray((normalized_roi * 255).astype('uint8'))
    logpros=model(normalized_roi)
    cv2.waitKey(0)
    # Apply thresholding to make the image binary
    # Use Tesseract OCR to recognize text
    # ABCDEFGHIJKLMNOPQRSTUVWXYZ
    # Print the recognized text


# extract file name without extension
if __name__ == '__main__':
    model_digits = LeNet()
    model_digits.load_state_dict(torch.load('saved_models/digits_model_nn'))
    # extract_characters('C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields/0001_DATE.jpg')
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
                extract_characters(img_path,model_digits)
