import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

from Lenet import LeNet

def preprocess(img):
    #cv2.imshow("input",img)
    #cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(gray_inv, 35, 255, cv2.THRESH_BINARY)
    # get the indices of the black pixels
    y, x = np.nonzero(binary)
    # get the maximum and minimum x and y coordinates
    max_x = np.max(x)
    min_x = np.min(x)
    max_y = np.max(y)
    min_y = np.min(y)
    nonzero_indices = np.nonzero(binary)
    zero_indices = np.where(binary == 0)
    darkened_gray = gray.astype('float32')
    darkened_gray[zero_indices] = 229+0.1*darkened_gray[zero_indices]
    darkened_gray[nonzero_indices] *= 0.1
    darkened_gray = darkened_gray.astype('uint8')
    cropped = darkened_gray[min_y:max_y, min_x:max_x]
    # _, thresh_img = cv2.threshold(gray_roi, 155, 255, cv2.THRESH_BINARY)
    inverted_img = cv2.bitwise_not(cropped)
    aspect_ratio = inverted_img.shape[1] / inverted_img.shape[0]
    # resize the image while keeping the aspect ratio
    new_height = 20
    new_width = int(new_height * aspect_ratio)
    if new_width > new_height:
        new_width = 20
        aspect_ratio = inverted_img.shape[0] / inverted_img.shape[1]
        new_height = int(new_width * aspect_ratio)
    resized_img = cv2.resize(inverted_img, (new_width, new_height))
    # pad the resized image with black pixels to make it 28x28
    pad_width = (
        ((28 - new_height) // 2, (28 - new_height + 1) // 2),  # no padding on top and bottom
        ((28 - new_width) // 2, (28 - new_width + 1) // 2))  # pad equally on both sides to make the width 28
    padded_img = np.pad(resized_img, pad_width, mode='constant', constant_values=0)
    # img_brighter = cv2.add(inverted_img, 50)
    normalized_img = (cv2.resize(padded_img, (28, 28)))
    # Turn off gradients to speed up this part
    img_tensor = torch.from_numpy(normalized_img).unsqueeze(0)
    normalized_tensor = 2 * (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()) - 1
    # Define a transform to normalize the data
    # Apply the transform to the tensor
    return normalized_tensor


def extract_characters(img_name):
    img = cv2.imread(img_name)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 5)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_SIMPLE)
    fields_points_sorted = []
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
            fields_points_sorted.append([max((y + 3), 0), y + h - 3, max((x + 3), 0), x + w - 3])
            # cv2.imwrite("C:/Users/30698/Desktop/KULeuven/Capita Selecta/test.jpeg", character)
            # show_one_prediction(character,model)
            #cv2.rectangle(img, (x + 3, y + 3), (x + w - 3, y + h - 3), (0, 255, 0), 2)
    fields_points_sorted.sort(key=lambda r: r[2])
    characters_sorted = []
    for field_points in fields_points_sorted:
        characters_sorted.append(img[field_points[0]:field_points[1], field_points[2]:field_points[3]])

    # Display the image with bounding boxes
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return characters_sorted


def show_one_prediction(img, model, i=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(gray_inv, 25, 255, cv2.THRESH_BINARY)
    # get the indices of the black pixels
    y, x = np.nonzero(binary)

    # get the maximum and minimum x and y coordinates
    max_x = np.max(x)
    min_x = np.min(x)
    max_y = np.max(y)
    min_y = np.min(y)
    nonzero_indices = np.nonzero(binary)
    darkened_gray = gray.astype('float32')
    darkened_gray[nonzero_indices] *= 0.3
    darkened_gray[not nonzero_indices] *= 1.1
    darkened_gray = darkened_gray.astype('uint8')
    cropped = darkened_gray[min_y:max_y, min_x:max_x]
    cv2.imshow("bin", cropped)
    cv2.waitKey(0)
    # _, thresh_img = cv2.threshold(gray_roi, 155, 255, cv2.THRESH_BINARY)
    inverted_img = cv2.bitwise_not(cropped)
    aspect_ratio = inverted_img.shape[1] / inverted_img.shape[0]

    # resize the image while keeping the aspect ratio
    new_height = 24
    new_width = int(new_height * aspect_ratio)
    if new_width > new_height:
        return
    resized_img = cv2.resize(inverted_img, (new_width, new_height))

    # pad the resized image with black pixels to make it 28x28
    pad_width = (
        ((28 - new_height) // 2, (28 - new_height + 1) // 2),  # no padding on top and bottom
        ((28 - new_width) // 2, (28 - new_width + 1) // 2))  # pad equally on both sides to make the width 28
    padded_img = np.pad(resized_img, pad_width, mode='constant', constant_values=0)
    # img_brighter = cv2.add(inverted_img, 50)
    normalized_img = (cv2.resize(padded_img, (28, 28)))
    # Turn off gradients to speed up this part
    img_tensor = torch.from_numpy(normalized_img).unsqueeze(0)
    normalized_tensor = 2 * (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()) - 1
    # Define a transform to normalize the data
    # Apply the transform to the tensor
    print(normalized_tensor)
    with torch.no_grad():
        logps = model(normalized_tensor)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    # print(("pred:", probab.index(max(probab))))
    # print(("pred:", ))
    view_classify(img_tensor, ps)
    cv2.imshow("actual", img)
    cv2.waitKey(0)


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.numpy().squeeze(), cmap='gray_r')
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


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
                extract_characters(img_path, model_digits)
