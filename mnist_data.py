import numpy as np
import cv2
from zipfile import ZipFile
import os
import urllib
import urllib.request
import matplotlib.pyplot as plt

np.random.seed(0)

'''
This code is from the textbook and allows the user to download the 'Fashion MNIST' dataset as png images
with the proper licensing to use them'
'''

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
    }

# Import the Fashion MNIST dataset
def import_fashion_mnist_dataset():
    URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    FILE = 'fashion_mnist_images.zip'
    FOLDER = 'fashion_mnist_images'
    # Check and see if the images already exist in the directory and return if true
    if os.path.isdir(FOLDER):
        print(f'Dataset already exists in {FOLDER}.')
        return FOLDER
    # Download the files if necessary
    if not os.path.isfile(FILE):
        print(f'Downloading {URL} and saving as {FILE}...')
        urllib.request.urlretrieve(URL, FILE)
    # Unzip the images
    print('Unzipping images...')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)
    print('DONE!')
    return FOLDER

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))
    # Create lists for samples and labels
    X = []
    y = []
    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                        os.path.join(path, dataset, label, file),
                        cv2.IMREAD_UNCHANGED)
            # And append it and a label to the lists
            X.append(image)
            y.append(label)
    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# Shuffle the MINST data to train more more effectively
def shuffle_mnist_data(X, y):
    # Shuffle the data
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]
    return X, y

# Scale and reshape MNIST data to work well in Neural Network
def transform_data(data):
    # Reshape and scale data
    data = (data.reshape(data.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    return data

# MNIST dataset (train + test)
def create_mnist_data(path):
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    # Reshape and scale each dataset
    X = transform_data(X)
    X_test = transform_data(X_test)
    return X, y, X_test, y_test

# Function to import, format, and scale images to test with the model
def format_images(path):
    # Read the image and convert it to grayscale
    image_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to be 28x28 pixels
    image_data = cv2.resize(image_data, (28, 28))
    # Invert the color to match the MNIST data
    image_data = 255 - image_data
    # Apply the scale the image
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
    return image_data

def display_image(path):
    image_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    plt.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
    plt.show()