import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
%matplotlib inline
import cv2
from skimage.io import imread, imshow
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dropout, Lambda
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1

NUM_TEST_IMAGES = 10

# Data paths
train_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/train'
val_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/val'
test_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/test'

def load_and_preprocess_data(data_dir):
    images = []
    masks = []

    for image_filename in os.listdir(os.path.join(data_dir, 'images')):
        image_path = os.path.join(data_dir, 'images', image_filename)
        mask_path = os.path.join(data_dir, 'masks', image_filename)

        # Read and preprocess image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # Read and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask >= 10).astype(np.float32)  # Ensure the correct data type
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        mask = mask / 255.0  # Normalize to [0, 1]
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)


def visualize_data(images, masks, num_samples=5):
    for i in range(num_samples):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(images[i])
        plt.title('Input Image')

        plt.subplot(1, 2, 2)
        plt.imshow(masks[i][:, :, 0], cmap='viridis')
        plt.title('Mask')
        plt.show()

# Load and preprocess training data
train_images, train_masks = load_and_preprocess_data(train_data_dir)

# Load and preprocess validation data
val_images, val_masks = load_and_preprocess_data(val_data_dir)

# Load and preprocess test data
test_images, test_masks = load_and_preprocess_data(test_data_dir)

# U-Net model
def build_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Decoder
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2)
    up1 = concatenate([up1, conv1], axis=3)

    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Build the U-Net model
model = build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
model.summary()

# Training callbacks
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ModelCheckpoint('model.h5', save_best_only=True, verbose=1)
]

# Train the model
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=50,
    batch_size=16,
    callbacks=callbacks
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_masks)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
