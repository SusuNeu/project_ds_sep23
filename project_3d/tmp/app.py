
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  

import keras
from keras import layers
import nibabel as nib
import random

from scipy import ndimage, stats

# Model definition

def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define input and output of the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model
model = get_model(width=128, height=128, depth=64)


# data preprocessing
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    volume = stats.zscore(volume, axis=None)
    minV = volume.min()
    maxV = volume.max()
    # volume[volume < minV] = minV
    # volume[volume > maxV] = maxV
    volume = (volume - minV) / (maxV - minV)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


# Model Deploy  

import streamlit as st

from PIL import Image


st.header("Model Deploy")
st.text("Upload a sMRI Image for image classification as ADHD or normal")


uploaded_file = st.file_uploader("Upload the sMRI via file path...")
if uploaded_file is not None:

    bytes_data = str(uploaded_file.read().strip().decode('utf-8'))
    st.write(bytes_data)
    x_test = process_scan(bytes_data)
    st.write("Classifying...")

    # Load best weights.
    model.load_weights("3d_image_classification.keras")
    prediction = model.predict(np.expand_dims(x_test, axis=0))[0]
    scores = [1 - prediction[0], prediction[0]]

    class_names = ["normal", "ADHD"]
    for score, name in zip(scores, class_names):
        st.write(
            "This model is %.2f percent confident that the sMRI scan is %s"
                % ((100 * score), name)
                )
        




