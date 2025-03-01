#!/usr/bin/env python
# Authors: Fatih E. NAR (Red Hat), Sedat Kaymaz (Microsoft)
#
"""
Mammography Image Classification using InceptionV3
This script processes mammography images from the MIAS dataset and classifies them
as Benign (B), Malignant (M), or Normal (NORM) using a fine-tuned InceptionV3 model.
"""

# Section 1 - Import Libraries
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import scipy
import os
import warnings
warnings.filterwarnings('ignore')
from livelossplot import PlotLossesKeras
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from skimage.io import *
import time
from tqdm import tqdm
from colorama import Fore
import json

# Set environment variables before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Specify which GPUs to use
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import (
    Flatten, BatchNormalization, Dense, Activation, Dropout
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, cohen_kappa_score, roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb


# Section 2 - Helper Functions
def read_info_txt(file, bg_to_process, class_to_process):
    """
    Read and process information from the MIAS dataset info.txt file.
    
    Parameters:
    file (str): Path to the info.txt file
    bg_to_process (list): List of background tissue types to process
    class_to_process (list): List of abnormality classes to process
    
    Returns:
    tuple: (DataFrame with processed information, Dictionary mapping references to rows)
    """
    print("Starting to read in file:", file)
    rows = []
    mmi = {}
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            # Skip non-data lines
            if not line.startswith("mdb"):
                continue
            
            parts = line.split()
            
            # Prepare a dict for each row:
            row_dict = {
                "REF": None,        # e.g. "mdb001"
                "BG": None,         # background tissue: F/G/D
                "CLASS": None,      # abnormality class: CALC/CIRC/SPIC/MISC/ARCH/ASYM/NORM
                "SEVERITY": None,   # B or M, if present
                "X": None,
                "Y": None,
                "RADIUS": None
            }
            
            # 1) REF = the first item, e.g. "mdb001"
            row_dict["REF"] = parts[0]
            # 2) BG (background tissue) = second item, e.g. "G"
            row_dict["BG"] = parts[1]
            # 3) CLASS (abnormality) = third item, e.g. "CIRC" or "NORM"
            row_dict["CLASS"] = parts[2]

            # Skip if the background tissue is not in the list of tissues to process
            if row_dict["BG"] not in bg_to_process:
                continue
            if row_dict["CLASS"] not in class_to_process:
                continue
            
            # If there's exactly 3 parts, that means something like "mdb003 D NORM"
            if len(parts) == 3:
                # No severity/coords
                rows.append(row_dict)
                mmi[row_dict["REF"]] = row_dict
                continue
            
            # 4) If there's a 4th part, it's typically severity (B/M)
            row_dict["SEVERITY"] = parts[3]
            
            # Some lines might stop at 4 parts (e.g. "mdb059 F CIRC B")
            if len(parts) == 4:
                rows.append(row_dict)
                mmi[row_dict["REF"]] = row_dict
                continue
            
            # 5) If we have at least 7 parts, we have x,y,radius
            if len(parts) >= 7:
                row_dict["X"] = parts[4]
                row_dict["Y"] = parts[5]
                row_dict["RADIUS"] = parts[6]
                # If there are more than 7 parts, it typically indicates multiple abnormalities
                # or more radius definitions. For simplicity, we just capture the first set here.
            
            mmi[row_dict["REF"]] = row_dict
            rows.append(row_dict)

    # Create the DataFrame
    pd_info = pd.DataFrame(rows)
    return pd_info, mmi


def clear_memory():
    """Free memory after finishing with the model"""
    tf.keras.backend.clear_session()
    import gc
    gc.collect()
    print("TensorFlow session cleared and Python garbage collected.")


def get_roi_coords(mmi, img):
    """
    Extract coordinates of region of interest from info.txt file
    
    Parameters:
    mmi (dict): Dictionary containing image metadata
    img (str): Image reference ID
    
    Returns:
    tuple: (class, severity, x, y, radius) information for the ROI
    """
    if mmi[img]['CLASS'] == 'NORM':
        return mmi[img]['CLASS'], None, None, None, None
    elif mmi[img]['CLASS'] == 'CIRC':
        if mmi[img]['X'] is not None:
            BouM = mmi[img]['SEVERITY']
            x = int(mmi[img]['X'])
            y = int(mmi[img]['Y'])
            radius = int(mmi[img]['RADIUS'])
            return mmi[img]['CLASS'], BouM, x, y, radius
        else:
            BouM = mmi[img]['SEVERITY']
            return mmi[img]['CLASS'], BouM, None, None, None

    return None, None, None, None, None


def read_lables(mmi, no_angles, angle_interval):
    """
    Read labels from the metadata dictionary
    
    Parameters:
    mmi (dict): Dictionary containing image metadata
    no_angles (int): Number of angles for rotation
    angle_interval (int): Interval between angles
    
    Returns:
    dict: Dictionary mapping image IDs to labels for each angle
    """
    print("Reading labels...")
    info = {}

    for key, value in mmi.items():
        img = key
        if mmi[img]['CLASS'] == 'NORM':
            info[img] = {angle: 2 for angle in range(0, no_angles, angle_interval)}  # Label "Normal" -> {0: 2}
        elif mmi[img]['SEVERITY'] == 'B':
            info[img] = {angle: 0 for angle in range(0, no_angles, angle_interval)}  # Label "Benign" -> {0: 0}
        elif mmi[img]['SEVERITY'] == 'M':
            info[img] = {angle: 1 for angle in range(0, no_angles, angle_interval)}  # Label "Malignant" -> {0: 1}

    print('..The number of read lables:%d' % len(mmi))
    return info


def read_rotate_flip_image3(mmi, url, no_angles, angle_interval):
    """
    Read, rotate, and flip images to augment dataset
    
    Parameters:
    mmi (dict): Dictionary containing image metadata
    url (str): Base path to image files
    no_angles (int): Number of angles for rotation
    angle_interval (int): Interval between angles
    
    Returns:
    dict: Dictionary mapping image IDs to processed images for each angle
    """
    print("Read, rotate, flip images..")
    info = {}
    for key, value in mmi.items():
        image_name = key
        image_address = url + image_name + '.pgm'
        
        class_label, BouM, x_center, y_center, radius = get_roi_coords(mmi, image_name)

        img = cv2.imread(image_address, 1)
        if class_label == "CIRC" and x_center is not None and y_center is not None and radius is not None:
            # Crop the region of interest (ROI)
            x1 = max(x_center - radius, 0)
            y1 = max(y_center - radius, 0)
            x2 = min(x_center + radius, img.shape[1])
            y2 = min(y_center + radius, img.shape[0])
            roi = img[y1:y2, x1:x2]
            # Resize based on ROI
            img = cv2.resize(roi, (224, 224))
        else:
            # Just resize
            img = cv2.resize(img, (224, 224))
      
        rows, cols, channel = img.shape
        info[image_name] = {}

        # Rotation + Flip
        for angle in range(0, no_angles, angle_interval):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_rotated = cv2.warpAffine(img, M, (cols, rows))
            img_flipped = cv2.flip(img_rotated, 1)
            info[image_name][angle] = img_flipped
    
    print('..The number of read Image:%d' % len(mmi))
    return info


# Custom Activation Layer
class GompertzReLU(tf.keras.layers.Layer):
    """
    Custom activation function combining ReLU with Gompertz function
    """
    def __init__(self, a=1.0, b=1.0, c=1.0):
        super(GompertzReLU, self).__init__()
        self.a = a
        self.b = b
        self.c = c

    def call(self, inputs):
        return tf.where(inputs >= 0, inputs, self.a * tf.exp(-self.b * tf.exp(-self.c * tf.abs(inputs))))


# Main execution function
def main():
    # Variables
    bg_to_process = ['G', 'D', 'F']
    class_to_process = ['CIRC', 'NORM']
    file_path = "data/images/Info.txt"

    # Definition of the number of angles for rotations
    no_angles = 360
    angle_interval = 8
    url = 'data/images/'

    # Set the memory growth for the GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth and memory limit set for all GPUs")
        except RuntimeError as e:
            print(f"Error in GPU config: {e}")

    # Get the info from the file
    info, mmi = read_info_txt(file_path, bg_to_process, class_to_process)
    print(info.head())

    # Get the labels
    lable_info = read_lables(mmi, no_angles, angle_interval)

    # Get the images
    image_info = read_rotate_flip_image3(mmi, url, no_angles, angle_interval)

    # Get the ids
    ids = lable_info.keys()

    # Prepare data for training
    X = []
    Y = []
    for id in ids:
        for angle in range(0, no_angles, angle_interval):
            X.append(image_info[id][angle])
            Y.append(lable_info[id][angle])
    
    X = np.array(X)
    Y = np.array(Y)
    Y = to_categorical(Y, 3)

    # Split the data into training and testing sets
    x_train, x_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test1, y_test1, test_size=0.3, random_state=42)

    # Print dataset details
    print("Dataset Details")
    print(f"Original Image numbers Image={len(image_info)} Label={len(lable_info)}")
    print(f"After rotate,flip with {no_angles//angle_interval} angles: X={len(X)},Y={len(Y)}")
    print(f"Dataset TRAINING: x_train={len(x_train)}, TESTING(Validation+Test) x_test1={len(x_test1)}")
    print(f"Dataset TESTING: Validation={len(x_val)}, Test={len(x_test)}")

    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=6,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=6,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0.001
    )

    # Create custom activation function
    gompertzrelu_activation = GompertzReLU(a=1.0, b=1.0, c=1.0)

    # Model training
    print("Starting to train the model without Augmentation")

    # Load the InceptionV3 model without its final classification layer
    base_Neural_Net = InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

    # Create the Sequential model and add layers
    model = Sequential()
    model.add(base_Neural_Net)

    # Add custom layers for classification
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation(gompertzrelu_activation))  # Custom activation function
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    # Freeze the base model's (InceptionV3) weights
    for layer in base_Neural_Net.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])

    # Define the path to save the weights
    checkpoint_filepath = 'best_weights1.weights.h5'

    # Callback to save the best weights
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # Callback for tracking losses with PlotLossesKeras
    plot_losses = PlotLossesKeras()

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=16,
        callbacks=[plot_losses, checkpoint_callback]
    )

    # Evaluate the model
    evaluation_results = model.evaluate(x_val, y_val, callbacks=[early_stopping, reduce_lr], batch_size=16)
    print(f"Validation Loss: {evaluation_results[0]}, Validation Accuracy: {evaluation_results[1]}")

    # Performance reporting
    print("Performance Report:")

    def predict_prob(number):
        return [number[0], 1 - number[0]]

    predict_y = model.predict(x_test)
    y_pred7 = np.argmax(predict_y, axis=1)
    y_test7 = [np.argmax(x) for x in y_test]
    y_pred_prb7 = model.predict(x_test)

    target = ["B", "M", "NORM"]

    print('Accuracy score is :', np.round(accuracy_score(y_test7, y_pred7), 4))
    print('Precision score is :', np.round(precision_score(y_test7, y_pred7, average='weighted'), 4))
    print('Recall score is :', np.round(recall_score(y_test7, y_pred7, average='weighted'), 4))
    print('F1 Score is :', np.round(f1_score(y_test7, y_pred7, average='weighted'), 4))

    # Ensure that y_test is binary/multi-label
    y_test_bin = label_binarize(y_test7, classes=[0, 1, 2])  # One-hot encoding of labels (0=B, 1=M, 2=NORM)

    # Calculate the ROC AUC for a multiclass problem
    roc_auc = roc_auc_score(y_test_bin, y_pred_prb7, multi_class='ovo', average='weighted')
    print('ROC AUC Score is :', np.round(roc_auc, 4))

    print('\t\tClassification Report:\n', classification_report(y_test7, y_pred7, target_names=target))
    print('Cohen Kappa Score:', np.round(cohen_kappa_score(y_test7, y_pred7), 4))

    # Print class distribution
    unique_classes, class_counts = np.unique(y_test7, return_counts=True)
    print("Classes present in y_test:", unique_classes)
    print("Number of occurrences per class:", class_counts)

    # Calculate ROC curves and AUC for each class
    roc_results = {}
    fpr = {}
    tpr = {}
    thresholds = {}

    for i in range(3):  # Number of classes
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_bin[:, i], y_pred_prb7[:, i])
        roc_results[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curve for each class
    plt.figure(figsize=(8, 6))

    colors = ['darkorange', 'green', 'blue']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, 
                 label=f'Class {i} ROC curve (area = {roc_results[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('roc_curves.png')
    plt.close()

    # Save the model
    model.save('mammography_classification_model.h5')
    
    # Free memory
    model = None
    clear_memory()


if __name__ == "__main__":
    main()
