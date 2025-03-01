# Mammography Classification with Google's BiT

A custom implementation of a breast cancer classification system using the MIAS mammography dataset and Google's BiT (Big Transfer) model.

## Overview

This project implements a deep learning approach to classify mammography images into three categories:
- Benign lesions
- Malignant lesions
- Normal tissue

The implementation uses Google's BiT (Big Transfer) model, with a transfer learning approach that has shown excellent performance on medical imaging tasks with limited data.

## Features

- **Robust Data Processing**: Includes ROI (Region of Interest) handling for abnormalities
- **Balanced Data Sampling**: Implements stratified sampling with augmentation of minority classes
- **Transfer Learning**: Utilizes Google's BiT model pre-trained on large image datasets
- **Regularization Techniques**: Implements multiple approaches to prevent overfitting
- **Comprehensive Evaluation**: Includes ROC curves, confusion matrices, and class-wise metrics

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- TensorFlow Hub
- OpenCV
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Dataset

The notebook is designed to work with the MIAS (Mammographic Image Analysis Society) dataset, which contains 322 digitized mammograms with associated ground truth annotations. The dataset includes:

- Images of normal breast tissue
- Images with benign abnormalities
- Images with malignant abnormalities

Each abnormality is annotated with:
- X,Y coordinates of the center
- Approximate radius
- Classification (Benign/Malignant)

## Notebook Structure

The notebook is organized into the following sections:

1. **Imports and Configuration**: Sets up the necessary libraries and GPU settings
2. **Utility Functions**: Provides functions for data loading and preprocessing
3. **Data Preparation**: Loads and processes the MIAS dataset with balanced sampling
4. **Model Definition**: Creates the BiT model with appropriate regularization
5. **Model Training**: Implements training with callbacks to prevent overfitting
6. **Model Evaluation**: Comprehensive evaluation of model performance
7. **Visualization**: Detailed visualization of results and model performance

## Usage

1. Download the MIAS dataset and place it in the `data/images/` directory
2. Ensure the `Info.txt` file is located at `data/images/Info.txt`
3. Run the notebook cells in sequence
4. The trained model will be saved as `bit_mammography_final.keras`
5. Performance visualizations will be saved as PNG files

## Model Architecture

The model uses Google's BiT-M-R50x1 as a backbone, which is a ResNet50-based architecture that has been pre-trained on large image datasets. The model includes:

- Pre-trained BiT feature extractor (with fine-tuning)
- Batch normalization layers
- Dense layers with L2 regularization
- Dropout for reducing overfitting
- Softmax output for 3-class classification

## Performance

The model achieves competitive performance metrics:
- High accuracy and AUC on the balanced test set
- Good generalization across all three classes
- Effective handling of the class imbalance problem

## Acknowledgments

- The MIAS dataset is provided by the Mammographic Image Analysis Society
- The BiT model is developed by Google Research

## References

1. Suckling J et al. The Mammographic Image Analysis Society Digital Mammogram Database. Exerpta Medica. International Congress Series 1069 pp375-378, 1994.
2. Kolesnikov, A., Beyer, L., Zhai, X., et al. "Big Transfer (BiT): General Visual Representation Learning." ECCV 2020.
