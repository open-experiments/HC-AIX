#!/usr/bin/env python3
"""
Visualization utilities for breast cancer detection model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)

def plot_roc_curves(y_test, y_pred_probs, num_classes=3, class_names=None, save_path=None):
    """
    Plot ROC curves for each class.
    
    Args:
        y_test: True labels (one-hot encoded)
        y_pred_probs: Predicted probabilities
        num_classes: Number of classes
        class_names: Names of the classes for the legend
        save_path: Path to save the figure
    """
    if class_names is None:
        class_names = ["Benign", "Malignant", "Normal"]
    
    # Convert to class indices if one-hot encoded
    if len(y_test.shape) > 1:
        y_test_indices = np.argmax(y_test, axis=1)
    else:
        y_test_indices = y_test
        
    # Convert to one-hot for ROC calculation
    y_test_bin = label_binarize(y_test_indices, classes=list(range(num_classes)))
    
    # Calculate ROC curves and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    colors = ['darkorange', 'green', 'blue']
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
        )
    
    # Diagonal line (random chance)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of the classes
        save_path: Path to save the figure
    """
    if class_names is None:
        class_names = ["Benign", "Malignant", "Normal"]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Training history
        save_path: Path to save the figure
    """
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    ax1.grid(alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.close()
    
    # Plot additional metrics if available
    if 'auc' in history.history and 'val_auc' in history.history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot AUC
        ax1.plot(history.history['auc'])
        ax1.plot(history.history['val_auc'])
        ax1.set_title('Model AUC')
        ax1.set_ylabel('AUC')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='lower right')
        ax1.grid(alpha=0.3)
        
        # Plot Precision/Recall if available
        if 'precision' in history.history and 'val_precision' in history.history:
            ax2.plot(history.history['precision'])
            ax2.plot(history.history['val_precision'])
            ax2.set_title('Model Precision')
            ax2.set_ylabel('Precision')
            ax2.set_xlabel('Epoch')
            ax2.legend(['Train', 'Validation'], loc='lower right')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            metrics_save_path = save_path.replace('.png', '_metrics.png')
            plt.savefig(metrics_save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Additional metrics plot saved to {metrics_save_path}")
        
        plt.close()


def display_sample_images(x_data, y_data, n_samples=4, class_names=None, save_dir=None):
    """
    Display or save a sample of images from the dataset.
    
    Args:
        x_data: Image data
        y_data: Labels (one-hot encoded)
        n_samples: Number of samples per class to display
        class_names: Names of classes
        save_dir: Directory to save images (if None, images are displayed)
    """
    if class_names is None:
        class_names = ["Benign", "Malignant", "Normal"]
    
    logger.info(f"Displaying {n_samples} sample images per class")
    
    # Convert to class indices if one-hot encoded
    if len(y_data.shape) > 1:
        y_indices = np.argmax(y_data, axis=1)
    else:
        y_indices = y_data
    
    # Get indices for each class
    class_indices = {}
    for i in range(len(class_names)):
        class_indices[i] = np.where(y_indices == i)[0]
        np.random.shuffle(class_indices[i])  # Shuffle to get random samples
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Display/save samples from each class
    for class_idx, class_name in enumerate(class_names):
        if class_idx in class_indices and len(class_indices[class_idx]) > 0:
            # Get sample indices for this class
            samples = class_indices[class_idx][:min(n_samples, len(class_indices[class_idx]))]
            
            for i, idx in enumerate(samples):
                plt.figure(figsize=(4, 4))
                
                # Denormalize if needed
                img = x_data[idx]
                if img.max() <= 1.0:
                    img = img * 255
                    
                plt.imshow(img.astype(np.uint8))
                plt.title(f"Class: {class_name}")
                plt.axis('off')
                
                if save_dir:
                    save_path = os.path.join(save_dir, f"{class_name}_{i+1}.png")
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()


def plot_calibration_curve(mean_predicted_value, fraction_of_positives, save_path=None):
    """
    Plot calibration curve for the model.
    
    Args:
        mean_predicted_value: Mean predicted probability in each bin
        fraction_of_positives: Fraction of positive samples in each bin
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Plot calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    
    # Plot perfect calibration (diagonal line)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration curve saved to {save_path}")
    
    plt.close()


def plot_feature_maps(model, img, layer_names=None, save_dir=None):
    """
    Plot feature maps from intermediate layers for a given input image.
    
    Args:
        model: Trained model
        img: Input image to visualize (should be preprocessed)
        layer_names: List of layer names to visualize (if None, uses conv layers)
        save_dir: Directory to save visualizations (if None, displays them)
    """
    import tensorflow as tf
    
    # Expand dimensions for batch
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    # Create a model that will output feature maps
    if layer_names is None:
        # Find convolutional layers
        layer_names = []
        for layer in model.layers:
            if 'conv' in layer.name.lower():
                layer_names.append(layer.name)
    
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get feature maps
    activations = activation_model.predict(img)
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot feature maps
    for i, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
        n_features = layer_activation.shape[-1]
        if n_features > 64:
            # If too many features, just show first 64
            n_features = 64
        
        # Calculate grid size
        size = int(np.ceil(np.sqrt(n_features)))
        
        # Create figure
        fig, ax = plt.subplots(size, size, figsize=(size*2, size*2))
        fig.suptitle(f"Feature maps for layer '{layer_name}'")
        
        # Plot each feature map
        for row in range(size):
            for col in range(size):
                feature_idx = row * size + col
                if feature_idx < n_features:
                    ax[row, col].imshow(layer_activation[0, :, :, feature_idx], cmap='viridis')
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f"layer_{i+1}_{layer_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()