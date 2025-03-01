#!/usr/bin/env python3
"""
Evaluation metrics for breast cancer detection model.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    cohen_kappa_score, roc_curve, auc, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, x_test, y_test, batch_size=32, class_names=None):
    """
    Evaluate the model performance.
    
    Args:
        model: The trained model
        x_test: Test images
        y_test: Test labels (one-hot encoded)
        batch_size: Batch size for evaluation
        class_names: Names of classes for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    if class_names is None:
        class_names = ["Benign", "Malignant", "Normal"]
    
    logger.info("Evaluating model performance...")
    
    # Get predictions
    predict_y = model.predict(x_test, batch_size=batch_size, verbose=1)
    
    # Check for NaN values in predictions and replace them
    if np.isnan(predict_y).any() or np.isinf(predict_y).any():
        logger.warning("NaN or Inf values found in predictions! Replacing with zeros.")
        predict_y = np.nan_to_num(predict_y)
    
    # Normalize predictions if needed
    row_sums = predict_y.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-3):
        logger.warning("Predictions are not properly normalized, normalizing...")
        predict_y = predict_y / np.maximum(row_sums[:, np.newaxis], 1e-10)
    
    y_pred = np.argmax(predict_y, axis=1)
    
    # Convert one-hot encoded test labels to class indices
    y_test_indices = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_indices, y_pred)
    precision = precision_score(y_test_indices, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_indices, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_indices, y_pred, average='weighted', zero_division=0)
    
    # Calculate per-class metrics
    per_class_precision = precision_score(y_test_indices, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test_indices, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test_indices, y_pred, average=None, zero_division=0)
    
    # Prepare for ROC AUC calculation
    num_classes = len(class_names)
    y_test_bin = label_binarize(y_test_indices, classes=list(range(num_classes)))
    
    # Calculate ROC AUC - Add try-except to handle errors
    try:
        roc_auc = roc_auc_score(y_test_bin, predict_y, multi_class='ovo', average='weighted')
    except ValueError as e:
        logger.error(f"Error calculating ROC AUC: {e}")
        roc_auc = 0.0  # Default value when calculation fails
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test_indices, y_pred)
    
    # Print metrics
    logger.info(f'Accuracy score: {accuracy:.4f}')
    logger.info(f'Precision score: {precision:.4f}')
    logger.info(f'Recall score: {recall:.4f}')
    logger.info(f'F1 Score: {f1:.4f}')
    logger.info(f'ROC AUC Score: {roc_auc:.4f}')
    logger.info(f'Cohen Kappa Score: {cohen_kappa_score(y_test_indices, y_pred):.4f}')
    
    logger.info('\nClassification Report:')
    logger.info(classification_report(y_test_indices, y_pred, target_names=class_names, zero_division=0))
    
    logger.info('\nConfusion Matrix:')
    logger.info(f'{cm}')
    
    # Print per-class metrics
    logger.info("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        if i < len(per_class_precision):
            logger.info(f"{class_name}:")
            logger.info(f"  Precision: {per_class_precision[i]:.4f}")
            logger.info(f"  Recall: {per_class_recall[i]:.4f}")
            logger.info(f"  F1: {per_class_f1[i]:.4f}")
    
    # Store results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'y_test': y_test_indices,
        'y_pred': y_pred,
        'y_pred_proba': predict_y
    }
    
    return results


def calculate_calibration_curve(y_true, y_pred_proba, n_bins=10):
    """
    Calculate calibration curve for model predictions.
    
    Args:
        y_true: True labels (one-hot encoded or indices)
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for the calibration curve
        
    Returns:
        Tuple of (mean_predicted_value, fraction_of_positives, bin_counts)
    """
    # Convert to indices if one-hot encoded
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Initialize arrays
    mean_predicted_value = np.zeros(n_bins)
    fraction_of_positives = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Get class with highest probability for each sample
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    
    # Get probability of predicted class
    y_pred_prob = np.max(y_pred_proba, axis=1)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    # Assign samples to bins
    for i in range(n_bins):
        bin_mask = (y_pred_prob >= bin_boundaries[i]) & (y_pred_prob < bin_boundaries[i + 1])
        bin_counts[i] = np.sum(bin_mask)
        
        if bin_counts[i] > 0:
            # Mean predicted value in bin
            mean_predicted_value[i] = np.mean(y_pred_prob[bin_mask])
            # Fraction of correctly predicted samples in bin
            fraction_of_positives[i] = np.mean(y_true[bin_mask] == y_pred_class[bin_mask])
    
    # Remove empty bins
    mask = bin_counts > 0
    return mean_predicted_value[mask], fraction_of_positives[mask], bin_counts[mask]


def calculate_clinical_metrics(y_true, y_pred, class_names=None):
    """
    Calculate clinical metrics like sensitivity, specificity, PPV, NPV.
    
    Args:
        y_true: True labels (one-hot encoded or indices)
        y_pred: Predicted labels
        class_names: Names of classes
        
    Returns:
        Dictionary with clinical metrics for each class
    """
    # Convert to indices if one-hot encoded
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if class_names is None:
        class_names = ["Benign", "Malignant", "Normal"]
    
    results = {}
    
    # For each class, treat it as the positive class
    for i, class_name in enumerate(class_names):
        # Create binary classification problem
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Store metrics
        results[class_name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv
        }
    
    return results


def evaluate_ensemble(models, x_test, y_test, batch_size=32, strategy=None, class_names=None):
    """
    Evaluate ensemble of models.
    
    Args:
        models: List of models or a single ensemble model
        x_test: Test images
        y_test: Test labels (one-hot encoded)
        batch_size: Batch size for evaluation
        strategy: TensorFlow distribution strategy (optional)
        class_names: Names of classes for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    if class_names is None:
        class_names = ["Benign", "Malignant", "Normal"]
    
    logger.info("Evaluating ensemble performance...")
    
    # If models is a list, get individual predictions and average them
    if isinstance(models, list):
        all_predictions = []
        for i, model in enumerate(models):
            logger.info(f"Getting predictions from model {i+1}/{len(models)}")
            predict_y = model.predict(x_test, batch_size=batch_size, verbose=1)
            all_predictions.append(predict_y)
        
        # Average predictions
        predict_y = np.mean(all_predictions, axis=0)
    else:
        # Single model (possibly an ensemble model)
        predict_y = models.predict(x_test, batch_size=batch_size, verbose=1)
    
    # Continue with regular evaluation
    y_pred = np.argmax(predict_y, axis=1)
    y_test_indices = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_indices, y_pred)
    precision = precision_score(y_test_indices, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_indices, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_indices, y_pred, average='weighted', zero_division=0)
    
    logger.info(f'Ensemble Accuracy: {accuracy:.4f}')
    logger.info(f'Ensemble F1 Score: {f1:.4f}')
    
    # Full evaluation
    return evaluate_model(models if not isinstance(models, list) else None, 
                         x_test, y_test, batch_size, class_names, 
                         precomputed_predictions=predict_y)