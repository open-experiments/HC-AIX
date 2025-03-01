#!/usr/bin/env python3
"""
Cross-validation training functionality for breast cancer detection.
"""

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from preprocessing.generators import create_data_generators
from training.train import train_model
from utils.memory import clear_memory
from evaluation.visualization import (
    plot_training_history,
    plot_roc_curves,
    plot_confusion_matrix
)
import logging

logger = logging.getLogger(__name__)

def train_with_cross_validation(X, Y, Img_ids, strategy, model_builder_fn, config):
    """
    Train model with cross-validation.
    
    Args:
        X: Image data
        Y: Labels (one-hot encoded)
        Img_ids: Image IDs to maintain data independence
        strategy: TensorFlow distribution strategy
        model_builder_fn: Function to build a model
        config: Configuration dictionary
        
    Returns:
        List of model weights files, evaluation results, and history objects
    """
    n_folds = config.get('n_folds', 5)
    random_state = config.get('random_state', 42)
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 30)
    handle_imbalance = config.get('handle_imbalance', True)
    results_dir = config.get('results_dir', 'results')
    logs_dir = config.get('logs_dir', 'logs/training')
    
    logger.info(f"Starting {n_folds}-fold cross-validation training...")
    
    # Convert one-hot encoded Y to class indices for stratification
    Y_indices = np.argmax(Y, axis=1)
    
    # Get unique image IDs for stratification
    # This ensures that augmented versions of the same image don't go to different splits
    unique_ids = np.unique(Img_ids)
    unique_id_to_idx = {id: idx for idx, id in enumerate(unique_ids)}
    
    # Map each image ID to its class
    id_classes = {}
    for i, img_id in enumerate(Img_ids):
        if img_id not in id_classes:
            id_classes[img_id] = Y_indices[i]
    
    # Create arrays for stratification
    id_indices = np.array([unique_id_to_idx[id] for id in unique_ids])
    id_Y = np.array([id_classes[id] for id in unique_ids])
    
    # Initialize stratified K-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Store results for each fold
    fold_weights_files = []
    fold_results = []
    fold_histories = []
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    for fold, (train_id_idx, test_id_idx) in enumerate(skf.split(id_indices, id_Y)):
        logger.info(f"\n-------------- Fold {fold+1}/{n_folds} --------------")
        
        # Get train and test image IDs
        train_ids = set(unique_ids[train_id_idx])
        test_ids = set(unique_ids[test_id_idx])
        
        # Create train and test masks
        train_mask = np.array([img_id in train_ids for img_id in Img_ids])
        test_mask = np.array([img_id in test_ids for img_id in Img_ids])
        
        # Split data
        x_train_fold, x_test_fold = X[train_mask], X[test_mask]
        y_train_fold, y_test_fold = Y[train_mask], Y[test_mask]
        
        logger.info(f"Training set size: {len(x_train_fold)}")
        logger.info(f"Test set size: {len(x_test_fold)}")
        
        # Further split training into train/val
        train_img_ids = Img_ids[train_mask]
        unique_train_ids = np.unique(train_img_ids)
        unique_train_id_to_idx = {id: idx for idx, id in enumerate(unique_train_ids)}
        
        # Map each training image ID to its class
        train_id_classes = {}
        for i, img_id in enumerate(train_img_ids):
            if img_id not in train_id_classes:
                train_id_classes[img_id] = np.argmax(y_train_fold[i])
        
        # Create arrays for stratification
        train_id_indices = np.array([unique_train_id_to_idx[id] for id in unique_train_ids])
        train_id_Y = np.array([train_id_classes[id] for id in unique_train_ids])
        
        # Split train into train/val
        train_id_idx, val_id_idx = train_test_split(
            np.arange(len(unique_train_ids)), 
            test_size=0.2, 
            random_state=random_state,
            stratify=train_id_Y
        )
        
        # Get train and val image IDs
        final_train_ids = set(unique_train_ids[train_id_idx])
        val_ids = set(unique_train_ids[val_id_idx])
        
        # Create train and val masks
        final_train_mask = np.array([img_id in final_train_ids for img_id in train_img_ids])
        val_mask = np.array([img_id in val_ids for img_id in train_img_ids])
        
        # Get final train and val sets
        x_train, x_val = x_train_fold[final_train_mask], x_train_fold[val_mask]
        y_train, y_val = y_train_fold[final_train_mask], y_train_fold[val_mask]
        
        logger.info(f"Final training set size: {len(x_train)}")
        logger.info(f"Validation set size: {len(x_val)}")
        
        # Check class distribution
        train_classes = np.argmax(y_train, axis=1)
        val_classes = np.argmax(y_val, axis=1)
        test_classes = np.argmax(y_test_fold, axis=1)
        
        logger.info(f"Class distribution in training set: {np.bincount(train_classes)}")
        logger.info(f"Class distribution in validation set: {np.bincount(val_classes)}")
        logger.info(f"Class distribution in test set: {np.bincount(test_classes)}")
        
        # Create data generators with class weights for imbalance
        train_generator, val_generator, class_weights = create_data_generators(
            x_train, y_train, x_val, y_val, 
            batch_size=batch_size,
            handle_imbalance=handle_imbalance
        )
        
        # Build model
        model = model_builder_fn(
            strategy,
            input_shape=(*config.get('image_size', (299, 299)), 3),
            num_classes=config.get('num_classes', 3),
            initial_lr=config.get('initial_lr', 0.001)
        )
        
        if fold == 0:  # Only print summary for first fold to avoid repetition
            model.summary()
        
        # Create checkpoint file path
        checkpoint_filepath = os.path.join(results_dir, f"weights_fold{fold+1}.weights.h5")
        fold_weights_files.append(checkpoint_filepath)
        
        # Create logs directory for this fold
        fold_logs_dir = os.path.join(logs_dir, f"fold{fold+1}")
        
        # Train model
        history = train_model(
            model,
            train_generator,
            val_generator,
            class_weights=class_weights,
            epochs=epochs,
            checkpoint_filepath=checkpoint_filepath,
            logs_dir=fold_logs_dir
        )
        
        # Save history
        fold_histories.append(history)
        
        # Plot training history
        history_plot_path = os.path.join(results_dir, f"history_fold{fold+1}.png")
        plot_training_history(history, save_path=history_plot_path)
        
        # Load best weights
        model.load_weights(checkpoint_filepath)
        
        # Evaluate model
        from evaluation.metrics import evaluate_model
        
        fold_result = evaluate_model(
            model, 
            x_test_fold, 
            y_test_fold, 
            batch_size=batch_size,
            class_names=config.get('class_names', ["Benign", "Malignant", "Normal"])
        )
        
        # Plot ROC curves
        roc_plot_path = os.path.join(results_dir, f"roc_fold{fold+1}.png")
        plot_roc_curves(
            y_test_fold,
            fold_result['y_pred_proba'],
            num_classes=config.get('num_classes', 3),
            class_names=config.get('class_names', ["Benign", "Malignant", "Normal"]),
            save_path=roc_plot_path
        )
        
        # Plot confusion matrix
        cm_plot_path = os.path.join(results_dir, f"cm_fold{fold+1}.png")
        plot_confusion_matrix(
            fold_result['confusion_matrix'],
            class_names=config.get('class_names', ["Benign", "Malignant", "Normal"]),
            save_path=cm_plot_path
        )
        
        # Save results
        fold_results.append(fold_result)
        
        # Clear memory
        clear_memory()
    
    # Calculate average metrics
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'roc_auc': np.mean([r['roc_auc'] for r in fold_results])
    }
    
    logger.info("\nAverage metrics across folds:")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f} (+/- {np.std([r[metric] for r in fold_results]):.4f})")
    
    # Per-class metrics
    class_names = config.get('class_names', ["Benign", "Malignant", "Normal"])
    for i, class_name in enumerate(class_names):
        per_class_f1 = np.mean([r['per_class_f1'][i] for r in fold_results])
        per_class_precision = np.mean([r['per_class_precision'][i] for r in fold_results])
        per_class_recall = np.mean([r['per_class_recall'][i] for r in fold_results])
        
        logger.info(f"\n{class_name} metrics:")
        logger.info(f"  Precision: {per_class_precision:.4f}")
        logger.info(f"  Recall: {per_class_recall:.4f}")
        logger.info(f"  F1: {per_class_f1:.4f}")
    
    return fold_weights_files, fold_results, fold_histories