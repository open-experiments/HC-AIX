#!/usr/bin/env python3
"""
Ensemble model functionality for breast cancer detection.
"""

import numpy as np
import tensorflow as tf
from models.architecture import build_advanced_model
from utils.memory import clear_memory
import logging

logger = logging.getLogger(__name__)

def ensemble_predict(model_files, x_test, strategy, input_shape=(299, 299, 3), 
                     num_classes=3, batch_size=32):
    """
    Make predictions using an ensemble of models.
    
    Args:
        model_files: List of model weight files
        x_test: Test data
        strategy: TensorFlow distribution strategy
        input_shape: Input shape for the model
        num_classes: Number of output classes
        batch_size: Batch size for prediction
        
    Returns:
        Ensemble predictions
    """
    logger.info(f"Creating ensemble predictions using {len(model_files)} models")
    predictions = []
    
    for i, model_file in enumerate(model_files):
        logger.info(f"Loading model {i+1}/{len(model_files)}: {model_file}")
        
        with strategy.scope():
            # Build a fresh model
            model = build_advanced_model(
                strategy,
                input_shape=input_shape,
                num_classes=num_classes
            )
            
            # Load weights
            model.load_weights(model_file)
            
            # Make predictions
            pred = model.predict(x_test, batch_size=batch_size)
            predictions.append(pred)
            
            # Clear model from memory
            del model
            clear_memory()
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    logger.info("Ensemble predictions created successfully")
    return ensemble_pred


def create_stacked_ensemble(base_models, X_train, y_train, X_val, y_val, strategy, 
                           input_shape=(299, 299, 3), num_classes=3, batch_size=32):
    """
    Create a stacked ensemble model where a meta-model learns to combine predictions.
    
    Args:
        base_models: List of trained base models
        X_train, y_train: Training data
        X_val, y_val: Validation data
        strategy: TensorFlow distribution strategy
        input_shape: Input shape for the models
        num_classes: Number of output classes
        batch_size: Batch size for training
        
    Returns:
        Trained meta-model
    """
    logger.info(f"Creating stacked ensemble from {len(base_models)} base models")
    
    # Get base model predictions on training data
    base_predictions = []
    for i, model in enumerate(base_models):
        logger.info(f"Getting predictions from base model {i+1}/{len(base_models)}")
        pred = model.predict(X_train, batch_size=batch_size)
        base_predictions.append(pred)
    
    # Stack predictions
    stacked_predictions = np.column_stack([p.reshape(p.shape[0], -1) for p in base_predictions])
    
    # Create meta-model
    with strategy.scope():
        meta_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(stacked_predictions.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        meta_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
    
    # Get base model predictions on validation data
    val_base_predictions = []
    for i, model in enumerate(base_models):
        logger.info(f"Getting validation predictions from base model {i+1}/{len(base_models)}")
        pred = model.predict(X_val, batch_size=batch_size)
        val_base_predictions.append(pred)
    
    # Stack validation predictions
    stacked_val_predictions = np.column_stack([p.reshape(p.shape[0], -1) for p in val_base_predictions])
    
    # Train meta-model
    logger.info("Training meta-model")
    meta_model.fit(
        stacked_predictions, y_train,
        validation_data=(stacked_val_predictions, y_val),
        epochs=30,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    logger.info("Stacked ensemble created successfully")
    return meta_model, base_models


def save_ensemble(base_models, meta_model, folder_path):
    """
    Save ensemble models to disk.
    
    Args:
        base_models: List of base models
        meta_model: Meta-model for stacked ensemble
        folder_path: Folder to save models
        
    Returns:
        List of saved model paths
    """
    import os
    os.makedirs(folder_path, exist_ok=True)
    
    # Save base models
    base_model_paths = []
    for i, model in enumerate(base_models):
        path = os.path.join(folder_path, f"base_model_{i+1}.h5")
        model.save_weights(path)
        base_model_paths.append(path)
    
    # Save meta-model
    meta_model_path = os.path.join(folder_path, "meta_model.h5")
    meta_model.save_weights(meta_model_path)
    
    logger.info(f"Saved ensemble models to {folder_path}")
    return base_model_paths, meta_model_path


def load_ensemble(base_model_paths, meta_model_path, strategy, 
                 input_shape=(299, 299, 3), num_classes=3):
    """
    Load ensemble models from disk.
    
    Args:
        base_model_paths: List of paths to base model weights
        meta_model_path: Path to meta-model weights
        strategy: TensorFlow distribution strategy
        input_shape: Input shape for the models
        num_classes: Number of output classes
        
    Returns:
        Loaded base models and meta-model
    """
    from models.architecture import build_advanced_model
    
    # Load base models
    base_models = []
    for i, path in enumerate(base_model_paths):
        logger.info(f"Loading base model {i+1}/{len(base_model_paths)}")
        with strategy.scope():
            model = build_advanced_model(
                strategy,
                input_shape=input_shape,
                num_classes=num_classes
            )
            model.load_weights(path)
            base_models.append(model)
    
    # Load meta-model
    logger.info("Loading meta-model")
    with strategy.scope():
        meta_model = tf.keras.models.load_model(meta_model_path)
    
    logger.info("Ensemble models loaded successfully")
    return base_models, meta_model