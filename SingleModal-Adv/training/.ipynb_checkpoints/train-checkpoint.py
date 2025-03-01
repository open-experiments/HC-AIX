#!/usr/bin/env python3
"""
Training functions for the breast cancer detection model.
"""

import os
import time
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau, TensorBoard
)
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from preprocessing.generators import create_data_generators

logger = logging.getLogger(__name__)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for better handling of class imbalance.
    
    Args:
        gamma: Focusing parameter that controls the weight given to hard misclassified examples
        alpha: Balancing parameter
        
    Returns:
        Focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        # Ensure the predictions are not too close to 0 or 1
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma) * y_true
        loss = alpha * weight * cross_entropy
        
        # Sum over classes and return mean over batch
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    return focal_loss_fixed


def train_model(model, train_generator, val_generator, class_weights=None, epochs=30, 
                steps_per_epoch=None, validation_steps=None, 
                checkpoint_filepath='data/best_weights.weights.h5', 
                logs_dir='logs/training', enable_profiling=False):
    """
    Train the model.
    
    Args:
        model: The model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        class_weights: Optional dictionary of class weights for imbalanced data
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch (if None, calculated from generator)
        validation_steps: Validation steps (if None, calculated from generator)
        checkpoint_filepath: Path to save the best weights
        logs_dir: Directory for TensorBoard logs
        enable_profiling: Whether to enable TensorBoard profiling
        
    Returns:
        Training history
    """
    # Create logs and checkpoint directories if they don't exist
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Define callbacks
    
    # Make sure the checkpoint filepath ends with .weights.h5
    if not checkpoint_filepath.endswith('.weights.h5'):
        checkpoint_filepath = checkpoint_filepath.replace('.h5', '.weights.h5')
    
    # Save the best weights
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_auc',  # Change to AUC for imbalanced data
        mode='max',         # AUC should be maximized
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_auc",  # Change to AUC for imbalanced data
        mode="max",         # AUC should be maximized
        min_delta=0.001,
        patience=8,
        baseline=None,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction with faster reduction for instability
    reduce_lr = ReduceLROnPlateau(
        monitor="val_auc",  # Change to AUC for imbalanced data
        mode="max",         # AUC should be maximized
        factor=0.2,
        patience=3,
        min_delta=0.001,
        cooldown=1,
        min_lr=1e-6,
        verbose=1
    )
    
    # TensorBoard callback for monitoring
    profile_batch = '500,520' if enable_profiling else 0
    
    tensorboard_callback = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1,
        update_freq='epoch',
        profile_batch=profile_batch
    )
    
    # Create a custom F1 score metric
    def f1_score(y_true, y_pred):
        # Cast to the same dtype (float32) to avoid type mismatch
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calculate precision and recall
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
        possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
        
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1
    
    # Add F1 score metric to model
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            f1_score
        ]
    )
    
    # Combine all callbacks
    callbacks = [
        checkpoint_callback,
        early_stopping,
        reduce_lr,
        tensorboard_callback
    ]
    
    logger.info("Starting model training...")
    
    # Calculate steps if not provided
    if steps_per_epoch is None:
        steps_per_epoch = len(train_generator)
    
    if validation_steps is None:
        validation_steps = len(val_generator)
    
    # Time the training process
    start_time = time.time()
    
    # Train the model - removed incompatible parameters
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights  # Keep class weights for imbalanced data
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    
    return history


def train_with_data_splitting(model, X, Y, Img_ids, batch_size=32, epochs=30,
                              handle_imbalance=True, checkpoint_filepath='data/best_weights.weights.h5',
                              logs_dir='logs/training', random_state=42, 
                              cache_dataset=False, enable_profiling=False):
    """
    Train model with data splitting into train, validation, and test sets.
    
    Args:
        model: The model to train
        X: Image data
        Y: Labels (one-hot encoded)
        Img_ids: Image IDs to maintain data independence
        batch_size: Batch size for training
        epochs: Number of training epochs
        handle_imbalance: Whether to handle class imbalance
        checkpoint_filepath: Path to save the best weights
        logs_dir: Directory for TensorBoard logs
        random_state: Random seed for reproducibility
        cache_dataset: Whether to cache dataset in memory
        enable_profiling: Whether to enable profiling
        
    Returns:
        Training history, test data
    """
    logger.info("Preparing data for training...")
    
    # Get unique image IDs for stratification
    unique_ids = np.unique(Img_ids)
    
    # Map each image ID to its class
    id_classes = {}
    for i, img_id in enumerate(Img_ids):
        if img_id not in id_classes:
            id_classes[img_id] = np.argmax(Y[i])
    
    # Create arrays for stratification
    id_Y = np.array([id_classes[id] for id in unique_ids])
    
    # Split IDs into train and test
    train_id_idx, test_id_idx = train_test_split(
        np.arange(len(unique_ids)),
        test_size=0.2,
        random_state=random_state,
        stratify=id_Y
    )
    
    train_ids = set(unique_ids[train_id_idx])
    test_ids = set(unique_ids[test_id_idx])
    
    # Create train and test masks
    train_mask = np.array([img_id in train_ids for img_id in Img_ids])
    test_mask = np.array([img_id in test_ids for img_id in Img_ids])
    
    # Get train and test data
    x_train_all, x_test = X[train_mask], X[test_mask]
    y_train_all, y_test = Y[train_mask], Y[test_mask]
    
    # Get train image IDs
    train_img_ids = Img_ids[train_mask]
    
    # Split train into train and validation
    train_unique_ids = np.unique(train_img_ids)
    train_id_classes = {}
    
    for i, img_id in enumerate(train_img_ids):
        if img_id not in train_id_classes:
            train_id_classes[img_id] = np.argmax(y_train_all[i])
    
    train_unique_Y = np.array([train_id_classes[id] for id in train_unique_ids])
    
    # Split train IDs into train and validation
    final_train_id_idx, val_id_idx = train_test_split(
        np.arange(len(train_unique_ids)),
        test_size=0.2,
        random_state=random_state,
        stratify=train_unique_Y
    )
    
    final_train_ids = set(train_unique_ids[final_train_id_idx])
    val_ids = set(train_unique_ids[val_id_idx])
    
    # Create train and validation masks
    final_train_mask = np.array([img_id in final_train_ids for img_id in train_img_ids])
    val_mask = np.array([img_id in val_ids for img_id in train_img_ids])
    
    # Get final train and validation data
    x_train, x_val = x_train_all[final_train_mask], x_train_all[val_mask]
    y_train, y_val = y_train_all[final_train_mask], y_train_all[val_mask]
    
    logger.info(f"Training set size: {len(x_train)}")
    logger.info(f"Validation set size: {len(x_val)}")
    logger.info(f"Test set size: {len(x_test)}")
    
    # Check class distribution
    train_classes = np.argmax(y_train, axis=1)
    val_classes = np.argmax(y_val, axis=1)
    test_classes = np.argmax(y_test, axis=1)
    
    logger.info(f"Class distribution in training set: {np.bincount(train_classes)}")
    logger.info(f"Class distribution in validation set: {np.bincount(val_classes)}")
    logger.info(f"Class distribution in test set: {np.bincount(test_classes)}")
    
    # Create data generators with class weights for imbalance
    train_generator, val_generator, class_weights = create_data_generators(
        x_train, y_train, x_val, y_val, 
        batch_size=batch_size,
        handle_imbalance=handle_imbalance,
        cache_dataset=cache_dataset
    )
    
    # Use focal loss if class imbalance is severe
    if handle_imbalance:
        class_counts = np.bincount(train_classes)
        imbalance_ratio = np.max(class_counts) / np.min(class_counts)
        
        if imbalance_ratio > 10:
            logger.info(f"Severe class imbalance detected (ratio {imbalance_ratio:.2f}:1). Using focal loss.")
            model.compile(
                optimizer=model.optimizer,
                loss=focal_loss(gamma=2.0, alpha=0.25),
                metrics=model.compiled_metrics._metrics
            )
    
    # Train model
    history = train_model(
        model,
        train_generator,
        val_generator,
        class_weights=class_weights,
        epochs=epochs,
        checkpoint_filepath=checkpoint_filepath,
        logs_dir=logs_dir,
        enable_profiling=enable_profiling
    )
    
    # Return history and test data for evaluation
    return history, (x_test, y_test)


def fine_tune_model(model, train_generator, val_generator, initial_epochs=10, fine_tune_epochs=20,
                   learning_rate=0.0001, checkpoint_filepath='data/fine_tuned_weights.weights.h5',
                   logs_dir='logs/fine_tuning'):
    """
    Fine-tune a pre-trained model by unfreezing all layers and training with a lower learning rate.
    
    Args:
        model: Pre-trained model to fine-tune
        train_generator: Training data generator
        val_generator: Validation data generator
        initial_epochs: Number of epochs the model was previously trained
        fine_tune_epochs: Number of epochs to fine-tune
        learning_rate: Learning rate for fine-tuning (typically lower than initial training)
        checkpoint_filepath: Path to save the best weights
        logs_dir: Directory for TensorBoard logs
        
    Returns:
        Fine-tuning history
    """
    logger.info("Starting model fine-tuning...")
    
    # Unfreeze all layers for fine-tuning
    for layer in model.layers:
        layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # Create callbacks for fine-tuning
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        min_delta=0.0005,  # Smaller delta for fine-tuning
        patience=10,       # More patience for fine-tuning
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.1,        # Larger reduction for fine-tuning
        patience=6,
        min_delta=0.0005,
        min_lr=1e-6,       # Lower minimum learning rate
        verbose=1
    )
    
    tensorboard_callback = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1,
        update_freq='epoch'
    )
    
    callbacks = [
        checkpoint_callback,
        early_stopping,
        reduce_lr,
        tensorboard_callback
    ]
    
    # Time the fine-tuning process
    start_time = time.time()
    
    # Train the model - removed incompatible parameters
    history = model.fit(
        train_generator,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=initial_epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    fine_tuning_time = time.time() - start_time
    logger.info(f"Fine-tuning completed in {fine_tuning_time/60:.2f} minutes")
    
    return history


def export_model_for_serving(model, export_dir):
    """
    Export a trained model for TensorFlow Serving.
    
    Args:
        model: Trained model to export
        export_dir: Directory to save the model
        
    Returns:
        Path to the exported model
    """
    import os
    import time
    
    # Create timestamp for versioning
    timestamp = int(time.time())
    export_path = os.path.join(export_dir, str(timestamp))
    
    # Create directory if not exists
    os.makedirs(export_dir, exist_ok=True)
    
    # Save model with signatures
    signatures = {
        'serving_default': tf.function(
            lambda x: model(x),
            input_signature=[tf.TensorSpec(shape=[None, *model.input_shape[1:]], dtype=tf.float32, name='input_image')]
        ).get_concrete_function()
    }
    
    # Use the optimized SavedModel format
    options = tf.saved_model.SaveOptions(experimental_custom_gradients=False)
    tf.saved_model.save(model, export_path, signatures=signatures, options=options)
    logger.info(f"Model exported for serving to {export_path}")
    
    return export_path