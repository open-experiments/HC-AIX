#!/usr/bin/env python3
"""
Model architecture definitions for breast cancer detection.
"""

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Flatten, BatchNormalization, Dense, 
    Activation, Dropout, Input, 
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)

def build_advanced_model(strategy, input_shape=(299, 299, 3), num_classes=3, initial_lr=0.001):
    """
    Build an advanced model with InceptionV3 as base.
    
    Args:
        strategy: TensorFlow distribution strategy
        input_shape: Input image shape
        num_classes: Number of output classes
        initial_lr: Initial learning rate
        
    Returns:
        Compiled model
    """
    with strategy.scope():
        # Create input layer
        inputs = Input(shape=input_shape)
        
        # Load pre-trained InceptionV3 model
        base_model = InceptionV3(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False
        )
        
        # Freeze initial layers
        for layer in base_model.layers[:249]:
            layer.trainable = False
        
        # Fine-tune final layers
        for layer in base_model.layers[249:]:
            layer.trainable = True
        
        # Use global average pooling to reduce parameters
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        
        # Add classification head with dropout for regularization
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="BreastCancerInceptionV3")
        
        # Compile with Adam optimizer, additional metrics, and gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_lr,
            clipnorm=1.0  # Add gradient clipping to prevent exploding gradients
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    
    logger.info(f"Built InceptionV3 model with input shape {input_shape} and {num_classes} output classes")
    return model


def build_lightweight_model(strategy, input_shape=(299, 299, 3), num_classes=3, initial_lr=0.001):
    """
    Build a lightweight model for deployment on constrained environments.
    Uses MobileNetV2 instead of InceptionV3 for a smaller footprint.
    
    Args:
        strategy: TensorFlow distribution strategy
        input_shape: Input image shape
        num_classes: Number of output classes
        initial_lr: Initial learning rate
        
    Returns:
        Compiled model
    """
    with strategy.scope():
        # Create input layer
        inputs = Input(shape=input_shape)
        
        # Load pre-trained MobileNetV2 model (smaller than InceptionV3)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Build model
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="BreastCancerMobileNet")
        
        # Compile with Adam optimizer
        optimizer = Adam(learning_rate=initial_lr)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    
    logger.info(f"Built lightweight MobileNetV2 model with input shape {input_shape} and {num_classes} output classes")
    return model


def build_ensemble_model(models, strategy, input_shape=(299, 299, 3), num_classes=3):
    """
    Build an ensemble model that averages predictions from multiple models.
    
    Args:
        models: List of pre-trained models
        strategy: TensorFlow distribution strategy
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Ensemble model
    """
    with strategy.scope():
        # Create input layer
        inputs = Input(shape=input_shape)
        
        # Get outputs from each model
        outputs = []
        for i, model in enumerate(models):
            # Use the model without its final layer
            x = model(inputs)
            outputs.append(x)
        
        # Average predictions if there are multiple models
        if len(outputs) > 1:
            avg_output = tf.keras.layers.Average()(outputs)
        else:
            avg_output = outputs[0]
        
        # Create model
        ensemble_model = Model(inputs=inputs, outputs=avg_output, name="BreastCancerEnsemble")
    
    logger.info(f"Built ensemble model from {len(models)} base models")
    return ensemble_model