#!/usr/bin/env python3
"""
Data generator utilities for the training pipeline.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import logging

logger = logging.getLogger(__name__)

def prepare_dataset(image_info, label_info, no_angles, angle_interval, test_size=0.2, random_state=42):
    """
    Prepare the dataset for training and testing.
    
    Args:
        image_info: Dictionary with image data
        label_info: Dictionary with label data
        no_angles: Maximum rotation angle
        angle_interval: Rotation step size
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, Y, Img_ids) for the complete dataset
    """
    # Get the image IDs
    ids = list(label_info.keys())
    
    X = []
    Y = []
    Img_ids = []  # Keep track of which image each sample came from
    
    # Combine images and labels
    for id in ids:
        for angle in range(0, no_angles, angle_interval):
            if id in image_info and angle in image_info[id]:
                X.append(image_info[id][angle])
                Y.append(label_info[id][angle])
                Img_ids.append(id)  # Store the image ID
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y)
    Img_ids = np.array(Img_ids)
    
    # Check for NaN values in X
    if np.isnan(X).any():
        logger.warning("NaN values found in image data! Replacing with zeros.")
        X = np.nan_to_num(X)
    
    # One-hot encode labels
    Y_onehot = to_categorical(Y, len(np.unique(Y)))
    
    # Normalize images to [0, 1] range for better training stability
    X = X.astype('float32') / 255.0
    
    logger.info(f"Original images: {len(image_info)}, Labels: {len(label_info)}")
    logger.info(f"After augmentation: X={len(X)}, Y={len(Y)}")
    
    # Check class balance
    unique, counts = np.unique(Y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    logger.info(f"Class distribution in dataset: {class_distribution}")
    
    return X, Y_onehot, Img_ids


def custom_augment(image):
    """
    Custom augmentation function for more aggressive data augmentation.
    
    Args:
        image: Input image
        
    Returns:
        Augmented image
    """
    # Apply random contrast
    if np.random.random() > 0.5:
        contrast_factor = np.random.uniform(0.7, 1.3)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = (image - mean) * contrast_factor + mean
        image = np.clip(image, 0, 1)
    
    # Apply random noise
    if np.random.random() > 0.8:  # Only apply to 20% of images
        noise_level = np.random.uniform(0, 0.05)
        noise = np.random.normal(0, noise_level, image.shape)
        image = image + noise
        image = np.clip(image, 0, 1)
    
    # Apply random saturation (for RGB images)
    if image.shape[-1] == 3 and np.random.random() > 0.5:
        # Convert to HSV
        hsv = tf.image.rgb_to_hsv(image)
        # Adjust saturation
        saturation_factor = np.random.uniform(0.7, 1.3)
        hsv = tf.concat([
            hsv[:, :, 0:1],
            hsv[:, :, 1:2] * saturation_factor,
            hsv[:, :, 2:3]
        ], axis=2)
        # Convert back to RGB
        image = tf.image.hsv_to_rgb(hsv)
    
    return image

def create_data_generators(x_train, y_train, x_val, y_val, batch_size=128, handle_imbalance=True, cache_dataset=False):
    """
    Create data generators with augmentation.
    
    Args:
        x_train, y_train: Training data
        x_val, y_val: Validation data
        batch_size: Batch size
        handle_imbalance: Whether to apply class balancing
        cache_dataset: Whether to cache dataset in memory
        
    Returns:
        train_generator, val_generator, class_weights
    """
    # Calculate class weights if handling imbalance
    if handle_imbalance:
        y_integers = np.argmax(y_train, axis=1)
        class_counts = np.bincount(y_integers)
        
        # Adjust class weights based on counts
        n_samples = len(y_integers)
        n_classes = len(class_counts)
        
        # More sophisticated weight calculation
        # Higher exponent emphasizes minority classes more
        weight_multiplier = 0.75  # Adjust if needed
        
        class_weights = n_samples / (n_classes * class_counts)
        class_weights = class_weights ** weight_multiplier  # Apply power transform
        
        # Scale weights to have mean of 1 to avoid loss scaling issues
        class_weights = class_weights / np.mean(class_weights)
        
        class_weights_dict = {i: float(class_weights[i]) for i in range(len(class_weights))}
        logger.info(f"Class weights: {class_weights_dict}")
    else:
        class_weights_dict = None
    
    # Create an image data generator with augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=30,                # Increased rotation range
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,                   # Increased zoom range
        horizontal_flip=True,
        vertical_flip=True,               # Enable vertical flipping
        brightness_range=[0.7, 1.3],      # Increased brightness variation
        fill_mode='reflect',              # Changed from nearest to reflect
        preprocessing_function=custom_augment  # Add custom augmentation
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Create standard Keras generators
    train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        x_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create TensorFlow datasets for better performance if requested
    if cache_dataset:
        # Create wrapper functions to ensure correct types
        def train_gen():
            for x_batch, y_batch in train_generator:
                # Ensure both are float32 tensors
                yield (tf.convert_to_tensor(x_batch, dtype=tf.float32), 
                       tf.convert_to_tensor(y_batch, dtype=tf.float32))
        
        def val_gen():
            for x_batch, y_batch in val_generator:
                # Ensure both are float32 tensors
                yield (tf.convert_to_tensor(x_batch, dtype=tf.float32), 
                       tf.convert_to_tensor(y_batch, dtype=tf.float32))
        
        # Create datasets with explicit output signatures
        train_dataset = tf.data.Dataset.from_generator(
            train_gen,
            output_signature=(
                tf.TensorSpec(shape=(None,) + tuple(x_train.shape[1:]), dtype=tf.float32),
                tf.TensorSpec(shape=(None,) + tuple(y_train.shape[1:]), dtype=tf.float32)
            )
        )
        
        val_dataset = tf.data.Dataset.from_generator(
            val_gen,
            output_signature=(
                tf.TensorSpec(shape=(None,) + tuple(x_val.shape[1:]), dtype=tf.float32),
                tf.TensorSpec(shape=(None,) + tuple(y_val.shape[1:]), dtype=tf.float32)
            )
        )
        
        # Add optimization steps
        train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, class_weights_dict
    
    # Return standard Keras generators if not using TF Datasets
    return train_generator, val_generator, class_weights_dict

class BalancedBatchGenerator:
    """
    Custom batch generator that ensures balanced classes in each batch.
    
    This is particularly useful for highly imbalanced datasets.
    """
    
    def __init__(self, x, y, batch_size=32, augmentation=None):
        """
        Initialize the generator.
        
        Args:
            x: Input images
            y: One-hot encoded labels
            batch_size: Size of batches to generate
            augmentation: Optional augmentation function to apply
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augmentation = augmentation
        
        # Get class indices
        self.n_classes = y.shape[1]
        self.class_indices = [np.where(np.argmax(y, axis=1) == i)[0] for i in range(self.n_classes)]
        self.class_counts = [len(indices) for indices in self.class_indices]
        self.min_class_count = min(self.class_counts)
        
        # Calculate samples per class for balanced batches
        self.samples_per_class = self.batch_size // self.n_classes
        
        # Add residual if batch size is not perfectly divisible
        self.residual = self.batch_size % self.n_classes
        
        # Pre-allocate memory for batch data
        self.batch_x = np.zeros((self.batch_size,) + tuple(x.shape[1:]), dtype=x.dtype)
        self.batch_y = np.zeros((self.batch_size,) + tuple(y.shape[1:]), dtype=y.dtype)
        
        logger.info(f"Created balanced batch generator with class counts: {self.class_counts}")
    
    def __iter__(self):
        """Initialize indices for iteration."""
        # Shuffle indices for each class
        self.current_indices = [np.copy(indices) for indices in self.class_indices]
        for indices in self.current_indices:
            np.random.shuffle(indices)
        
        # Initialize position in each class
        self.pos = [0] * self.n_classes
        
        # Calculate number of batches
        samples_per_epoch = min(self.min_class_count * self.n_classes, self.batch_size * 100)
        self.steps_per_epoch = samples_per_epoch // self.batch_size
        
        return self
    
    def __next__(self):
        """Return the next batch."""
        # Check if we've reached the end of the epoch
        if any(self.pos[i] + self.samples_per_class > len(self.current_indices[i]) for i in range(self.n_classes)):
            # Reset for next epoch
            for i in range(self.n_classes):
                np.random.shuffle(self.current_indices[i])
                self.pos[i] = 0
            
            # Signal end of epoch
            raise StopIteration
        
        # Create batch with balanced classes
        batch_idx = 0
        
        # Add samples_per_class from each class
        for class_idx in range(self.n_classes):
            n_samples = self.samples_per_class
            
            # Add residual samples to some classes
            if class_idx < self.residual:
                n_samples += 1
            
            # Get indices for this class
            start_pos = self.pos[class_idx]
            end_pos = start_pos + n_samples
            
            # Check if we have enough samples left
            if end_pos > len(self.current_indices[class_idx]):
                # Reshuffle this class
                np.random.shuffle(self.current_indices[class_idx])
                self.pos[class_idx] = 0
                start_pos = 0
                end_pos = n_samples
            
            # Get indices
            indices = self.current_indices[class_idx][start_pos:end_pos]
            
            # Update position
            self.pos[class_idx] = end_pos
            
            # Add samples to batch
            for i, idx in enumerate(indices):
                self.batch_x[batch_idx] = self.x[idx]
                self.batch_y[batch_idx] = self.y[idx]
                batch_idx += 1
        
        # Apply augmentation if provided
        if self.augmentation is not None:
            # Apply to each image individually
            for i in range(self.batch_size):
                self.batch_x[i] = self.augmentation(self.batch_x[i])
        
        return self.batch_x.copy(), self.batch_y.copy()
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return self.steps_per_epoch


def create_balanced_generator(x, y, batch_size=32, augmentation=None):
    """
    Create a balanced data generator for highly imbalanced datasets.
    
    Args:
        x: Input images
        y: One-hot encoded labels
        batch_size: Batch size
        augmentation: Optional augmentation function
        
    Returns:
        BalancedBatchGenerator instance
    """
    return BalancedBatchGenerator(x, y, batch_size, augmentation)


def create_tf_dataset(x, y, batch_size=32, is_training=True, cache=True, prefetch=True, shuffle_buffer=1000):
    """
    Create an optimized TensorFlow dataset for training.
    
    Args:
        x: Input data
        y: Labels
        batch_size: Batch size
        is_training: Whether this is for training (enables caching, shuffling, prefetching)
        cache: Whether to cache the dataset
        prefetch: Whether to prefetch data
        shuffle_buffer: Size of shuffle buffer
        
    Returns:
        TensorFlow Dataset
    """
    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # Cache if requested - speeds up training after first epoch
    if cache and is_training:
        dataset = dataset.cache()
    
    # Shuffle if training
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(shuffle_buffer, len(x)))
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    # Add augmentation if training
    if is_training:
        # Define augmentations using tf.image
        def augment(images, labels):
            # Random flip left/right
            images = tf.image.random_flip_left_right(images)
            # Random flip up/down
            images = tf.image.random_flip_up_down(images)
            # Random brightness
            images = tf.image.random_brightness(images, max_delta=0.2)
            # Random contrast
            images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
            
            # Ensure values stay in [0, 1] range
            images = tf.clip_by_value(images, 0.0, 1.0)
            return images, labels
        
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetch if requested - allows the GPU to process one batch while CPU prepares the next
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
