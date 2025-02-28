#!/usr/bin/env python3
"""
Image augmentation utilities for the mammography dataset.
"""

import cv2
import numpy as np
import logging
from preprocessing.data_loader import get_roi_coords, enhance_image, crop_roi

logger = logging.getLogger(__name__)

def read_rotate_flip_images(mmi, url, no_angles, angle_interval, image_size):
    """
    Read, rotate, and flip images for data augmentation.
    
    Args:
        mmi: Dictionary with image information
        url: Base directory for images
        no_angles: Maximum rotation angle
        angle_interval: Rotation step size
        image_size: Target image size (width, height)
        
    Returns:
        Dictionary with image reference and angle as keys and processed images as values
    """
    logger.info("Reading, rotating, and flipping images...")
    info = {}
    failed_images = 0
    
    for key, value in mmi.items():
        image_name = key
        image_address = url + image_name + '.pgm'
        
        # Get ROI coordinates
        class_label, severity, x_center, y_center, radius = get_roi_coords(mmi, image_name)
        
        # Read the image
        img = cv2.imread(image_address, 1)
        if img is None:
            logger.warning(f"Could not read image {image_address}")
            failed_images += 1
            continue
        
        # Apply CLAHE for contrast enhancement
        img = enhance_image(img)
        
        if class_label == "CIRC" and x_center is not None and y_center is not None and radius is not None:
            # Crop the region of interest (ROI) with margin
            roi = crop_roi(img, x_center, y_center, radius)
            # Resize based on ROI
            img = cv2.resize(roi, image_size)
        else:
            # Just resize
            img = cv2.resize(img, image_size)
        
        rows, cols, channel = img.shape
        info[image_name] = {}
        
        # Rotation + Flip + Multiple Augmentations
        for angle in range(0, no_angles, angle_interval):
            # Regular rotation
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_rotated = cv2.warpAffine(img, M, (cols, rows))
            
            # Horizontal flip
            img_flipped = cv2.flip(img_rotated, 1)
            
            # Save both augmentations
            info[image_name][angle] = img_flipped
            
            # For important classes (Benign, Malignant), add extra augmentations
            if class_label == "CIRC":
                # Additional angles for minority classes
                extra_angle = angle + angle_interval // 2
                if extra_angle < no_angles:
                    M_extra = cv2.getRotationMatrix2D((cols / 2, rows / 2), extra_angle, 1)
                    img_rotated_extra = cv2.warpAffine(img, M_extra, (cols, rows))
                    info[image_name][extra_angle] = img_rotated_extra
    
    logger.info(f"Number of processed images: {len(info)}")
    if failed_images > 0:
        logger.warning(f"Failed to read {failed_images} images")
    
    return info


def apply_advanced_augmentation(img, intensity=1.0):
    """
    Apply advanced augmentation techniques to a single image.
    
    Args:
        img: Input image
        intensity: Intensity factor for augmentations (0.0-1.0)
        
    Returns:
        Augmented image
    """
    # Make a copy to avoid modifying the original
    result = img.copy()
    
    # Random brightness adjustment
    brightness_factor = np.random.uniform(0.8, 1.2) * intensity
    if brightness_factor != 1.0:
        if brightness_factor > 1.0:
            result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)
        else:
            result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)
    
    # Random contrast adjustment
    contrast_factor = np.random.uniform(0.8, 1.2) * intensity
    if contrast_factor != 1.0:
        mean = np.mean(result)
        result = cv2.convertScaleAbs(result, alpha=contrast_factor, beta=(1-contrast_factor)*mean)
    
    # Random Gaussian noise
    if np.random.random() < 0.3 * intensity:
        row, col, ch = result.shape
        mean = 0
        sigma = np.random.uniform(1, 5) * intensity
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        result = cv2.add(result, gauss.astype(np.uint8))
    
    # Random rotation (small angles)
    if np.random.random() < 0.5 * intensity:
        angle = np.random.uniform(-10, 10) * intensity
        h, w = result.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        result = cv2.warpAffine(result, M, (w, h))
    
    return result


def generate_augmented_batch(images, batch_size, augmentation_intensity=1.0):
    """
    Generate a batch of augmented images.
    
    Args:
        images: List of original images
        batch_size: Desired batch size
        augmentation_intensity: Intensity of augmentation (0.0-1.0)
        
    Returns:
        Batch of augmented images
    """
    # Select random indices (with replacement if needed)
    if len(images) < batch_size:
        indices = np.random.choice(len(images), size=batch_size, replace=True)
    else:
        indices = np.random.choice(len(images), size=batch_size, replace=False)
    
    # Apply augmentation to each selected image
    augmented_batch = [apply_advanced_augmentation(images[i], augmentation_intensity) for i in indices]
    
    # Convert to numpy array
    return np.array(augmented_batch)