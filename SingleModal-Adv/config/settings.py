#!/usr/bin/env python3
"""
Configuration management module for breast cancer detection.

Follows the 12-factor app methodology for configuration:
- Uses environment variables for configuration
- Provides sensible defaults
- Separates config from code
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from .env file if it exists
load_dotenv()

# Base directory - use environment variable or default
BASE_DIR = os.getenv('BASE_DIR', './')

# Parse background tissue types and abnormality classes
BG_TO_PROCESS = os.getenv('DATASET_BG_TYPES', 'G,D,F').split(',')
CLASS_TO_PROCESS = os.getenv('DATASET_CLASSES', 'CIRC,NORM').split(',')

# Parse image size
try:
    image_size_str = os.getenv('IMAGE_SIZE', '331,331')
    IMAGE_SIZE = tuple(map(int, image_size_str.split(',')))
except (ValueError, TypeError):
    logging.warning("Invalid IMAGE_SIZE format, using default (299, 299)")
    IMAGE_SIZE = (299, 299)

# Configuration dictionary
CONFIG = {
    # Dataset configuration
    'bg_to_process': BG_TO_PROCESS,
    'class_to_process': CLASS_TO_PROCESS,
    'info_file_path': os.getenv('INFO_FILE_PATH', os.path.join(BASE_DIR, "data/images/Info.txt")),
    'image_url': os.getenv('IMAGE_URL', os.path.join(BASE_DIR, 'data/images/')),
    'no_angles': int(os.getenv('NO_ANGLES', '360')),
    'angle_interval': int(os.getenv('ANGLE_INTERVAL', '8')),
    'image_size': IMAGE_SIZE,
    
    # Model parameters
    'num_classes': int(os.getenv('NUM_CLASSES', '3')),
    'batch_size': int(os.getenv('BATCH_SIZE', '512')),
    'epochs': int(os.getenv('EPOCHS', '30')),
    'initial_lr': float(os.getenv('INITIAL_LEARNING_RATE', '0.001')),
    'min_lr': float(os.getenv('MIN_LEARNING_RATE', '0.00001')),
    
    # Training configuration
    'multi_gpu': os.getenv('MULTI_GPU', 'true').lower() == 'true',
    'augmentation_factor': float(os.getenv('AUGMENTATION_FACTOR', '2.0')),
    'use_cross_validation': os.getenv('USE_CROSS_VALIDATION', 'true').lower() == 'true',
    'n_folds': int(os.getenv('N_FOLDS', '5')),
    'use_ensemble': os.getenv('USE_ENSEMBLE', 'true').lower() == 'true',
    'train_test_split_ratio': float(os.getenv('TRAIN_TEST_SPLIT_RATIO', '0.2')),
    'handle_imbalance': os.getenv('HANDLE_IMBALANCE', 'true').lower() == 'true',
    'random_state': int(os.getenv('RANDOM_STATE', '42')),
    
    # Output paths
    'checkpoint_filepath': os.getenv('CHECKPOINT_FILEPATH', 'data/best_weights.weights.h5'),
    'logs_dir': os.getenv('LOGS_DIR', 'logs/training'),
    'results_dir': os.getenv('RESULTS_DIR', 'results'),
    
    # Class names
    'class_names': ["Benign", "Malignant", "Normal"]
}

# Create necessary directories
os.makedirs(os.path.dirname(CONFIG['checkpoint_filepath']), exist_ok=True)
os.makedirs(CONFIG['logs_dir'], exist_ok=True)
os.makedirs(CONFIG['results_dir'], exist_ok=True)

# Set TensorFlow environment variables for better performance
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = os.getenv('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.getenv('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['TF_XLA_FLAGS'] = os.getenv('TF_XLA_FLAGS', '--tf_xla_enable_xla_devices')

def get_config():
    """Return the configuration dictionary"""
    return CONFIG