#!/usr/bin/env python3
"""
Memory management utilities for TensorFlow and Keras.
"""

import tensorflow as tf
import gc
import os
import psutil
import logging

logger = logging.getLogger(__name__)

def clear_memory():
    """
    Free memory after finishing with a model.
    Clears the TensorFlow session and runs garbage collection.
    """
    tf.keras.backend.clear_session()
    gc.collect()
    logger.info("TensorFlow session cleared and Python garbage collected")


def log_memory_usage():
    """
    Log current memory usage.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Log memory usage in MB
    logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
    # Try to get GPU memory info if available
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for i, gpu in enumerate(gpus):
                # This is a simplified version and may not work on all systems
                logger.info(f"GPU {i} memory: (see nvidia-smi for details)")
    except:
        logger.debug("Could not retrieve GPU memory information")


def limit_memory_growth():
    """
    Limit TensorFlow memory growth to avoid OOM errors.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
        except RuntimeError as e:
            logger.error(f"Error setting memory growth: {e}")


def optimize_for_inference(model):
    """
    Optimize model for inference by converting to TensorFlow Lite format.
    
    Args:
        model: Trained Keras model
    
    Returns:
        tf.lite.Interpreter: TFLite interpreter ready for inference
    """
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Create a TFLite interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    logger.info("Model optimized for inference")
    return interpreter