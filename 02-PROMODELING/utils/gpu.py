#!/usr/bin/env python3
"""
GPU utilities for configuring TensorFlow with optimal settings.
"""

import tensorflow as tf
import logging
import os

logger = logging.getLogger(__name__)

def configure_mixed_precision():
    """Configure mixed precision for better performance on GPUs."""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    return policy

def setup_gpus(multi_gpu=True):
    """
    Configure GPUs for TensorFlow with optimal settings.
    
    Args:
        multi_gpu (bool): Whether to use multiple GPUs if available
    
    Returns:
        tf.distribute.Strategy: TensorFlow distribution strategy
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Configure GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logger.info(f"Found {len(gpus)} GPUs")
            
            # Disable GPU memory logging to reduce output
            tf.debugging.set_log_device_placement(False)
            
            # Set additional environment variables for better performance
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['TF_GPU_THREAD_COUNT'] = str(len(gpus) * 2)
            os.environ['TF_USE_CUDNN_AUTOTUNE'] = '1'
            
            # Create a MirroredStrategy if multiple GPUs are available
            if len(gpus) > 1 and multi_gpu:
                # Use NCCL for better multi-GPU communication
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.NcclAllReduce()
                )
                logger.info(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
                return strategy
            else:
                logger.info("Using default strategy with single GPU")
                return tf.distribute.get_strategy()
        except RuntimeError as e:
            logger.error(f"Error in GPU config: {e}")
            return tf.distribute.get_strategy()
    else:
        logger.info("No GPUs detected, using CPU")
        return tf.distribute.get_strategy()

def log_gpu_info():
    """
    Log detailed information about available GPUs
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        logger.info("No GPUs detected")
        return
    
    # Log basic GPU count
    logger.info(f"Found {len(gpus)} GPUs")
    
    # Detailed GPU info
    for i, gpu in enumerate(gpus):
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            logger.info(f"GPU {i}: {gpu_details}")
            
            # Additional memory info
            mem_info = tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=4096
            )
            logger.info(f"GPU {i} memory configuration: {mem_info}")
        except:
            # Fallback if detailed info isn't available
            logger.info(f"GPU {i}: Basic information only")

def optimize_gpu_performance():
    """
    Apply additional optimizations for GPU performance.
    """
    # Enable JIT compilation
    tf.config.optimizer.set_jit(True)
    
    # Enable tensor fusion
    tf.config.optimizer.set_experimental_options(
        {"auto_mixed_precision": True, "device_compilation": "xla"}
    )
    
    # Configure GPU operations
    tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=0.95,  # Use 95% of GPU memory
        allow_growth=True,
        polling_inactive_delay_msecs=50,
        experimental=None
    )
    
    logger.info("Applied GPU performance optimizations")

def setup_rtx_4090(batch_size=128):
    """
    Configure specifically for RTX 4090 GPUs, which have 24GB of VRAM each.
    
    Args:
        batch_size: Batch size to use, adjusted for RTX 4090
        
    Returns:
        Recommended batch size and strategy
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    # Check if GPUs are RTX 4090
    rtx_4090_found = False
    for gpu in gpus:
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if 'GeForce RTX 4090' in str(gpu_details):
                rtx_4090_found = True
        except:
            pass
    
    if not rtx_4090_found:
        logger.info("No RTX 4090 GPUs detected")
        return batch_size, setup_gpus()
    
    logger.info("RTX 4090 GPUs detected, applying optimal settings")
    
    # For RTX 4090, we can use a larger batch size
    recommended_batch_size = min(batch_size, 256)  # Up to 256 is typically efficient
    
    # Apply RTX specific optimizations
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    
    # Configure GPU options for RTX 4090
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=22*1024)]  # 22GB VRAM allocation
    )
    
    # Get distribution strategy
    strategy = setup_gpus()
    
    return recommended_batch_size, strategy