#!/usr/bin/env python3
"""
Main entry point for breast cancer detection using InceptionV3.

Authors: Fatih E. NAR Red Hat, Sedat Kaymaz Microsoft
Optimized Breast Cancer Detection using InceptionV3

This script processes mammography images to classify them as Benign, Malignant, or Normal
using a transfer learning approach with InceptionV3, optimized for multi-GPU performance.

"""

import os
import time
import argparse
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

# Import modules - following modularization principle
from config.settings import get_config
from utils.gpu import setup_gpus, log_gpu_info, configure_mixed_precision
from utils.memory import clear_memory, log_memory_usage
from preprocessing.data_loader import read_info_txt, read_labels
from preprocessing.augmentation import read_rotate_flip_images
from preprocessing.generators import prepare_dataset
from models.architecture import build_advanced_model, build_lightweight_model
from training.cross_validation import train_with_cross_validation
from training.train import train_with_data_splitting, export_model_for_serving
from models.ensemble import ensemble_predict
from evaluation.metrics import evaluate_model, calculate_clinical_metrics
from evaluation.visualization import (
    plot_roc_curves, plot_confusion_matrix, plot_training_history,
    display_sample_images
)

# Setup logging
def setup_logging(log_level=logging.INFO):
    """Configure logging for the application."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    # Reduce TensorFlow logging verbosity
    tf.get_logger().setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Breast Cancer Detection with InceptionV3')
    
    # Training configuration
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'],
                        help='Mode to run the model in')
    parser.add_argument('--cross-validation', action='store_true',
                        help='Use cross-validation for training')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble of models for prediction')
    parser.add_argument('--lightweight', action='store_true',
                        help='Use lightweight model (MobileNetV2) instead of InceptionV3')
    
    # Model parameters
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Initial learning rate')
    
    # Paths
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory with dataset')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model weights file for evaluation or prediction')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory to save results')
    
    # Performance options
    parser.add_argument('--xla', action='store_true', default=True,
                        help='Enable XLA compilation for better performance')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Enable Automatic Mixed Precision training')
    parser.add_argument('--cache-dataset', action='store_true', default=True,
                        help='Cache dataset in memory for faster training')
    
    # Other options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample images and results')
    parser.add_argument('--export-model', action='store_true',
                        help='Export model for serving after training')
    parser.add_argument('--profile', action='store_true',
                        help='Enable TensorBoard profiling')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("\n===== Breast Cancer Detection with InceptionV3 =====")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    config = get_config()
    
    # Override config with command-line arguments
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.learning_rate:
        config['initial_lr'] = args.learning_rate
    if args.data_dir:
        config['image_url'] = os.path.join(args.data_dir, '')
        config['info_file_path'] = os.path.join(args.data_dir, 'Info.txt')
    if args.results_dir:
        config['results_dir'] = args.results_dir
    
    # Create necessary directories
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Enable XLA compilation if requested
    if args.xla:
        tf.config.optimizer.set_jit(True)
        logger.info("XLA compilation enabled for better GPU performance")
    
    # Enable mixed precision if requested
    if args.amp:
        policy = configure_mixed_precision()
        logger.info(f"Automatic Mixed Precision enabled with policy: {policy.name}")
    
    # Setup GPU and distribution strategy
    logger.info("Setting up GPU environment...")
    strategy = setup_gpus(config['multi_gpu'])
    log_gpu_info()
    
    # Set memory growth for better GPU memory management
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            logger.warning(f"Could not set memory growth for GPU {gpu}")
    
    # Start timer
    start_time = time.time()
    
    # Log key configuration parameters
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Image size: {config['image_size']}")
    logger.info(f"Number of epochs: {config['epochs']}")
    logger.info(f"Initial learning rate: {config['initial_lr']}")
    logger.info(f"Using cross-validation: {config['use_cross_validation']}")
    
    if args.mode == 'train':
        train_model(config, strategy, args)
    elif args.mode == 'evaluate':
        evaluate_saved_model(config, strategy, args)
    elif args.mode == 'predict':
        predict_with_model(config, strategy, args)
    
    # Log total execution time
    execution_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {execution_time/60:.2f} minutes")
    
    # Clean up
    clear_memory()
    logger.info("Process completed successfully.")


def train_model(config, strategy, args):
    """Train the breast cancer detection model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # Read dataset information
    logger.info("Loading dataset information...")
    info, mmi = read_info_txt(
        config['info_file_path'],
        config['bg_to_process'],
        config['class_to_process']
    )
    
    # Get labels
    label_info = read_labels(
        mmi,
        config['no_angles'],
        config['angle_interval']
    )
    
    # Read and process images
    logger.info("Reading and processing images...")
    image_info = read_rotate_flip_images(
        mmi,
        config['image_url'],
        config['no_angles'],
        config['angle_interval'],
        config['image_size']
    )
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    X, Y, Img_ids = prepare_dataset(
        image_info,
        label_info,
        config['no_angles'],
        config['angle_interval'],
        test_size=config['train_test_split_ratio'],
        random_state=config['random_state']
    )
    
    # Display sample images if requested
    if args.visualize:
        display_sample_images(
            X, Y, n_samples=3, 
            class_names=config['class_names'],
            save_dir=os.path.join(config['results_dir'], 'samples')
        )
    
    # Log class distribution and dataset statistics
    class_counts = np.bincount(np.argmax(Y, axis=1))
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Class imbalance ratio (majority:minority): {np.max(class_counts) / np.min(class_counts):.2f}:1")
    
    # Select model architecture
    if args.lightweight:
        logger.info("Using lightweight MobileNetV2 architecture...")
        model_builder = build_lightweight_model
    else:
        logger.info("Using InceptionV3 architecture...")
        model_builder = build_advanced_model
    
    # Enable profiling if requested
    if args.profile:
        config['enable_profiling'] = True
        config['profile_batch'] = '500,520'
    
    # Train with cross-validation if requested
    if args.cross_validation or config['use_cross_validation']:
        logger.info("Training with cross-validation...")
        model_files, fold_results, fold_histories = train_with_cross_validation(
            X, Y, Img_ids, strategy, model_builder, config
        )
        
        # If using ensemble prediction, create a final ensemble prediction
        if (args.ensemble or config['use_ensemble']) and len(model_files) > 1:
            create_ensemble_prediction(X, Y, Img_ids, model_files, strategy, config)
    else:
        # Single model training
        logger.info("Training single model...")
        
        # Build model
        model = model_builder(
            strategy,
            input_shape=(*config['image_size'], 3),
            num_classes=config['num_classes'],
            initial_lr=config['initial_lr']
        )
        
        # Print model summary
        model.summary()
        
        # Train model with data splitting
        history, test_data = train_with_data_splitting(
            model,
            X, Y, Img_ids,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            handle_imbalance=config['handle_imbalance'],
            checkpoint_filepath=config['checkpoint_filepath'],
            logs_dir=config['logs_dir'],
            random_state=config['random_state'],
            cache_dataset=args.cache_dataset,
            enable_profiling=args.profile
        )
        
        # Plot training history
        history_plot_path = os.path.join(config['results_dir'], "history.png")
        plot_training_history(history, save_path=history_plot_path)
        
        # Load best weights
        model.load_weights(config['checkpoint_filepath'])
        
        # Evaluate model
        x_test, y_test = test_data
        results = evaluate_model(
            model, 
            x_test, 
            y_test, 
            batch_size=config['batch_size'],
            class_names=config['class_names']
        )
        
        # Plot ROC curves
        roc_plot_path = os.path.join(config['results_dir'], "roc.png")
        plot_roc_curves(
            y_test,
            results['y_pred_proba'],
            num_classes=config['num_classes'],
            class_names=config['class_names'],
            save_path=roc_plot_path
        )
        
        # Plot confusion matrix
        cm_plot_path = os.path.join(config['results_dir'], "cm.png")
        plot_confusion_matrix(
            results['confusion_matrix'],
            class_names=config['class_names'],
            save_path=cm_plot_path
        )
        
        # Calculate clinical metrics
        clinical_metrics = calculate_clinical_metrics(
            results['y_test'],
            results['y_pred'],
            class_names=config['class_names']
        )
        
        # Log clinical metrics
        logger.info("\nClinical Metrics:")
        for class_name, metrics in clinical_metrics.items():
            logger.info(f"{class_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        
        # Export model for serving
        if args.export_model:
            export_path = export_model_for_serving(
                model, os.path.join(config['results_dir'], 'serving')
            )
            logger.info(f"Model exported for serving to: {export_path}")
    
    logger.info("Model training completed")


def evaluate_saved_model(config, strategy, args):
    """Evaluate a saved model."""
    logger = logging.getLogger(__name__)
    
    if not args.model_path:
        logger.error("No model path provided for evaluation. Use --model-path")
        return
    
    logger.info(f"Evaluating saved model: {args.model_path}")
    
    # Read dataset information
    info, mmi = read_info_txt(
        config['info_file_path'],
        config['bg_to_process'],
        config['class_to_process']
    )
    
    # Get labels
    label_info = read_labels(
        mmi,
        config['no_angles'],
        config['angle_interval']
    )
    
    # Read and process images
    image_info = read_rotate_flip_images(
        mmi,
        config['image_url'],
        config['no_angles'],
        config['angle_interval'],
        config['image_size']
    )
    
    # Prepare dataset
    X, Y, Img_ids = prepare_dataset(
        image_info,
        label_info,
        config['no_angles'],
        config['angle_interval'],
        test_size=config['train_test_split_ratio'],
        random_state=config['random_state']
    )
    
    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    
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
    _, test_id_idx = train_test_split(
        np.arange(len(unique_ids)),
        test_size=config['train_test_split_ratio'],
        random_state=config['random_state'],
        stratify=id_Y
    )
    
    test_ids = set(unique_ids[test_id_idx])
    
    # Create test mask
    test_mask = np.array([img_id in test_ids for img_id in Img_ids])
    
    # Get test data
    x_test = X[test_mask]
    y_test = Y[test_mask]
    
    logger.info(f"Test set size: {len(x_test)}")
    
    # Select model architecture
    if args.lightweight:
        logger.info("Using lightweight MobileNetV2 architecture...")
        model_builder = build_lightweight_model
    else:
        logger.info("Using InceptionV3 architecture...")
        model_builder = build_advanced_model
    
    # Build model
    model = model_builder(
        strategy,
        input_shape=(*config['image_size'], 3),
        num_classes=config['num_classes'],
        initial_lr=config['initial_lr']
    )
    
    # Load weights
    model.load_weights(args.model_path)
    
    # Evaluate model
    results = evaluate_model(
        model, 
        x_test, 
        y_test, 
        batch_size=config['batch_size'],
        class_names=config['class_names']
    )
    
    # Plot ROC curves
    roc_plot_path = os.path.join(config['results_dir'], "evaluation_roc.png")
    plot_roc_curves(
        y_test,
        results['y_pred_proba'],
        num_classes=config['num_classes'],
        class_names=config['class_names'],
        save_path=roc_plot_path
    )
    
    # Plot confusion matrix
    cm_plot_path = os.path.join(config['results_dir'], "evaluation_cm.png")
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names=config['class_names'],
        save_path=cm_plot_path
    )
    
    # Calculate clinical metrics
    clinical_metrics = calculate_clinical_metrics(
        results['y_test'],
        results['y_pred'],
        class_names=config['class_names']
    )
    
    # Log clinical metrics
    logger.info("\nClinical Metrics:")
    for class_name, metrics in clinical_metrics.items():
        logger.info(f"{class_name}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")


def predict_with_model(config, strategy, args):
    """Make predictions with a saved model."""
    logger = logging.getLogger(__name__)
    
    if not args.model_path:
        logger.error("No model path provided for prediction. Use --model-path")
        return
    
    if not args.data_dir:
        logger.error("No data directory provided for prediction. Use --data-dir")
        return
    
    logger.info(f"Making predictions with model: {args.model_path}")
    
    # Check if model_path is a directory (could be multiple model files for ensemble)
    if os.path.isdir(args.model_path):
        model_files = [os.path.join(args.model_path, f) for f in os.listdir(args.model_path) 
                      if f.endswith('.h5') or f.endswith('.weights.h5')]
        logger.info(f"Found {len(model_files)} model files for ensemble prediction")
    else:
        model_files = [args.model_path]
    
    # Read dataset information
    info, mmi = read_info_txt(
        config['info_file_path'],
        config['bg_to_process'],
        config['class_to_process']
    )
    
    # Get labels
    label_info = read_labels(
        mmi,
        config['no_angles'],
        config['angle_interval']
    )
    
    # Read and process images
    image_info = read_rotate_flip_images(
        mmi,
        config['image_url'],
        config['no_angles'],
        config['angle_interval'],
        config['image_size']
    )
    
    # Prepare dataset
    X, Y, Img_ids = prepare_dataset(
        image_info,
        label_info,
        config['no_angles'],
        config['angle_interval'],
        random_state=config['random_state']
    )
    
    # If using ensemble prediction with multiple models
    if len(model_files) > 1 and (args.ensemble or config['use_ensemble']):
        logger.info("Using ensemble prediction with multiple models")
        predictions = ensemble_predict(
            model_files,
            X,
            strategy,
            input_shape=(*config['image_size'], 3),
            num_classes=config['num_classes'],
            batch_size=config['batch_size']
        )
        
        # Convert predictions to class indices
        y_pred = np.argmax(predictions, axis=1)
    else:
        # Single model prediction
        logger.info("Using single model prediction")
        
        # Select model architecture
        if args.lightweight:
            logger.info("Using lightweight MobileNetV2 architecture...")
            model_builder = build_lightweight_model
        else:
            logger.info("Using InceptionV3 architecture...")
            model_builder = build_advanced_model
        
        # Build model
        model = model_builder(
            strategy,
            input_shape=(*config['image_size'], 3),
            num_classes=config['num_classes'],
            initial_lr=config['initial_lr']
        )
        
        # Load weights
        model.load_weights(model_files[0])
        
        # Make predictions
        predictions = model.predict(X, batch_size=config['batch_size'])
        y_pred = np.argmax(predictions, axis=1)
    
    # Convert true labels to indices if one-hot encoded
    if len(Y.shape) > 1:
        y_true = np.argmax(Y, axis=1)
    else:
        y_true = Y
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)
    logger.info(f"Prediction accuracy: {accuracy:.4f}")
    
    # If we have ground truth labels, evaluate the predictions
    if Y is not None:
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_true, y_pred, target_names=config['class_names']))
        
        # Calculate and print confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"{cm}")
        
        # Plot confusion matrix
        cm_plot_path = os.path.join(config['results_dir'], "prediction_cm.png")
        plot_confusion_matrix(
            cm,
            class_names=config['class_names'],
            save_path=cm_plot_path
        )
    
    # Save predictions to file
    predictions_path = os.path.join(config['results_dir'], "predictions.csv")
    import pandas as pd
    
    # Create dataframe
    predictions_df = pd.DataFrame({
        'image_id': Img_ids,
        'true_label': y_true,
        'predicted_label': y_pred,
        'true_class': [config['class_names'][i] for i in y_true],
        'predicted_class': [config['class_names'][i] for i in y_pred]
    })
    
    # Add prediction probabilities
    for i, class_name in enumerate(config['class_names']):
        predictions_df[f'prob_{class_name}'] = predictions[:, i]
    
    # Save to CSV
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")


def create_ensemble_prediction(X, Y, Img_ids, model_files, strategy, config):
    """Create ensemble prediction from multiple models."""
    logger = logging.getLogger(__name__)
    logger.info("\n===== Creating Ensemble Prediction =====")
    
    # Split data into train and test sets once more for final evaluation
    # Use stratified sampling to get test set
    Y_indices = np.argmax(Y, axis=1)
    unique_ids = np.unique(Img_ids)
    id_classes = {}
    
    for i, img_id in enumerate(Img_ids):
        if img_id not in id_classes:
            id_classes[img_id] = Y_indices[i]
    
    unique_Y = np.array([id_classes[id] for id in unique_ids])
    
    # Split IDs into train and test
    from sklearn.model_selection import train_test_split
    _, test_id_idx = train_test_split(
        np.arange(len(unique_ids)),
        test_size=0.2,
        random_state=config['random_state'],
        stratify=unique_Y
    )
    
    test_ids = set(unique_ids[test_id_idx])
    
    # Create test mask
    test_mask = np.array([img_id in test_ids for img_id in Img_ids])
    
    # Get test data
    x_test_final = X[test_mask]
    y_test_final = Y[test_mask]
    
    logger.info(f"Final ensemble test set size: {len(x_test_final)}")
    
    # Make ensemble prediction
    ensemble_predictions = ensemble_predict(
        model_files, 
        x_test_final, 
        strategy,
        input_shape=(*config['image_size'], 3),
        num_classes=config['num_classes'],
        batch_size=config['batch_size']
    )
    
    # Evaluate ensemble predictions
    y_pred_ensemble = np.argmax(ensemble_predictions, axis=1)
    y_test_indices = np.argmax(y_test_final, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    
    accuracy = accuracy_score(y_test_indices, y_pred_ensemble)
    precision = precision_score(y_test_indices, y_pred_ensemble, average='weighted')
    recall = recall_score(y_test_indices, y_pred_ensemble, average='weighted')
    f1 = f1_score(y_test_indices, y_pred_ensemble, average='weighted')
    
    # Calculate per-class metrics
    per_class_precision = precision_score(y_test_indices, y_pred_ensemble, average=None)
    per_class_recall = recall_score(y_test_indices, y_pred_ensemble, average=None)
    per_class_f1 = f1_score(y_test_indices, y_pred_ensemble, average=None)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test_indices, y_pred_ensemble)
    
    logger.info("\nEnsemble Model Performance:")
    logger.info(f'Accuracy score: {accuracy:.4f}')
    logger.info(f'Precision score: {precision:.4f}')
    logger.info(f'Recall score: {recall:.4f}')
    logger.info(f'F1 Score: {f1:.4f}')
    
    logger.info('\nClassification Report:')
    logger.info(classification_report(y_test_indices, y_pred_ensemble, target_names=config['class_names']))
    
    logger.info('\nConfusion Matrix:')
    logger.info(f"{cm}")
    
    # Print per-class metrics
    logger.info("\nPer-class metrics:")
    for i, class_name in enumerate(config['class_names']):
        logger.info(f"{class_name}:")
        logger.info(f"  Precision: {per_class_precision[i]:.4f}")
        logger.info(f"  Recall: {per_class_recall[i]:.4f}")
        logger.info(f"  F1: {per_class_f1[i]:.4f}")
    
    # Plot ROC curves
    roc_plot_path = os.path.join(config['results_dir'], "ensemble_roc.png")
    plot_roc_curves(
        y_test_final,
        ensemble_predictions,
        num_classes=config['num_classes'],
        class_names=config['class_names'],
        save_path=roc_plot_path
    )
    
    # Plot confusion matrix
    cm_plot_path = os.path.join(config['results_dir'], "ensemble_cm.png")
    plot_confusion_matrix(
        cm,
        class_names=config['class_names'],
        save_path=cm_plot_path
    )
    
    # Calculate clinical metrics
    clinical_metrics = calculate_clinical_metrics(
        y_test_indices,
        y_pred_ensemble,
        class_names=config['class_names']
    )
    
    # Log clinical metrics
    logger.info("\nClinical Metrics:")
    for class_name, metrics in clinical_metrics.items():
        logger.info(f"{class_name}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()