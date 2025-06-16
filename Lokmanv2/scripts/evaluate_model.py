#!/usr/bin/env python3
"""
Unified Model Evaluation Script
Compatible with the unified training script and all model modes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
import json
from datetime import datetime
import yaml

from core.dataset import CTVolumeDataset
from torch.utils.data import DataLoader


class SimpleAttentionUNet3D(nn.Module):
    """Unified 3D U-Net with configurable output modes - SAME AS TRAINING"""
    
    def __init__(self, in_channels=1, out_channels=4, base_channels=32, mode="segmentation"):
        super().__init__()
        self.mode = mode
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*4, base_channels*4, 3, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*2, 3, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose3d(base_channels*2, base_channels, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output heads
        if mode == "classification":
            # Classification head
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            self.classifier = nn.Sequential(
                nn.Linear(base_channels, base_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(base_channels // 2, out_channels)
            )
        else:
            # Segmentation head
            self.final = nn.Conv3d(base_channels, out_channels, 1)
        
    def _match_size(self, x, target):
        """Match spatial dimensions of x to target by padding or cropping"""
        x_shape = x.shape[2:]
        target_shape = target.shape[2:]
        
        pads = []
        for i in range(len(x_shape)):
            diff = target_shape[i] - x_shape[i]
            if diff > 0:
                pad_before = diff // 2
                pad_after = diff - pad_before
                pads.extend([pad_before, pad_after])
            else:
                pads.extend([0, 0])
        
        if any(p > 0 for p in pads):
            pads_reversed = []
            for i in range(len(pads)//2 - 1, -1, -1):
                pads_reversed.extend([pads[i*2], pads[i*2+1]])
            x = F.pad(x, pads_reversed)
        
        slices = [slice(None), slice(None)]
        for i in range(len(x_shape)):
            diff = x.shape[i+2] - target_shape[i]
            if diff > 0:
                start = diff // 2
                end = start + target_shape[i]
                slices.append(slice(start, end))
            else:
                slices.append(slice(None))
        
        return x[tuple(slices)]

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        # Decoder
        dec2 = self.upconv2(enc3)
        dec2 = self._match_size(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self._match_size(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Output based on mode
        if self.mode == "classification":
            features = self.global_pool(dec1)  # [batch, channels, 1, 1, 1]
            features = features.view(features.size(0), -1)  # [batch, channels]
            output = self.classifier(features)
        else:
            output = self.final(dec1)
        
        return output


class ModelEvaluator:
    """Unified model evaluator supporting both classification and segmentation"""
    
    def __init__(self, model, device, model_mode="auto"):
        self.model = model.to(device)
        self.device = device
        self.model_mode = model_mode
        self.predictions = []
        self.targets = []
        self.confidences = []
        
    def detect_evaluation_mode(self, batch):
        """Auto-detect evaluation mode based on batch content and model"""
        if self.model_mode != "auto":
            return self.model_mode
        
        # Check if model has classifier (classification mode)
        if hasattr(self.model, 'classifier'):
            return "classification"
        elif hasattr(self.model, 'final'):
            return "segmentation"
        else:
            # Fallback to batch content
            if 'label' in batch and batch['label'] is not None:
                return "classification"
            else:
                return "segmentation"
        
    def evaluate_dataset(self, test_loader, dataset_name="Test"):
        """Evaluate model on dataset with adaptive mode detection"""
        
        print(f"\n🧪 Evaluating on {dataset_name} set...")
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {dataset_name}")):
                if batch is None:
                    continue
                    
                volume = batch['volume'].to(self.device)
                evaluation_mode = self.detect_evaluation_mode(batch)
                
                # Forward pass
                output = self.model(volume)
                
                # Handle deep supervision output
                if isinstance(output, (tuple, list)):
                    output = output[0]  # Use main output
                
                # Compute loss and metrics based on mode
                if evaluation_mode == "classification":
                    if 'label' in batch:
                        labels = batch['label'].to(self.device)
                        loss = criterion(output, labels)
                        
                        # Get predictions and confidences
                        probabilities = torch.softmax(output, dim=1)
                        predicted = output.argmax(dim=1)
                        confidence = probabilities.max(dim=1)[0]
                        
                        # Store results (volume-level classification)
                        all_predictions.extend(predicted.cpu().numpy())
                        all_targets.extend(labels.cpu().numpy())
                        all_confidences.extend(confidence.cpu().numpy())
                    else:
                        print(f"Warning: Classification mode but no labels in batch")
                        continue
                        
                else:  # segmentation mode
                    if 'mask' in batch:
                        mask = batch['mask'].to(self.device)
                        loss = criterion(output, mask)
                        
                        # Get predictions and confidences (pixel-level)
                        probabilities = torch.softmax(output, dim=1)
                        predicted = output.argmax(dim=1)
                        confidence = probabilities.max(dim=1)[0]
                        
                        # Store results (pixel-level segmentation)
                        all_predictions.extend(predicted.cpu().numpy().flatten())
                        all_targets.extend(mask.cpu().numpy().flatten())
                        all_confidences.extend(confidence.cpu().numpy().flatten())
                    else:
                        print(f"Warning: Segmentation mode but no mask in batch")
                        continue
                
                total_loss += loss.item()
        
        # Store for later analysis
        self.predictions = np.array(all_predictions)
        self.targets = np.array(all_targets)
        self.confidences = np.array(all_confidences)
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(self.targets, self.predictions)
        
        print(f"📊 Evaluation mode: {evaluation_mode}")
        print(f"📊 Total predictions: {len(self.predictions)}")
        
        return avg_loss, accuracy
    
    def calculate_detailed_metrics(self):
        """Calculate comprehensive metrics"""
        
        print("\n📊 Calculating detailed metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(self.targets, self.predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.targets, self.predictions, average=None, zero_division=0
        )
        
        # Macro averages
        precision_macro = precision_recall_fscore_support(
            self.targets, self.predictions, average='macro', zero_division=0
        )[0]
        recall_macro = precision_recall_fscore_support(
            self.targets, self.predictions, average='macro', zero_division=0
        )[1]
        f1_macro = precision_recall_fscore_support(
            self.targets, self.predictions, average='macro', zero_division=0
        )[2]
        
        # Confusion matrix
        cm = confusion_matrix(self.targets, self.predictions)
        
        # Class-specific metrics
        num_classes = len(np.unique(self.targets))
        class_names = [f"Class {i}" for i in range(num_classes)]
        
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'num_samples': len(self.targets),
                'confidence_mean': float(np.mean(self.confidences)),
                'confidence_std': float(np.std(self.confidences))
            },
            'per_class': {},
            'confusion_matrix': cm.tolist()
        }
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                metrics['per_class'][class_name] = {
                    'precision': float(precision[i]) if not np.isnan(precision[i]) else 0.0,
                    'recall': float(recall[i]) if not np.isnan(recall[i]) else 0.0,
                    'f1_score': float(f1[i]) if not np.isnan(f1[i]) else 0.0,
                    'support': int(support[i]) if i < len(support) else 0
                }
        
        return metrics
    
    def plot_confusion_matrix(self, save_path="confusion_matrix.png"):
        """Plot and save confusion matrix"""
        
        cm = confusion_matrix(self.targets, self.predictions)
        num_classes = cm.shape[0]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f'Class {i}' for i in range(num_classes)],
                    yticklabels=[f'Class {i}' for i in range(num_classes)])
        plt.title('Confusion Matrix - CT Breast Cancer Detection')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved: {save_path}")
    
    def plot_confidence_distribution(self, save_path="confidence_distribution.png"):
        """Plot confidence score distribution"""
        
        plt.figure(figsize=(12, 5))
        
        # Overall confidence distribution
        plt.subplot(1, 2, 1)
        plt.hist(self.confidences, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Overall Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        # Confidence by correctness
        plt.subplot(1, 2, 2)
        correct_mask = (self.predictions == self.targets)
        
        plt.hist(self.confidences[correct_mask], bins=30, alpha=0.7, 
                label='Correct Predictions', color='green', edgecolor='black')
        plt.hist(self.confidences[~correct_mask], bins=30, alpha=0.7, 
                label='Incorrect Predictions', color='red', edgecolor='black')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence by Prediction Correctness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confidence distribution saved: {save_path}")


def load_trained_model(model_path, device, config):
    """Load the trained model with automatic architecture detection"""
    
    print(f"📥 Loading model from: {model_path}")
    
    # Load state dict to detect model type
    try:
        state_dict = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"❌ Error loading model file: {e}")
        print(f"💡 Hint: This might be an old model from previous architecture.")
        print(f"💡 Try training a new model with the unified training script.")
        return None, None
    
    # Detect model type based on keys in state dict
    has_classifier = any('classifier' in key for key in state_dict.keys())
    has_final = any('final' in key for key in state_dict.keys())
    
    print(f"🔍 Model type detection:")
    print(f"   Has classifier layers: {has_classifier}")
    print(f"   Has final layer: {has_final}")
    
    # Create appropriate model architecture
    if has_classifier:
        print("📊 Loading classification model...")
        model_mode = "classification"
    elif has_final:
        print("📊 Loading segmentation model...")
        model_mode = "segmentation"
    else:
        print("❌ Unknown model architecture - cannot detect model type")
        return None, None
    
    # Create model with detected mode
    model = SimpleAttentionUNet3D(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        mode=model_mode
    )
    
    # Load trained weights
    try:
        model.load_state_dict(state_dict)
        print(f"✅ Model state dict loaded successfully ({model_mode} architecture)")
    except Exception as e:
        print(f"❌ Error loading model state dict: {e}")
        print(f"Available keys: {list(state_dict.keys())[:5]}...")
        return None, None
    
    model = model.to(device)
    print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, model_mode


def create_dataloader(data_root, dataset_index, mode, target_size, batch_size):
    """Create dataloader for evaluation"""
    
    dataset = CTVolumeDataset(
        data_root=data_root,
        dataset_index=dataset_index,
        mode=mode,
        target_size=target_size,
        use_cache=True
    )
    
    def safe_collate(batch):
        if not batch:
            return None
            
        result = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            
            if key in ['volume', 'mask']:
                result[key] = torch.stack(values)
            elif key in ['label']:
                result[key] = torch.tensor(values)
            elif key in ['series_id', 'has_mask']:
                result[key] = values
            elif key == 'metadata':
                result[key] = values
            else:
                try:
                    from torch.utils.data._utils.collate import default_collate
                    result[key] = default_collate(values)
                except:
                    result[key] = values
        
        return result
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate,
        pin_memory=True
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Unified CT Model Evaluation")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", help="Path to trained model (overrides config)")
    parser.add_argument("--data-root", help="Path to data directory (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--mode", choices=["auto", "classification", "segmentation"], 
                       default="auto", help="Evaluation mode")
    parser.add_argument("--fast-test", action="store_true", help="Fast test mode with smaller volumes")
    parser.add_argument("--target-size", nargs=3, type=int, help="Override target size (D H W)")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_root:
        config['paths']['data_root'] = args.data_root
    if args.output_dir:
        config['paths']['results_dir'] = args.output_dir
        
    # Fast test mode adjustments
    if args.fast_test:
        print("🚀 Fast test mode enabled!")
        config['data']['target_size'] = [64, 64, 64]
        config['training']['batch_size'] = 1
        print(f"   📊 Reduced target size to: {config['data']['target_size']}")
        print(f"   📊 Reduced batch size to: {config['training']['batch_size']}")
    
    # Override target size if specified
    if args.target_size:
        config['data']['target_size'] = args.target_size
        print(f"📊 Override target size to: {config['data']['target_size']}")
        
    # Set defaults from config
    data_root = config['paths']['data_root']
    output_dir = config['paths']['results_dir']
    model_path = args.model if args.model else f"{config['paths']['model_save_dir']}/best_model.pth"
    
    print("🧪 Unified CT Breast Cancer Model Evaluation")
    print("=" * 60)
    print(f"📋 Evaluation Mode: {args.mode}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    try:
        model, model_mode = load_trained_model(model_path, device, config)
        if model is None:
            print("❌ Failed to load model. Exiting.")
            return
            
        print(f"📊 Model mode: {model_mode}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create data loaders
    print("📊 Loading test data...")
    try:
        if 'processed' not in data_root:
            data_root = f"{data_root}/processed"
        dataset_index = f"{data_root}/dataset_index_with_splits.csv"
        target_size = tuple(config['data']['target_size'])
        
        test_loader = create_dataloader(
            data_root, dataset_index, "test", target_size, 
            config['training']['batch_size']
        )
        
        print(f"✅ Test dataset loaded: {len(test_loader.dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, args.mode)
    
    # Evaluate on test set
    test_loss, test_accuracy = evaluator.evaluate_dataset(test_loader, "Test")
    print(f"\n🎯 Test Results:")
    print(f"   📊 Test Loss: {test_loss:.4f}")
    print(f"   🎯 Test Accuracy: {test_accuracy:.2%}")
    
    # Calculate detailed metrics
    metrics = evaluator.calculate_detailed_metrics()
    
    # Print detailed results
    print(f"\n📈 Detailed Metrics:")
    print(f"   🎯 Overall Accuracy: {metrics['overall']['accuracy']:.2%}")
    print(f"   📊 Macro Precision: {metrics['overall']['precision_macro']:.2%}")
    print(f"   📊 Macro Recall: {metrics['overall']['recall_macro']:.2%}")
    print(f"   📊 Macro F1-Score: {metrics['overall']['f1_macro']:.2%}")
    print(f"   🔍 Mean Confidence: {metrics['overall']['confidence_mean']:.3f}")
    print(f"   📏 Confidence Std: {metrics['overall']['confidence_std']:.3f}")
    
    print(f"\n📋 Per-Class Results:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"   {class_name}:")
        print(f"      Precision: {class_metrics['precision']:.2%}")
        print(f"      Recall: {class_metrics['recall']:.2%}")
        print(f"      F1-Score: {class_metrics['f1_score']:.2%}")
        print(f"      Support: {class_metrics['support']}")
    
    # Generate and save plots
    plots_dir = output_dir.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    evaluator.plot_confusion_matrix(plots_dir / "confusion_matrix.png")
    evaluator.plot_confidence_distribution(plots_dir / "confidence_distribution.png")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"evaluation_results_{timestamp}.json"
    
    evaluation_report = {
        'model_path': model_path,
        'model_mode': model_mode,
        'evaluation_mode': args.mode,
        'evaluation_date': timestamp,
        'device': str(device),
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'detailed_metrics': metrics,
        'config': config
    }
    
    with open(results_file, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   📊 JSON Report: {results_file}")
    print(f"   📊 Confusion Matrix: {plots_dir / 'confusion_matrix.png'}")
    print(f"   📊 Confidence Plot: {plots_dir / 'confidence_distribution.png'}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()