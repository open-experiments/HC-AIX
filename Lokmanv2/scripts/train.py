#!/usr/bin/env python3
"""
Unified Training Script - CT Breast Cancer Detection
Single source of truth for training with multiple modes and advanced features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import argparse
import yaml
from tqdm import tqdm
from datetime import datetime
import json
import numpy as np

from core.dataset import CTVolumeDataset
from torch.utils.data import DataLoader


class SimpleAttentionUNet3D(nn.Module):
    """Unified 3D U-Net with configurable output modes"""
    
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


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()


class UnifiedTrainer:
    """Unified trainer supporting both segmentation and classification"""
    
    def __init__(self, model, device, config, mode="auto"):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.mode = mode  # "auto", "classification", "segmentation"
        
        # Setup loss functions
        self.setup_loss_functions()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Setup scheduler
        self.setup_scheduler()
        
        # Mixed precision
        self.use_amp = config['training']['use_mixed_precision'] and device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
    def setup_loss_functions(self):
        """Setup loss functions based on training mode"""
        # Balanced class weights
        class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(self.device)
        
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
    def setup_optimizer(self):
        """Setup optimizer based on config"""
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if self.config['training']['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        
        if scheduler_type == 'CosineAnnealingWarmRestarts':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        elif scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        else:
            self.scheduler = None
    
    def detect_training_mode(self, batch):
        """Auto-detect training mode based on batch content and model architecture"""
        if self.mode != "auto":
            return self.mode
        
        # If model has classifier, prefer classification mode when labels available
        if hasattr(self.model, 'classifier') and 'label' in batch and batch['label'] is not None:
            return "classification"
        # If model has final layer, prefer segmentation mode
        elif hasattr(self.model, 'final'):
            return "segmentation"
        # Fallback based on data availability
        elif 'label' in batch and batch['label'] is not None:
            return "classification"
        else:
            return "segmentation"
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with adaptive mode detection"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
                
            volume = batch['volume'].to(self.device)
            training_mode = self.detect_training_mode(batch)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(volume)
                    loss = self.compute_loss(outputs, batch, training_mode)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss detected!")
                        continue
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               self.config['training'].get('gradient_clip_val', 1.0))
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(volume)
                loss = self.compute_loss(outputs, batch, training_mode)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected!")
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               self.config['training'].get('gradient_clip_val', 1.0))
                
                self.optimizer.step()
            
            # Update learning rate
            if self.scheduler and hasattr(self.scheduler, 'step') and isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch + batch_idx / len(train_loader))
            
            # Calculate accuracy
            accuracy = self.compute_accuracy(outputs, batch, training_mode)
            if accuracy is not None:
                total_correct += accuracy['correct']
                total_samples += accuracy['total']
                current_acc = total_correct / total_samples
            else:
                current_acc = 0.0
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.3f}',
                'lr': f'{current_lr:.6f}',
                'mode': training_mode
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if batch is None:
                    continue
                    
                volume = batch['volume'].to(self.device)
                training_mode = self.detect_training_mode(batch)
                
                outputs = self.model(volume)
                loss = self.compute_loss(outputs, batch, training_mode)
                
                # Calculate accuracy
                accuracy = self.compute_accuracy(outputs, batch, training_mode)
                if accuracy is not None:
                    total_correct += accuracy['correct']
                    total_samples += accuracy['total']
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        # Update plateau scheduler
        if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        
        return avg_loss, avg_acc
    
    def compute_loss(self, outputs, batch, training_mode):
        """Compute loss based on training mode"""
        if training_mode == "classification":
            labels = batch['label'].to(self.device)
            # Combined focal + cross-entropy loss
            focal_loss = self.focal_loss(outputs, labels)
            ce_loss = self.ce_loss(outputs, labels)
            return 0.7 * focal_loss + 0.3 * ce_loss
        else:
            mask = batch['mask'].to(self.device)
            return self.ce_loss(outputs, mask)
    
    def compute_accuracy(self, outputs, batch, training_mode):
        """Compute accuracy based on training mode"""
        if training_mode == "classification":
            labels = batch['label'].to(self.device)
            predicted = outputs.argmax(dim=1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            return {'correct': correct, 'total': total}
        else:
            mask = batch['mask'].to(self.device)
            predicted = outputs.argmax(dim=1)
            correct = (predicted == mask).sum().item()
            total = mask.numel()
            return {'correct': correct, 'total': total}
    
    def save_checkpoint(self, epoch, models_dir, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = models_dir / f"checkpoints/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = models_dir / "best_model.pth"
            torch.save(self.model.state_dict(), best_path)
            print(f"💾 New best model saved with validation accuracy: {self.best_val_acc:.2%}")
        
        # Always save latest as final
        final_path = models_dir / "final_model.pth"
        torch.save(self.model.state_dict(), final_path)


def create_dataloader(data_root, dataset_index, mode, target_size, batch_size, use_augmented=True):
    """Create dataloader with unified collate function"""
    
    # Try to use augmented dataset if available and requested
    if use_augmented and mode == 'train':
        augmented_index = str(Path(dataset_index).parent / "dataset_index_with_splits_augmented.csv")
        if Path(augmented_index).exists():
            dataset_index = augmented_index
            print(f"📊 Using augmented dataset for {mode}: {Path(dataset_index).name}")
    
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
        shuffle=(mode == 'train'),
        num_workers=0,
        collate_fn=safe_collate,
        pin_memory=True
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Unified CT Breast Cancer Training")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--data-root", help="Override data root from config")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--models-dir", help="Override models directory")
    parser.add_argument("--mode", choices=["auto", "classification", "segmentation"], 
                       default="auto", help="Training mode")
    parser.add_argument("--model-mode", choices=["classification", "segmentation"], 
                       default="classification", help="Model architecture mode")
    parser.add_argument("--no-augmented", action="store_true", help="Don't use augmented dataset")
    parser.add_argument("--fast-test", action="store_true", help="Fast test mode with smaller volumes")
    parser.add_argument("--target-size", nargs=3, type=int, help="Override target size (D H W)")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_root:
        config['paths']['data_root'] = args.data_root
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.models_dir:
        config['paths']['model_save_dir'] = args.models_dir
        
    # Fast test mode adjustments
    if args.fast_test:
        print("🚀 Fast test mode enabled!")
        config['data']['target_size'] = [64, 64, 64]
        config['training']['batch_size'] = 1
        config['training']['num_epochs'] = min(2, config['training']['num_epochs'])
        print(f"   📊 Reduced target size to: {config['data']['target_size']}")
        print(f"   📊 Reduced batch size to: {config['training']['batch_size']}")
        print(f"   📊 Reduced epochs to: {config['training']['num_epochs']}")
    
    # Override target size if specified
    if args.target_size:
        config['data']['target_size'] = args.target_size
        print(f"📊 Override target size to: {config['data']['target_size']}")
        
    # Extract commonly used values
    data_root = config['paths']['data_root']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    models_dir = Path(config['paths']['model_save_dir'])
    
    print("🏥 Unified CT Breast Cancer Training")
    print("=" * 50)
    print(f"📋 Training Mode: {args.mode}")
    print(f"🏗️ Model Mode: {args.model_mode}")
    
    # Setup output directories
    models_dir.mkdir(exist_ok=True)
    (models_dir / "checkpoints").mkdir(exist_ok=True)
    (models_dir / "results").mkdir(exist_ok=True)
    (models_dir / "logs").mkdir(exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = models_dir / "logs" / f"training_log_{timestamp}.txt"
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Create model
    print("🔨 Creating unified model...")
    model = SimpleAttentionUNet3D(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        mode=args.model_mode
    )
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    print("📊 Creating data loaders...")
    try:
        if 'processed' not in data_root:
            data_root = f"{data_root}/processed"
        
        dataset_index = f"{data_root}/dataset_index_with_splits.csv"
        target_size = tuple(config['data']['target_size'])
        use_augmented = not args.no_augmented
        
        train_loader = create_dataloader(
            data_root, dataset_index, 'train', target_size, batch_size, use_augmented
        )
        
        val_loader = create_dataloader(
            data_root, dataset_index, 'val', target_size, batch_size, False
        )
        
        test_loader = create_dataloader(
            data_root, dataset_index, 'test', target_size, batch_size, False
        )
        
        print(f"✅ Data loaders created:")
        print(f"   📊 Train: {len(train_loader.dataset)} samples")
        print(f"   📊 Val: {len(val_loader.dataset)} samples") 
        print(f"   📊 Test: {len(test_loader.dataset)} samples")
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create trainer
    trainer = UnifiedTrainer(model, device, config, args.mode)
    
    # Function to log and print
    def log_and_print(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    # Training loop
    log_and_print(f"🚀 Starting unified training for {num_epochs} epochs...")
    log_and_print(f"Device: {device}")
    log_and_print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log_and_print(f"Batch size: {batch_size}")
    log_and_print(f"Learning rate: {learning_rate}")
    log_and_print(f"Target size: {target_size}")
    log_and_print(f"Training mode: {args.mode}")
    log_and_print(f"Model mode: {args.model_mode}")
    
    training_history = []
    
    for epoch in range(num_epochs):
        epoch_msg = f"📅 Epoch {epoch + 1}/{num_epochs}"
        log_and_print(f"\n{epoch_msg}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch + 1)
        train_msg = f"📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}"
        log_and_print(f"   {train_msg}")
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader)
        val_msg = f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}"
        log_and_print(f"   {val_msg}")
        
        # Check if best model
        is_best = val_acc > trainer.best_val_acc
        if is_best:
            trainer.best_val_acc = val_acc
            trainer.best_val_loss = val_loss
        
        # Save checkpoint
        trainer.save_checkpoint(epoch + 1, models_dir, is_best)
        
        # Track history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'is_best': is_best
        })
    
    # Final test
    log_and_print("\n🧪 Running final test...")
    test_loss, test_acc = trainer.validate(test_loader)
    test_msg = f"🎯 Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}"
    log_and_print(f"   {test_msg}")
    
    # Save training results
    results_file = models_dir / "results" / f"training_results_{timestamp}.json"
    results = {
        'config': config,
        'training_args': vars(args),
        'training_history': training_history,
        'best_val_acc': trainer.best_val_acc,
        'best_val_loss': trainer.best_val_loss,
        'final_test_acc': test_acc,
        'final_test_loss': test_loss,
        'total_epochs': len(training_history),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_and_print(f"\n🎉 Training completed!")
    log_and_print(f"📊 Best validation accuracy: {trainer.best_val_acc:.2%}")
    log_and_print(f"🎯 Final test accuracy: {test_acc:.2%}")
    log_and_print(f"💾 Models saved in: {models_dir}")
    log_and_print(f"📊 Results saved to: {results_file}")


if __name__ == "__main__":
    main()