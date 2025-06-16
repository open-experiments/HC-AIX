"""
Advanced trainer for CT-based breast cancer metastasis detection.
Implements state-of-the-art training strategies with comprehensive monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
from tqdm import tqdm
import json
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import metrics from utils
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)

# Simple loss functions
class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Convert to probabilities
        probs = torch.softmax(predictions, dim=1)
        
        # Convert targets to one-hot
        num_classes = predictions.shape[1]
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Calculate dice for each class
        dice_scores = []
        for c in range(num_classes):
            pred_c = probs[:, c].flatten()
            target_c = targets_one_hot[:, c].flatten()
            
            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + self.smooth) / (pred_c.sum() + target_c.sum() + self.smooth)
            dice_scores.append(dice)
        
        return 1.0 - torch.stack(dice_scores).mean()

def create_loss_function(config, class_weights=None):
    """Create loss function from config"""
    return DiceLoss()

def calculate_class_weights(dataloader, num_classes=4):
    """Calculate class weights from dataset"""
    return torch.ones(num_classes)

class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 25, min_delta: float = 0.001, 
                 mode: str = "max", restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        self.is_better = self._get_comparison_fn()
    
    def _get_comparison_fn(self):
        if self.mode == "max":
            return lambda current, best: current > best + self.min_delta
        else:
            return lambda current, best: current < best - self.min_delta
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info(f"Restored best weights with score: {self.best_score:.4f}")
            return True
        
        return False

class ModelCheckpoint:
    """Model checkpoint callback"""
    
    def __init__(self, checkpoint_dir: str, monitor: str = "val_dice_score", 
                 mode: str = "max", save_top_k: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        
        self.best_scores = []
        self.is_better = self._get_comparison_fn()
    
    def _get_comparison_fn(self):
        if self.mode == "max":
            return lambda current, best: current > best
        else:
            return lambda current, best: current < best
    
    def __call__(self, score: float, model: nn.Module, optimizer: optim.Optimizer, 
                 epoch: int, metrics: Dict[str, float]):
        
        # Check if this is a top-k score
        should_save = False
        
        if len(self.best_scores) < self.save_top_k:
            should_save = True
        elif self.is_better(score, min(self.best_scores)):
            should_save = True
            # Remove worst checkpoint
            worst_idx = self.best_scores.index(min(self.best_scores))
            worst_score = self.best_scores.pop(worst_idx)
            
            # Find and remove corresponding checkpoint file
            for ckpt_file in self.checkpoint_dir.glob(f"*_score_{worst_score:.4f}_*.pth"):
                ckpt_file.unlink()
        
        if should_save:
            self.best_scores.append(score)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score,
                'metrics': metrics
            }
            
            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:03d}_score_{score:.4f}_checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Also save as latest
            latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
            torch.save(checkpoint, latest_path)
        
        # Always save best model
        if not self.best_scores or self.is_better(score, max(self.best_scores) if self.mode == "max" else min(self.best_scores)):
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score,
                'metrics': metrics
            }, best_path)
            
            logger.info(f"Saved best model with score: {score:.4f}")

class CTTrainer:
    """Advanced trainer for CT breast cancer detection"""
    
    def __init__(self, model: nn.Module, config, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.training.use_mixed_precision else None
        
        # Initialize loss function (will be updated with class weights)
        self.criterion = None
        self.class_weights = None
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(num_classes=config.model.out_channels)
        
        # Initialize callbacks
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode="max",
            restore_best_weights=True
        )
        
        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=config.system.checkpoint_dir,
            monitor=config.training.monitor_metric,
            mode="max",
            save_top_k=config.training.save_top_k
        )
        
        # Initialize logging
        self.writer = SummaryWriter(config.system.tensorboard_dir)
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Initialize wandb if available
        self.use_wandb = False
        try:
            wandb.init(
                project="breast-cancer-ct-detection",
                config=config.to_dict(),
                name=f"attention-unet-3d-{int(time.time())}"
            )
            self.use_wandb = True
            logger.info("Weights & Biases logging initialized")
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        
        logger.info("Trainer initialized successfully")
    
    def _create_optimizer(self):
        """Create optimizer"""
        if self.config.training.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.scheduler.lower() == "cosineannealingwarmrestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
        elif self.config.training.scheduler.lower() == "reducelronplateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=self.config.training.reduce_lr_patience,
                verbose=True
            )
        elif self.config.training.scheduler.lower() == "steplr":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def calculate_and_set_class_weights(self, train_loader):
        """Calculate and set class weights from training data"""
        logger.info("Calculating class weights from training data...")
        
        self.class_weights = calculate_class_weights(train_loader, self.config.model.out_channels)
        self.class_weights = self.class_weights.to(self.device)
        
        # Create loss function with class weights
        self.criterion = create_loss_function(self.config, self.class_weights)
        
        logger.info(f"Class weights set: {self.class_weights}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = {
            'dice_scores': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.training.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            volumes = batch['volume'].to(self.device)
            masks = batch['mask'].to(self.device)
            has_mask = batch['has_mask']
            
            # Skip batches without masks
            if not has_mask.any():
                continue
            
            # Filter to only samples with masks
            mask_indices = has_mask.nonzero(as_tuple=True)[0]
            volumes = volumes[mask_indices]
            masks = masks[mask_indices]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(volumes)
                    loss_dict = self.criterion(outputs, masks)
                    loss = loss_dict['total']
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.config.training.gradient_clipping)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(volumes)
                loss_dict = self.criterion(outputs, masks)
                loss = loss_dict['total']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.config.training.gradient_clipping)
                
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                if isinstance(outputs, (list, tuple)):
                    main_outputs = outputs[0]
                else:
                    main_outputs = outputs
                
                predictions = torch.argmax(main_outputs, dim=1)
                batch_metrics = self.metrics.calculate_batch_metrics(predictions, masks)
                
                # Store metrics
                epoch_losses.append(loss.item())
                for key, value in batch_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{batch_metrics.get('dice_score', 0):.4f}",
                'Acc': f"{batch_metrics.get('accuracy', 0):.4f}"
            })
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                if self.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/dice_score': batch_metrics.get('dice_score', 0),
                        'global_step': self.global_step
                    })
            
            self.global_step += 1
        
        # Calculate epoch averages
        epoch_loss = np.mean(epoch_losses)
        epoch_dice = np.mean(epoch_metrics['dice_scores'])
        epoch_accuracy = np.mean(epoch_metrics['accuracy'])
        
        return {
            'loss': epoch_loss,
            'dice_score': epoch_dice,
            'accuracy': epoch_accuracy
        }
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                volumes = batch['volume'].to(self.device)
                masks = batch['mask'].to(self.device)
                has_mask = batch['has_mask']
                
                # Skip batches without masks
                if not has_mask.any():
                    continue
                
                # Filter to only samples with masks
                mask_indices = has_mask.nonzero(as_tuple=True)[0]
                volumes = volumes[mask_indices]
                masks = masks[mask_indices]
                
                # Forward pass
                outputs = self.model(volumes)
                loss_dict = self.criterion(outputs, masks)
                loss = loss_dict['total']
                
                # Get predictions
                if isinstance(outputs, (list, tuple)):
                    main_outputs = outputs[0]
                else:
                    main_outputs = outputs
                
                predictions = torch.argmax(main_outputs, dim=1)
                
                # Store for metrics calculation
                epoch_losses.append(loss.item())
                all_predictions.append(predictions.cpu())
                all_targets.append(masks.cpu())
        
        # Calculate comprehensive metrics
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            val_metrics = self.metrics.calculate_comprehensive_metrics(all_predictions, all_targets)
            val_loss = np.mean(epoch_losses)
            
            val_metrics['loss'] = val_loss
            
            return val_metrics
        else:
            return {'loss': float('inf'), 'dice_score': 0.0, 'accuracy': 0.0}
    
    def fit(self, train_loader, val_loader):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Calculate class weights
        self.calculate_and_set_class_weights(train_loader)
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['loss'])
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.config.training.monitor_metric])
                else:
                    self.scheduler.step()
            
            # Store current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs}")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Dice: {train_metrics['dice_score']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Dice: {val_metrics['dice_score']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            
            # Tensorboard logging
            self.writer.add_scalars('Loss', {
                'Train': train_metrics['loss'],
                'Val': val_metrics['loss']
            }, epoch)
            
            self.writer.add_scalars('Dice_Score', {
                'Train': train_metrics['dice_score'],
                'Val': val_metrics['dice_score']
            }, epoch)
            
            self.writer.add_scalars('Accuracy', {
                'Train': train_metrics['accuracy'],
                'Val': val_metrics['accuracy']
            }, epoch)
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_dice': train_metrics['dice_score'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/dice_score': val_metrics['dice_score'],
                    'val/accuracy': val_metrics['accuracy'],
                    'learning_rate': current_lr
                })
            
            # Model checkpointing
            monitor_score = val_metrics[self.config.training.monitor_metric]
            self.checkpoint_callback(monitor_score, self.model, self.optimizer, epoch, val_metrics)
            
            # Update best score
            if monitor_score > self.best_val_score:
                self.best_val_score = monitor_score
            
            # Early stopping
            if self.early_stopping(monitor_score, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation {self.config.training.monitor_metric}: {self.best_val_score:.4f}")
        
        # Close logging
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
    
    def save_training_plots(self, save_dir: str):
        """Save training plots"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 3, 2)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        # Metrics plot
        plt.subplot(1, 3, 3)
        epochs = range(len(self.train_losses))
        plt.plot(epochs, [1 - loss for loss in self.train_losses], label='Train Pseudo-Acc')
        plt.plot(epochs, [1 - loss for loss in self.val_losses], label='Val Pseudo-Acc')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('1 - Loss (Pseudo Accuracy)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {save_dir / 'training_plots.png'}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint.get('metrics', {})