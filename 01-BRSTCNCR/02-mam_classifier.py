#!/usr/bin/env python3

import os
import gc
import signal
import atexit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import numpy as np
from PIL import Image
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from rich import print as rprint
from rich.console import Console
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim import AdamW

console = Console()

def debug_print(msg, level='info'):
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(f"[{level}][{timestamp}] {msg}")

def cleanup_gpu(*args):
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    debug_print("GPU resources released", "info")
    if args:  # If called by signal handler
        exit(0)

class Config:
    def __init__(self):
        self.epochs = 30
        self.batch_size = 32
        self.total_batch_size = 64
        self.gradient_accumulation_steps = 4
        self.learning_rate = 5e-5
        self.weight_decay = 2e-4
        self.warmup_epochs = 2
        self.model_name = 'resnet18'
        self.dropout1 = 0.3
        self.dropout2 = 0.2
        self.hidden_size = 256
        self.num_classes = 2
        self.image_size = 512
        self.devices = [0, 1]
        self.num_workers = 8
        self.pin_memory = True
        self.prefetch_factor = 2
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

class MammogramDataset(Dataset):
    def __init__(self, split_file: str, split_name: str, augment=False):
        with open(split_file) as f:
            splits = json.load(f)
        self.files = splits[split_name]['files']
        self.labels = splits[split_name]['labels']
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((560, 560)),
                transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
                transforms.RandomErasing(p=0.3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'image': image, 'label': label, 'path': img_path}

class MammogramClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.build_training_components()
        self.print_setup_info()
    
    def build_training_components(self):
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scaler = amp.GradScaler('cuda')
        self.scheduler = self.build_scheduler()
    
    def build_model(self):
        model = models.resnet18(weights='DEFAULT')
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        feature_dim = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(feature_dim, self.config.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.config.hidden_size),
            nn.Dropout(self.config.dropout1),
            nn.Linear(self.config.hidden_size, self.config.num_classes)
        )
        
        model = model.cuda()
        return nn.DataParallel(model, device_ids=self.config.devices)
    
    def build_optimizer(self):
        return AdamW(self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    eps=1e-8)
    
    def build_scheduler(self):
        warmup = LinearLR(self.optimizer,
                         start_factor=0.1,
                         end_factor=1.0,
                         total_iters=self.config.warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer,
                                  T_max=self.config.epochs - self.config.warmup_epochs,
                                  eta_min=1e-6)
        return [warmup, cosine]
    
    def print_setup_info(self):
        for i in self.config.devices:
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            name = torch.cuda.get_device_name(i)
            debug_print(f"GPU {i}: {name} ({memory:.1f}GB)")
        debug_print(f"Batch size per GPU: {self.config.batch_size}")
        debug_print(f"Total batch size: {self.config.total_batch_size}")
        debug_print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        debug_print(f"Effective batch size: {self.config.total_batch_size * self.config.gradient_accumulation_steps}")
    
    def get_data_loaders(self, split_file: str):
        train_dataset = MammogramDataset(split_file, 'train', augment=True)
        val_dataset = MammogramDataset(split_file, 'validation', augment=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.total_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.total_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].cuda(non_blocking=True)
            labels = batch['label'].cuda(non_blocking=True)
            
            with amp.autocast('cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            gpu_stats = [f"GPU{i}: {torch.cuda.memory_reserved(i)/1e9:.1f}GB ({torch.cuda.utilization(i)}%)" 
                        for i in self.config.devices]
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.3f}',
                'acc': f'{100.*correct/total:.1f}%',
                'gpu': ' '.join(gpu_stats)
            })
        
        if epoch < self.config.warmup_epochs:
            self.scheduler[0].step()
        else:
            self.scheduler[1].step()
        
        return {
            'loss': total_loss/len(train_loader),
            'accuracy': 100.*correct/total
        }
    
    def validate(self, val_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].cuda(non_blocking=True)
                labels = batch['label'].cuda(non_blocking=True)
                
                with amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss/len(val_loader),
            'accuracy': 100.*correct/total
        }
    
    def train(self, split_file: str):
        try:
            train_loader, val_loader = self.get_data_loaders(split_file)
            best_acc = 0
            
            for epoch in range(self.config.epochs):
                train_metrics = self.train_epoch(train_loader, epoch)
                val_metrics = self.validate(val_loader)
                
                debug_print(
                    f"Epoch {epoch+1} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}% | "
                    f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%",
                    "success"
                )
                
                if val_metrics['accuracy'] > best_acc:
                    best_acc = val_metrics['accuracy']
                    self.save_checkpoint(epoch, best_acc)
        finally:
            # Clean up DataLoader workers
            if 'train_loader' in locals():
                del train_loader
            if 'val_loader' in locals():
                del val_loader
    
    def save_checkpoint(self, epoch: int, accuracy: float):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'scheduler_state_dict': [s.state_dict() for s in self.scheduler],
            'config': self.config.__dict__
        }
        
        save_path = self.config.checkpoint_dir / f'model_{int(accuracy)}_{epoch+1}.pth'
        torch.save(checkpoint, save_path)
        debug_print(f"Saved checkpoint: {save_path}", "success")

def main():
    try:
        config = Config()
        classifier = MammogramClassifier(config)
        classifier.train("dataset_splits.json")
    except Exception as e:
        debug_print(f"Training failed: {str(e)}", "error")
        raise
    finally:
        cleanup_gpu()

if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup_gpu)
    signal.signal(signal.SIGINT, cleanup_gpu)
    signal.signal(signal.SIGTERM, cleanup_gpu)
    
    main()
