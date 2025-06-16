"""
Advanced dataset class for CT-based breast cancer metastasis detection.
Handles 3D volumes with segmentation masks and advanced augmentations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Callable
import logging
import json
from scipy.ndimage import zoom, rotate, gaussian_filter
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import albumentations as A
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CTVolumeDataset(Dataset):
    """
    Advanced dataset for 3D CT volumes with segmentation masks.
    Supports comprehensive augmentations for medical imaging.
    """
    
    def __init__(self, 
                 data_root: str,
                 dataset_index: str,
                 mode: str = "train",
                 transforms: Optional[Callable] = None,
                 target_size: Tuple[int, int, int] = (128, 128, 128),
                 use_cache: bool = True,
                 cache_size: int = 50):
        """
        Initialize CT volume dataset
        
        Args:
            data_root: Root directory with processed data
            dataset_index: Path to dataset index CSV
            mode: Dataset mode ('train', 'val', 'test')
            transforms: Augmentation transforms
            target_size: Target volume size (D, H, W)
            use_cache: Whether to cache volumes in memory
            cache_size: Maximum number of volumes to cache
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.transforms = transforms
        self.target_size = target_size
        self.use_cache = use_cache
        self.cache_size = cache_size
        
        # Load dataset index
        self.dataset_df = pd.read_csv(dataset_index)
        # Filter by success column if it exists
        if 'success' in self.dataset_df.columns:
            self.dataset_df = self.dataset_df[self.dataset_df['success'] == True].reset_index(drop=True)
        
        # Filter by mode if split information is available
        if 'split' in self.dataset_df.columns:
            self.dataset_df = self.dataset_df[self.dataset_df['split'] == mode].reset_index(drop=True)
        
        # Initialize cache
        self.cache = {} if use_cache else None
        
        # Get class distribution
        self._analyze_class_distribution()
        
        logger.info(f"Loaded {len(self.dataset_df)} samples for {mode} mode")
    
    def _analyze_class_distribution(self):
        """Analyze class distribution in segmentation masks"""
        self.class_counts = {}
        self.class_weights = {}
        
        # Sample a few volumes to get class distribution
        sample_size = min(10, len(self.dataset_df))
        sample_indices = np.random.choice(len(self.dataset_df), sample_size, replace=False)
        
        all_classes = []
        for idx in sample_indices:
            try:
                _, mask, _ = self._load_volume_and_mask(idx)
                if mask is not None:
                    unique_classes = np.unique(mask)
                    all_classes.extend(unique_classes)
            except Exception as e:
                logger.warning(f"Could not load sample {idx} for class analysis: {e}")
        
        # Count classes
        unique_classes = np.unique(all_classes)
        for cls in unique_classes:
            self.class_counts[int(cls)] = all_classes.count(cls)
        
        # Calculate class weights (inverse frequency)
        total_samples = sum(self.class_counts.values())
        for cls, count in self.class_counts.items():
            self.class_weights[cls] = total_samples / (len(unique_classes) * count)
        
        logger.info(f"Class distribution: {self.class_counts}")
        logger.info(f"Class weights: {self.class_weights}")
    
    def __len__(self) -> int:
        return len(self.dataset_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get volume and mask for given index"""
        
        # Check cache first
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        
        try:
            # Load volume and mask
            volume, mask, metadata = self._load_volume_and_mask(idx)
            
            # Resize to target size if needed
            if volume.shape != self.target_size:
                volume = self._resize_volume(volume, self.target_size)
                if mask is not None:
                    mask = self._resize_volume(mask, self.target_size, is_mask=True)
            
            # Normalize volume (assuming CT intensity range)
            # Clip to typical CT range and normalize to [0, 1]
            volume = np.clip(volume, -1000, 1000)
            volume = (volume + 1000) / 2000.0
            
            # Apply transforms
            if self.transforms:
                volume, mask = self.transforms(volume, mask)
            
            # Convert to tensors (fix negative strides)
            volume_tensor = torch.from_numpy(volume.copy()).float()
            
            # Add channel dimension if needed
            if volume_tensor.dim() == 3:
                volume_tensor = volume_tensor.unsqueeze(0)  # Add channel dimension
            
            result = {
                'volume': volume_tensor,
                'series_id': self.dataset_df.iloc[idx]['series_id'],
                'label': self.dataset_df.iloc[idx]['label'],
                'metadata': metadata
            }
            
            # Add mask if available
            if mask is not None:
                mask_tensor = torch.from_numpy(mask.copy()).long()
                result['mask'] = mask_tensor
                result['has_mask'] = True
            else:
                # Create dummy mask for consistency
                result['mask'] = torch.zeros(volume_tensor.shape[1:], dtype=torch.long)
                result['has_mask'] = False
            
            # Cache if enabled
            if self.use_cache and len(self.cache) < self.cache_size:
                self.cache[idx] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy data to prevent training interruption
            return self._get_dummy_sample()
    
    def _load_volume_and_mask(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """Load volume and mask from disk"""
        
        row = self.dataset_df.iloc[idx]
        
        # Support both old format (output_path) and new format (volume_path/mask_path)
        if 'output_path' in row:
            # Old format
            output_path = Path(row['output_path'])
            volume_path = output_path / "volume.npy"
            mask_path = output_path / "mask.npy"
            metadata_path = output_path / "metadata.json"
        else:
            # New format - direct paths
            volume_path = Path(row['volume_path'])
            mask_path = Path(row['mask_path']) if 'mask_path' in row and pd.notna(row['mask_path']) else None
            metadata_path = Path(row['metadata_path']) if 'metadata_path' in row and pd.notna(row['metadata_path']) else None
        
        # Load volume
        volume = np.load(volume_path)
        # Fix negative strides issue
        volume = np.array(volume, copy=True)
        
        # Load mask if available
        mask = None
        if mask_path and mask_path.exists():
            mask = np.load(mask_path)
            # Fix negative strides issue
            mask = np.array(mask, copy=True)
        
        # Load metadata
        metadata = {}
        if metadata_path and metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return volume, mask, metadata
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int], is_mask: bool = False) -> np.ndarray:
        """Resize volume to target size using zoom"""
        if volume.shape == target_size:
            return volume
            
        # Calculate zoom factors
        zoom_factors = [t / s for t, s in zip(target_size, volume.shape)]
        
        # Use order=0 for masks (nearest neighbor) and order=1 for volumes (linear)
        order = 0 if is_mask else 1
        
        # Resize volume
        resized = zoom(volume, zoom_factors, order=order)
        
        # Ensure exact target size (zoom might have small precision errors)
        if resized.shape != target_size:
            # Crop or pad to exact target size
            result = np.zeros(target_size, dtype=resized.dtype)
            
            # Calculate slicing/padding for each dimension
            slices = []
            for i in range(3):
                actual_size = resized.shape[i]
                target_dim = target_size[i]
                
                if actual_size >= target_dim:
                    # Crop from center
                    start = (actual_size - target_dim) // 2
                    end = start + target_dim
                    slices.append(slice(start, end))
                else:
                    # Will pad later
                    slices.append(slice(None))
            
            # Extract the cropped volume
            cropped = resized[tuple(slices)]
            
            # Pad if necessary
            pad_widths = []
            for i in range(3):
                actual_size = cropped.shape[i]
                target_dim = target_size[i]
                if actual_size < target_dim:
                    diff = target_dim - actual_size
                    pad_before = diff // 2
                    pad_after = diff - pad_before
                    pad_widths.append((pad_before, pad_after))
                else:
                    pad_widths.append((0, 0))
            
            if any(sum(pw) > 0 for pw in pad_widths):
                result = np.pad(cropped, pad_widths, mode='constant', constant_values=0)
            else:
                result = cropped
                
            return result
        
        return resized
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Create dummy sample for error cases"""
        dummy_volume = torch.zeros((1,) + self.target_size, dtype=torch.float32)
        dummy_mask = torch.zeros(self.target_size, dtype=torch.long)
        
        return {
            'volume': dummy_volume,
            'mask': dummy_mask,
            'has_mask': False,
            'series_id': 'dummy',
            'label': 0,
            'metadata': {}
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss computation"""
        max_class = max(self.class_weights.keys()) if self.class_weights else 3
        weights = torch.ones(max_class + 1)
        
        for cls, weight in self.class_weights.items():
            weights[cls] = weight
        
        return weights

class CTVolumeTransforms:
    """Advanced 3D transforms for CT volumes"""
    
    def __init__(self, 
                 mode: str = "train",
                 rotation_range: float = 15.0,
                 translation_range: float = 0.1,
                 scaling_range: Tuple[float, float] = (0.9, 1.1),
                 flip_probability: float = 0.5,
                 elastic_deformation: bool = True,
                 gaussian_noise_std: float = 0.01,
                 intensity_shift_range: float = 0.1,
                 gamma_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Initialize transforms
        
        Args:
            mode: 'train', 'val', or 'test'
            rotation_range: Max rotation in degrees
            translation_range: Max translation as fraction of image size
            scaling_range: Scaling factor range
            flip_probability: Probability of flipping
            elastic_deformation: Whether to apply elastic deformation
            gaussian_noise_std: Standard deviation for Gaussian noise
            intensity_shift_range: Range for intensity shifting
            gamma_range: Range for gamma correction
        """
        self.mode = mode
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scaling_range = scaling_range
        self.flip_probability = flip_probability
        self.elastic_deformation = elastic_deformation
        self.gaussian_noise_std = gaussian_noise_std
        self.intensity_shift_range = intensity_shift_range
        self.gamma_range = gamma_range
    
    def __call__(self, volume: np.ndarray, 
                 mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply transforms to volume and mask"""
        
        if self.mode == "train":
            return self._apply_train_transforms(volume, mask)
        elif self.mode == "val":
            return self._apply_val_transforms(volume, mask)
        else:  # test
            return self._apply_test_transforms(volume, mask)
    
    def _apply_train_transforms(self, volume: np.ndarray, 
                              mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply training transforms (aggressive augmentation)"""
        
        # Spatial transforms
        if np.random.random() < 0.7:
            volume, mask = self._apply_spatial_transforms(volume, mask)
        
        # Intensity transforms (only on volume)
        if np.random.random() < 0.6:
            volume = self._apply_intensity_transforms(volume)
        
        # Noise
        if np.random.random() < 0.3:
            volume = self._add_gaussian_noise(volume)
        
        # Elastic deformation
        if self.elastic_deformation and np.random.random() < 0.2:
            volume, mask = self._apply_elastic_deformation(volume, mask)
        
        return volume, mask
    
    def _apply_val_transforms(self, volume: np.ndarray, 
                            mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply validation transforms (minimal augmentation)"""
        
        # Only apply flipping for validation
        if np.random.random() < 0.5:
            volume, mask = self._apply_flip(volume, mask)
        
        return volume, mask
    
    def _apply_test_transforms(self, volume: np.ndarray, 
                             mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply test transforms (no augmentation)"""
        
        # No transforms for test data
        return volume, mask
    
    def _apply_spatial_transforms(self, volume: np.ndarray, 
                                mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply spatial transforms"""
        
        # Rotation
        if np.random.random() < 0.7:
            volume, mask = self._apply_rotation(volume, mask)
        
        # Translation
        if np.random.random() < 0.5:
            volume, mask = self._apply_translation(volume, mask)
        
        # Scaling
        if np.random.random() < 0.5:
            volume, mask = self._apply_scaling(volume, mask)
        
        # Flipping
        if np.random.random() < self.flip_probability:
            volume, mask = self._apply_flip(volume, mask)
        
        return volume, mask
    
    def _apply_rotation(self, volume: np.ndarray, 
                       mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random rotation"""
        
        # Random rotation angles for each axis
        angles = np.random.uniform(-self.rotation_range, self.rotation_range, 3)
        
        # Rotate volume
        for i, angle in enumerate(angles):
            if abs(angle) > 0.1:  # Only rotate if angle is significant
                axes = [(0, 1), (0, 2), (1, 2)][i]
                volume = rotate(volume, angle, axes=axes, reshape=False, order=1)
                
                if mask is not None:
                    mask = rotate(mask, angle, axes=axes, reshape=False, order=0)
        
        return volume, mask
    
    def _apply_translation(self, volume: np.ndarray, 
                          mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random translation"""
        
        # Random translation for each axis
        shape = volume.shape
        shifts = [np.random.uniform(-self.translation_range * s, 
                                   self.translation_range * s) for s in shape]
        
        # Apply translation using scipy
        from scipy.ndimage import shift
        
        volume = shift(volume, shifts, order=1)
        if mask is not None:
            mask = shift(mask, shifts, order=0)
        
        return volume, mask
    
    def _apply_scaling(self, volume: np.ndarray, 
                      mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random scaling"""
        
        # Random scaling factor
        scale_factor = np.random.uniform(self.scaling_range[0], self.scaling_range[1])
        
        # Apply scaling
        volume = zoom(volume, scale_factor, order=1)
        if mask is not None:
            mask = zoom(mask, scale_factor, order=0)
        
        # Crop or pad to original size
        original_shape = volume.shape
        if scale_factor != 1.0:
            volume = self._crop_or_pad(volume, original_shape)
            if mask is not None:
                mask = self._crop_or_pad(mask, original_shape)
        
        return volume, mask
    
    def _apply_flip(self, volume: np.ndarray, 
                   mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random flipping"""
        
        # Random flip axes
        axes = np.random.choice([0, 1, 2], size=np.random.randint(1, 4), replace=False)
        
        for axis in axes:
            volume = np.flip(volume, axis=axis)
            if mask is not None:
                mask = np.flip(mask, axis=axis)
        
        return volume, mask
    
    def _apply_intensity_transforms(self, volume: np.ndarray) -> np.ndarray:
        """Apply intensity transforms to volume"""
        
        # Intensity shift
        if np.random.random() < 0.5:
            shift = np.random.uniform(-self.intensity_shift_range, self.intensity_shift_range)
            volume = np.clip(volume + shift, 0, 1)
        
        # Gamma correction
        if np.random.random() < 0.3:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            volume = np.power(volume, gamma)
        
        # Contrast adjustment
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.8, 1.2)
            mean_val = volume.mean()
            volume = np.clip((volume - mean_val) * factor + mean_val, 0, 1)
        
        return volume
    
    def _add_gaussian_noise(self, volume: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to volume"""
        
        noise = np.random.normal(0, self.gaussian_noise_std, volume.shape)
        volume = np.clip(volume + noise, 0, 1)
        
        return volume
    
    def _apply_elastic_deformation(self, volume: np.ndarray, 
                                  mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply elastic deformation"""
        
        from scipy.ndimage import map_coordinates
        
        # Generate displacement fields
        shape = volume.shape
        dx = gaussian_filter(np.random.randn(*shape), sigma=3) * 2
        dy = gaussian_filter(np.random.randn(*shape), sigma=3) * 2
        dz = gaussian_filter(np.random.randn(*shape), sigma=3) * 2
        
        # Create coordinate grids
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
        
        # Apply deformation
        volume = map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)
        
        if mask is not None:
            mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)
        
        return volume, mask
    
    def _crop_or_pad(self, array: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Crop or pad array to target shape"""
        
        current_shape = array.shape
        
        # Calculate crop/pad amounts
        crop_pad = []
        for i in range(len(target_shape)):
            diff = target_shape[i] - current_shape[i]
            if diff > 0:  # Need padding
                pad_before = diff // 2
                pad_after = diff - pad_before
                crop_pad.append((pad_before, pad_after))
            elif diff < 0:  # Need cropping
                crop_start = abs(diff) // 2
                crop_end = crop_start + target_shape[i]
                crop_pad.append((crop_start, crop_end))
            else:  # Same size
                crop_pad.append((0, 0))
        
        # Apply cropping/padding
        if any(cp[0] > 0 or cp[1] > 0 for cp in crop_pad):
            # Padding needed
            array = np.pad(array, crop_pad, mode='constant', constant_values=0)
        
        # Apply cropping if needed
        slices = []
        for i, (start, end) in enumerate(crop_pad):
            if end == 0:
                slices.append(slice(None))
            elif start < 0:  # Cropping case
                slices.append(slice(-start, -start + target_shape[i]))
            else:
                slices.append(slice(None))
        
        if any(isinstance(s, slice) and s.start is not None for s in slices):
            array = array[tuple(slices)]
        
        return array

def create_data_loaders(config, dataset_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        config: Configuration object
        dataset_df: Dataset DataFrame
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Split dataset if not already split
    if 'split' not in dataset_df.columns:
        train_df, temp_df = train_test_split(
            dataset_df, 
            test_size=config.data.val_ratio + config.data.test_ratio,
            random_state=config.data.random_seed,
            stratify=dataset_df['patient_id'] if len(dataset_df['patient_id'].unique()) > 1 else None
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=config.data.test_ratio / (config.data.val_ratio + config.data.test_ratio),
            random_state=config.data.random_seed,
            stratify=temp_df['patient_id'] if len(temp_df['patient_id'].unique()) > 1 else None
        )
        
        # Add split column
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        # Combine and save
        dataset_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        dataset_df.to_csv(Path(config.data.processed_data_dir) / "dataset_index_with_splits.csv", index=False)
    
    # Create transforms
    train_transforms = CTVolumeTransforms(
        mode="train",
        rotation_range=config.data.rotation_range,
        translation_range=config.data.translation_range,
        scaling_range=config.data.scaling_range,
        flip_probability=config.data.flip_probability,
        elastic_deformation=config.data.elastic_deformation,
        gaussian_noise_std=config.data.gaussian_noise_std
    )
    
    val_transforms = CTVolumeTransforms(mode="val")
    test_transforms = CTVolumeTransforms(mode="test")
    
    # Create datasets
    dataset_index_path = Path(config.data.processed_data_dir) / "dataset_index_with_splits.csv"
    
    train_dataset = CTVolumeDataset(
        data_root=config.data.processed_data_dir,
        dataset_index=str(dataset_index_path),
        mode="train",
        transforms=train_transforms if config.data.use_augmentation else None,
        target_size=config.data.target_size
    )
    
    val_dataset = CTVolumeDataset(
        data_root=config.data.processed_data_dir,
        dataset_index=str(dataset_index_path),
        mode="val",
        transforms=val_transforms,
        target_size=config.data.target_size
    )
    
    test_dataset = CTVolumeDataset(
        data_root=config.data.processed_data_dir,
        dataset_index=str(dataset_index_path),
        mode="test",
        transforms=test_transforms,
        target_size=config.data.target_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        persistent_workers=config.system.persistent_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        persistent_workers=config.system.persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Use batch size 1 for test
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory
    )
    
    logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader