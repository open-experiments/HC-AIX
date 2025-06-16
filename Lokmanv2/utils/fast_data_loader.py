"""
Fast Data Loader for Organized CT Breast Cancer Dataset
Optimized for high-performance training and inference
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class FastCTDataset(Dataset):
    """High-performance dataset loader for organized CT data"""
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 use_cache: bool = True,
                 preload_data: bool = False,
                 transform=None,
                 target_size: Tuple[int, int, int] = (160, 160, 160)):
        """
        Initialize fast dataset loader
        
        Args:
            data_root: Path to organized data directory
            split: 'train', 'val', or 'test'
            use_cache: Use cached preprocessed data if available
            preload_data: Load all data into memory at initialization
            transform: Data augmentation transforms
            target_size: Target volume size for resizing
        """
        self.data_root = Path(data_root)
        self.split = split
        self.use_cache = use_cache
        self.preload_data = preload_data
        self.transform = transform
        self.target_size = target_size
        
        # Initialize data structures
        self.samples = []
        self.data_cache = {}
        self.metadata_cache = {}
        
        # Load dataset indices and metadata
        self._load_dataset_info()
        
        # Preload data if requested
        if self.preload_data:
            self._preload_all_data()
        
        logger.info(f"FastCTDataset initialized: {len(self.samples)} samples in {split} split")
    
    def _load_dataset_info(self):
        """Load dataset splits and metadata for folder-based labeling"""
        
        # Load split data (simple list of series IDs)
        split_file = self.data_root / 'splits' / f'{self.split}.json'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file) as f:
            series_ids = json.load(f)
        
        # Load processed data index
        processed_index_file = self.data_root / 'processed' / 'dataset_index_with_splits.csv'
        
        if not processed_index_file.exists():
            raise FileNotFoundError(f"Dataset index not found: {processed_index_file}")
        
        df = pd.read_csv(processed_index_file)
        
        # Build sample list from series IDs in this split
        for series_id in series_ids:
            # Find matching row in dataset index
            matching_rows = df[df['series_id'] == series_id]
            
            if len(matching_rows) == 0:
                logger.warning(f"Series {series_id} not found in dataset index")
                continue
            
            row = matching_rows.iloc[0]
            
            sample = {
                'series_id': series_id,
                'patient_id': row['patient_dir'],
                'study_id': row['study_dir'],
                'series_dir': row['series_dir'],
                'label': row['label'],
                'label_name': row['label_name'],
                'volume_path': Path(row['volume_path']),
                'mask_path': Path(row['mask_path']),
                'metadata_path': Path(row['metadata_path']),
                'has_mask': row['has_mask'],
                'num_slices': row['num_slices'],
                'scan_type': row['scan_type'],
                'has_contrast': row['has_contrast'],
                'is_chest': row['is_chest'],
                'is_abdomen': row['is_abdomen'],
                'use_processed': True
            }
            
            self.samples.append(sample)
        
        logger.info(f"Found {len(self.samples)} samples in {self.split} split")
        
        # Log label distribution
        label_counts = {}
        for sample in self.samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"Label distribution: {label_counts}")
    
    def _preload_all_data(self):
        """Preload all data into memory for fastest access"""
        
        logger.info("Preloading all data into memory...")
        
        def load_sample_data(idx):
            return idx, self._load_volume_and_mask(idx)
        
        # Use threading for parallel loading
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load_sample_data, i) for i in range(len(self.samples))]
            
            for future in futures:
                idx, (volume, mask, metadata) = future.result()
                self.data_cache[idx] = (volume, mask, metadata)
        
        logger.info(f"Preloaded {len(self.data_cache)} samples into memory")
    
    def _load_volume_and_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load volume and mask for given index"""
        
        sample = self.samples[idx]
        
        # Try to load from processed data first
        if sample['use_processed'] and sample['volume_path'].exists():
            try:
                volume = np.load(sample['volume_path'])
                
                # Load mask if available
                if sample['mask_path'].exists():
                    mask = np.load(sample['mask_path'])
                else:
                    mask = np.zeros_like(volume, dtype=np.int64)
                
                # Load metadata
                metadata = {}
                if sample['metadata_path'].exists():
                    with open(sample['metadata_path']) as f:
                        metadata = json.load(f)
                
                return volume, mask, metadata
                
            except Exception as e:
                logger.warning(f"Could not load processed data for {idx}: {e}")
        
        # Fallback to loading from raw DICOM
        return self._load_from_raw_dicom(sample)
    
    def _load_from_raw_dicom(self, sample: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load volume from raw DICOM files"""
        
        try:
            import pydicom
            
            raw_path = sample['raw_path']
            dicom_files = sorted(raw_path.glob('slice_*.dcm'))
            
            if not dicom_files:
                raise FileNotFoundError(f"No DICOM files found in {raw_path}")
            
            # Load first slice to get dimensions
            first_slice = pydicom.dcmread(dicom_files[0])
            rows, cols = first_slice.Rows, first_slice.Columns
            
            # Initialize volume
            volume = np.zeros((len(dicom_files), rows, cols), dtype=np.float32)
            
            # Load all slices
            for i, dcm_file in enumerate(dicom_files):
                slice_data = pydicom.dcmread(dcm_file)
                volume[i] = slice_data.pixel_array.astype(np.float32)
            
            # Create empty mask
            mask = np.zeros_like(volume, dtype=np.uint8)
            
            # Basic metadata
            metadata = {
                'patient_id': sample['patient_id'],
                'series_id': sample['series_id'],
                'original_shape': volume.shape,
                'slice_thickness': getattr(first_slice, 'SliceThickness', None),
                'pixel_spacing': getattr(first_slice, 'PixelSpacing', None)
            }
            
            return volume, mask, metadata
            
        except Exception as e:
            logger.error(f"Could not load DICOM data: {e}")
            # Return dummy data
            volume = np.zeros(self.target_size, dtype=np.float32)
            mask = np.zeros(self.target_size, dtype=np.uint8)
            metadata = {'error': str(e)}
            return volume, mask, metadata
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample by index"""
        
        sample = self.samples[idx]
        
        # Try cache first
        if idx in self.data_cache:
            volume, mask, metadata = self.data_cache[idx]
        else:
            volume, mask, metadata = self._load_volume_and_mask(idx)
            
            # Cache if using cache
            if self.use_cache:
                self.data_cache[idx] = (volume, mask, metadata)
        
        # Resize to target size if needed
        if volume.shape != self.target_size:
            volume = self._resize_volume(volume, self.target_size)
            mask = self._resize_volume(mask, self.target_size)
        
        # Normalize volume
        volume = self._normalize_volume(volume)
        
        # Check for None values
        if volume is None:
            logger.error(f"Volume is None for sample {idx}")
            volume = np.zeros(self.target_size, dtype=np.float32)
        if mask is None:
            logger.error(f"Mask is None for sample {idx}")
            mask = np.zeros(self.target_size, dtype=np.int64)
        
        # Convert to tensors
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()  # Add channel dim
        mask_tensor = torch.from_numpy(mask).long()
        
        # Create classification label tensor
        class_label = torch.tensor(sample['label'], dtype=torch.long)
        
        # Apply transforms if specified
        if self.transform:
            volume_tensor, mask_tensor = self.transform(volume_tensor, mask_tensor)
        
        # Convert Path objects to strings for JSON serialization
        sample_info = dict(sample)
        for key, value in sample_info.items():
            if isinstance(value, Path):
                sample_info[key] = str(value)
        
        # Check metadata and sample_info for None values
        if metadata is None:
            metadata = {}
        
        # Filter out None values from sample_info
        filtered_sample_info = {}
        for key, value in sample_info.items():
            if value is not None:
                filtered_sample_info[key] = value
        
        return {
            'volume': volume_tensor,
            'mask': mask_tensor,
            'label': class_label,  # Add classification label
            'metadata': metadata,
            'sample_info': filtered_sample_info
        }
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize volume to target size"""
        try:
            from scipy.ndimage import zoom
            
            zoom_factors = [t / s for t, s in zip(target_size, volume.shape)]
            return zoom(volume, zoom_factors, order=1)
        except ImportError:
            # Fallback to simple interpolation
            return np.resize(volume, target_size)
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume intensity"""
        
        # Clip to reasonable CT range
        volume = np.clip(volume, -1000, 1000)
        
        # Normalize to [0, 1]
        volume = (volume + 1000) / 2000.0
        
        return volume.astype(np.float32)
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get sample information without loading data"""
        return self.samples[idx]
    
    def get_split_summary(self) -> Dict:
        """Get summary statistics for this split"""
        
        total_samples = len(self.samples)
        with_annotations = sum(1 for s in self.samples if s.get('has_mask', False))
        processed_available = sum(1 for s in self.samples if s['use_processed'])
        
        series_types = {}
        for sample in self.samples:
            series_type = sample.get('scan_type', 'unknown')
            series_types[series_type] = series_types.get(series_type, 0) + 1
        
        return {
            'total_samples': total_samples,
            'with_annotations': with_annotations,
            'processed_available': processed_available,
            'series_types': series_types,
            'cache_size': len(self.data_cache)
        }

class FastDataLoaderFactory:
    """Factory for creating optimized data loaders"""
    
    @staticmethod
    def create_loaders(data_root: str,
                      batch_size: int = 2,
                      num_workers: int = 4,
                      use_cache: bool = True,
                      preload_train: bool = False,
                      target_size: Tuple[int, int, int] = (160, 160, 160)) -> Dict[str, DataLoader]:
        """
        Create optimized data loaders for all splits
        
        Args:
            data_root: Path to organized data
            batch_size: Batch size for training
            num_workers: Number of worker processes
            use_cache: Enable caching for faster access
            preload_train: Preload training data into memory
            target_size: Target volume dimensions
        
        Returns:
            Dictionary with 'train', 'val', 'test' data loaders
        """
        
        loaders = {}
        
        for split in ['train', 'val', 'test']:
            try:
                dataset = FastCTDataset(
                    data_root=data_root,
                    split=split,
                    use_cache=use_cache,
                    preload_data=(preload_train and split == 'train'),
                    target_size=target_size
                )
                
                # Adjust batch size for validation/test
                split_batch_size = batch_size if split == 'train' else min(batch_size, 1)
                
                loader_kwargs = {
                    'batch_size': split_batch_size,
                    'shuffle': (split == 'train'),
                    'num_workers': num_workers,
                    'pin_memory': True
                }
                
                if num_workers > 0:
                    loader_kwargs['persistent_workers'] = True
                    loader_kwargs['prefetch_factor'] = 2
                
                loader = DataLoader(dataset, **loader_kwargs)
                
                loaders[split] = loader
                
                # Log summary
                summary = dataset.get_split_summary()
                logger.info(f"{split.upper()} loader: {summary}")
                
            except FileNotFoundError:
                logger.warning(f"Split {split} not found, skipping")
        
        return loaders
    
    @staticmethod
    def create_inference_loader(data_root: str,
                               patient_ids: Optional[List[str]] = None,
                               batch_size: int = 1,
                               target_size: Tuple[int, int, int] = (160, 160, 160)) -> DataLoader:
        """
        Create data loader for inference on specific patients
        
        Args:
            data_root: Path to organized data
            patient_ids: List of patient IDs to include (None for all)
            batch_size: Batch size for inference
            target_size: Target volume dimensions
        
        Returns:
            DataLoader for inference
        """
        
        # Create custom dataset for inference
        dataset = FastCTDataset(
            data_root=data_root,
            split='test',  # Use test split as base
            use_cache=True,
            preload_data=False,
            target_size=target_size
        )
        
        # Filter by patient IDs if specified
        if patient_ids:
            filtered_samples = [
                sample for sample in dataset.samples 
                if sample['patient_id'] in patient_ids
            ]
            dataset.samples = filtered_samples
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Single threaded for inference
            pin_memory=False
        )
        
        logger.info(f"Inference loader created: {len(dataset)} samples")
        return loader

# Utility functions
def get_dataset_statistics(data_root: str) -> Dict:
    """Get comprehensive dataset statistics"""
    
    data_path = Path(data_root)
    
    # Load metadata
    with open(data_path / 'metadata' / 'patient_registry.json') as f:
        registry = json.load(f)
    
    with open(data_path / 'splits' / 'summary.json') as f:
        split_summary = json.load(f)
    
    # Count processed data
    processed_volumes = len(list((data_path / 'processed' / 'volumes').glob('*.npy')))
    processed_masks = len(list((data_path / 'processed' / 'masks').glob('*.npy')))
    
    return {
        'total_patients': len(registry),
        'total_processed_volumes': processed_volumes,
        'total_processed_masks': processed_masks,
        'split_summary': split_summary,
        'data_organization_date': split_summary.get('created_at', 'unknown')
    }

def validate_data_integrity(data_root: str) -> Dict[str, bool]:
    """Validate data integrity and accessibility"""
    
    checks = {}
    data_path = Path(data_root)
    
    # Check required directories
    required_dirs = ['raw', 'processed', 'metadata', 'splits', 'annotations']
    for dir_name in required_dirs:
        checks[f'{dir_name}_exists'] = (data_path / dir_name).exists()
    
    # Check split files
    for split in ['train', 'val', 'test']:
        split_file = data_path / 'splits' / f'{split}.json'
        checks[f'{split}_split_exists'] = split_file.exists()
    
    # Check processed data consistency
    volume_dir = data_path / 'processed' / 'volumes'
    mask_dir = data_path / 'processed' / 'masks'
    
    if volume_dir.exists() and mask_dir.exists():
        volume_files = set(f.stem.replace('_volume', '') for f in volume_dir.glob('*_volume.npy'))
        mask_files = set(f.stem.replace('_mask', '') for f in mask_dir.glob('*_mask.npy'))
        checks['processed_data_consistent'] = volume_files == mask_files
    else:
        checks['processed_data_consistent'] = False
    
    return checks

# Example usage and testing
if __name__ == "__main__":
    # Test the fast data loader
    data_root = "/Users/fenar/projects/gitrepo/HC-AIX/New/breast-cancer-ct-detection/data"
    
    print("🚀 Testing Fast Data Loader")
    print("=" * 40)
    
    # Get statistics
    stats = get_dataset_statistics(data_root)
    print(f"📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Validate integrity
    integrity = validate_data_integrity(data_root)
    print(f"\n🔍 Data Integrity Checks:")
    for check, passed in integrity.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    # Create data loaders
    print(f"\n🏗️  Creating Data Loaders...")
    try:
        loaders = FastDataLoaderFactory.create_loaders(
            data_root=data_root,
            batch_size=1,  # Small batch for testing
            num_workers=0,  # Single threaded for testing
            use_cache=True,
            preload_train=False
        )
        
        print(f"✅ Created {len(loaders)} data loaders")
        
        # Test loading a batch
        if 'train' in loaders:
            print(f"\n🧪 Testing data loading...")
            train_loader = loaders['train']
            batch = next(iter(train_loader))
            
            print(f"✅ Loaded batch successfully:")
            print(f"   Volume shape: {batch['volume'].shape}")
            print(f"   Mask shape: {batch['mask'].shape}")
            print(f"   Patient ID: {batch['sample_info'][0]['patient_id']}")
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
    
    print("=" * 40)
    print("🎉 Fast Data Loader test completed!")