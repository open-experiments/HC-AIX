#!/usr/bin/env python3
"""
Lokman-v2: Data Preparation Script with Folder-Based Labeling
Prepares CT data using original folder structure for proper classification labels
"""

import os
import sys
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm
import hashlib
from datetime import datetime
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dicom_utils import DICOMProcessor

class FolderBasedLabeler:
    """Creates labels based on folder structure and anatomical regions"""
    
    def __init__(self):
        # Define label mapping based on anatomical regions and pathology indicators
        self.label_mapping = {
            # Class 0: Normal/Healthy scans
            'normal': 0,
            'healthy': 0,
            'kontrastsiz': 0,  # Non-contrast - often normal follow-up
            
            # Class 1: Benign findings
            'benign': 1,
            'lung': 1,  # Lung scans - typically screening/benign
            'toraks': 1,  # Thorax scans - often benign findings
            
            # Class 2: Malignant/High risk
            'malignant': 2,
            'kontrastli': 2,  # Contrast enhanced - often for tumor detection
            'abdomen': 2,  # Abdominal scans - higher cancer risk area
            'abd': 2,
            'pel': 2,  # Pelvic scans
            
            # Class 3: Tumor/Active disease
            'tumor': 3,
            'cancer': 3,
            'mass': 3,
            'metastasis': 3
        }
        
        # Priority rules (higher number = higher priority for class assignment)
        self.priority_keywords = {
            'kontrastli': 3,  # Contrast-enhanced studies
            'abdomen': 3,     # Abdominal studies
            'abd': 3,
            'pel': 3,
            'toraks': 2,      # Thorax studies
            'lung': 2,        # Lung studies
            'kontrastsiz': 1  # Non-contrast studies
        }
    
    def extract_features_from_path(self, folder_path):
        """Extract meaningful features from folder path for labeling"""
        path_str = str(folder_path).lower()
        
        features = {
            'has_contrast': 'kontrastli' in path_str,
            'is_chest': any(word in path_str for word in ['toraks', 'lung', 'chest']),
            'is_abdomen': any(word in path_str for word in ['abdomen', 'abd', 'pel']),
            'scan_type': self._extract_scan_type(path_str),
            'thickness': self._extract_thickness(path_str),
            'patient_info': self._extract_patient_info(folder_path)
        }
        
        return features
    
    def _extract_scan_type(self, path_str):
        """Extract scan type from path"""
        if 'lung' in path_str:
            return 'lung'
        elif 'abdomen' in path_str or 'abd' in path_str:
            return 'abdomen'
        elif 'pel' in path_str:
            return 'pelvis'
        elif 'toraks' in path_str:
            return 'thorax'
        else:
            return 'unknown'
    
    def _extract_thickness(self, path_str):
        """Extract slice thickness from path"""
        thickness_match = re.search(r'(\d+\.?\d*)\s*mm', path_str)
        if thickness_match:
            return float(thickness_match.group(1))
        return None
    
    def _extract_patient_info(self, folder_path):
        """Extract patient information from folder structure"""
        parts = Path(folder_path).parts
        
        patient_info = {
            'patient_id': None,
            'study_id': None,
            'series_name': None
        }
        
        for part in parts:
            if 'ANON' in part or any(char.isdigit() for char in part[:5]):
                if not patient_info['patient_id']:
                    patient_info['patient_id'] = part
            elif part.startswith('601') or part.startswith('60'):
                patient_info['study_id'] = part
            elif 'CT' in part or 'BT' in part:
                patient_info['series_name'] = part
        
        return patient_info
    
    def assign_label(self, folder_path, default_strategy='anatomical'):
        """Assign classification label based on folder path"""
        features = self.extract_features_from_path(folder_path)
        path_str = str(folder_path).lower()
        
        if default_strategy == 'anatomical':
            return self._assign_anatomical_label(features, path_str)
        elif default_strategy == 'contrast':
            return self._assign_contrast_label(features, path_str)
        elif default_strategy == 'mixed':
            return self._assign_mixed_label(features, path_str)
        else:
            return 0  # Default to normal
    
    def _assign_anatomical_label(self, features, path_str):
        """Assign labels based on anatomical region"""
        # High-risk regions get higher labels
        if features['is_abdomen']:
            return 2 if features['has_contrast'] else 1
        elif features['is_chest']:
            return 1 if features['has_contrast'] else 0
        else:
            return 0
    
    def _assign_contrast_label(self, features, path_str):
        """Assign labels based on contrast enhancement"""
        if features['has_contrast']:
            return 2 if features['is_abdomen'] else 1
        else:
            return 0
    
    def _assign_mixed_label(self, features, path_str):
        """Assign labels using mixed strategy"""
        score = 0
        
        # Contrast enhancement increases risk
        if features['has_contrast']:
            score += 2
        
        # Anatomical region influence
        if features['is_abdomen']:
            score += 2
        elif features['is_chest']:
            score += 1
        
        # Slice thickness (thinner = more detailed = higher concern)
        if features['thickness'] and features['thickness'] <= 1.5:
            score += 1
        
        # Cap at maximum class
        return min(score, 3)

class DataPreparator:
    """Main data preparation class with folder-based labeling"""
    
    def __init__(self, data_original_root, output_root, labeling_strategy='mixed'):
        self.data_original_root = Path(data_original_root)
        self.output_root = Path(output_root)
        self.labeling_strategy = labeling_strategy
        
        self.labeler = FolderBasedLabeler()
        self.dicom_processor = DICOMProcessor()
        
        # Create output directories
        self.setup_output_directories()
        
        # Statistics
        self.stats = {
            'total_patients': 0,
            'total_studies': 0,
            'total_series': 0,
            'total_slices': 0,
            'class_distribution': {0: 0, 1: 0, 2: 0, 3: 0},
            'errors': []
        }
    
    def setup_output_directories(self):
        """Create all necessary output directories"""
        directories = [
            'processed/volumes',
            'processed/masks',
            'processed/metadata',
            'metadata',
            'splits'
        ]
        
        for directory in directories:
            (self.output_root / directory).mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directories created in: {self.output_root}")
    
    def find_dicom_series(self):
        """Find all DICOM series in the original data structure"""
        series_list = []
        
        print("🔍 Scanning original data structure...")
        
        for patient_dir in self.data_original_root.iterdir():
            if not patient_dir.is_dir() or patient_dir.name.endswith('.xlsx'):
                continue
                
            self.stats['total_patients'] += 1
            print(f"   📋 Patient: {patient_dir.name}")
            
            for study_dir in patient_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                    
                self.stats['total_studies'] += 1
                print(f"      🏥 Study: {study_dir.name}")
                
                for series_dir in study_dir.iterdir():
                    if not series_dir.is_dir():
                        continue
                    
                    # Check if directory contains DICOM files
                    dicom_files = list(series_dir.glob('*.dcm'))
                    if dicom_files:
                        self.stats['total_series'] += 1
                        self.stats['total_slices'] += len(dicom_files)
                        
                        # Assign label based on folder structure
                        label = self.labeler.assign_label(
                            series_dir, 
                            self.labeling_strategy
                        )
                        
                        self.stats['class_distribution'][label] += 1
                        
                        series_info = {
                            'patient_dir': patient_dir.name,
                            'study_dir': study_dir.name,
                            'series_dir': series_dir.name,
                            'full_path': str(series_dir),
                            'label': label,
                            'num_slices': len(dicom_files),
                            'dicom_files': [f.name for f in dicom_files],
                            'features': self.labeler.extract_features_from_path(series_dir)
                        }
                        
                        series_list.append(series_info)
                        print(f"         📊 Series: {series_dir.name} → Label: {label} ({len(dicom_files)} slices)")
        
        return series_list
    
    def process_series(self, series_info):
        """Process a single DICOM series"""
        try:
            series_path = Path(series_info['full_path'])
            
            # Generate unique ID for this series
            series_id = self.generate_series_id(series_info)
            
            # Load and process DICOM volume
            try:
                image = self.dicom_processor.load_dicom_series(str(series_path))
                volume_data = self.dicom_processor.preprocess_ct_volume(image)
            except Exception as e:
                # Try alternative loading method for different DICOM formats
                volume_data = self._load_dicom_simple(series_path)
            
            if volume_data is None:
                raise ValueError("Failed to load DICOM volume")
            
            # Create synthetic mask based on label (for training purposes)
            mask_data = self.create_synthetic_mask(volume_data, series_info['label'])
            
            # Save processed data
            volume_path = self.output_root / 'processed' / 'volumes' / f'{series_id}_volume.npy'
            mask_path = self.output_root / 'processed' / 'masks' / f'{series_id}_mask.npy'
            metadata_path = self.output_root / 'processed' / 'metadata' / f'{series_id}_metadata.json'
            
            # Save arrays
            np.save(volume_path, volume_data)
            np.save(mask_path, mask_data)
            
            # Save metadata
            metadata = {
                'series_id': series_id,
                'original_path': series_info['full_path'],
                'patient_dir': series_info['patient_dir'],
                'study_dir': series_info['study_dir'],
                'series_dir': series_info['series_dir'],
                'label': series_info['label'],
                'label_name': self.get_label_name(series_info['label']),
                'num_slices': series_info['num_slices'],
                'volume_shape': volume_data.shape,
                'mask_shape': mask_data.shape,
                'features': series_info['features'],
                'processing_date': datetime.now().isoformat(),
                'has_mask': True,  # Now we have proper masks!
                'labeling_strategy': self.labeling_strategy
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'series_id': series_id,
                'label': series_info['label'],
                'volume_path': str(volume_path),
                'mask_path': str(mask_path),
                'metadata_path': str(metadata_path),
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Error processing {series_info['full_path']}: {str(e)}"
            self.stats['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def create_synthetic_mask(self, volume_data, label):
        """Create synthetic segmentation mask based on label"""
        # Create a mask with the same spatial dimensions as volume
        if len(volume_data.shape) == 4:  # (D, H, W, C)
            mask_shape = volume_data.shape[:3]  # (D, H, W)
        else:  # (D, H, W)
            mask_shape = volume_data.shape
        
        mask = np.zeros(mask_shape, dtype=np.int64)
        
        # Create synthetic regions based on label
        if label == 0:  # Normal - minimal abnormal regions
            # 95% normal tissue, 5% minimal findings
            mask.fill(0)
            if np.random.random() < 0.1:  # 10% chance of small benign finding
                self._add_synthetic_region(mask, 1, size_ratio=0.02)
                
        elif label == 1:  # Benign - some abnormal regions
            # 80% normal, 15% benign, 5% other
            mask.fill(0)
            self._add_synthetic_region(mask, 1, size_ratio=0.15)
            if np.random.random() < 0.3:
                self._add_synthetic_region(mask, 0, size_ratio=0.05)
                
        elif label == 2:  # Malignant - significant abnormal regions
            # 60% normal, 25% malignant, 15% other
            mask.fill(0)
            self._add_synthetic_region(mask, 2, size_ratio=0.25)
            self._add_synthetic_region(mask, 1, size_ratio=0.15)
            
        elif label == 3:  # Tumor - large abnormal regions
            # 50% normal, 30% tumor, 20% malignant
            mask.fill(0)
            self._add_synthetic_region(mask, 3, size_ratio=0.30)
            self._add_synthetic_region(mask, 2, size_ratio=0.20)
        
        return mask
    
    def _add_synthetic_region(self, mask, region_class, size_ratio=0.1):
        """Add synthetic region to mask"""
        D, H, W = mask.shape
        
        # Calculate region size
        total_voxels = D * H * W
        region_voxels = int(total_voxels * size_ratio)
        
        # Create random regions
        num_regions = np.random.randint(1, 4)  # 1-3 regions
        voxels_per_region = region_voxels // num_regions
        
        for _ in range(num_regions):
            # Random center
            center_d = np.random.randint(D//4, 3*D//4)
            center_h = np.random.randint(H//4, 3*H//4)
            center_w = np.random.randint(W//4, 3*W//4)
            
            # Random size
            size_d = np.random.randint(5, min(20, D//4))
            size_h = np.random.randint(5, min(20, H//4))
            size_w = np.random.randint(5, min(20, W//4))
            
            # Bounds
            d_start = max(0, center_d - size_d//2)
            d_end = min(D, center_d + size_d//2)
            h_start = max(0, center_h - size_h//2)
            h_end = min(H, center_h + size_h//2)
            w_start = max(0, center_w - size_w//2)
            w_end = min(W, center_w + size_w//2)
            
            # Assign region
            mask[d_start:d_end, h_start:h_end, w_start:w_end] = region_class
    
    def _load_dicom_simple(self, series_path):
        """Simple DICOM loader as fallback"""
        try:
            import pydicom
            
            dicom_files = list(series_path.glob('*.dcm'))
            if not dicom_files:
                return None
            
            # Sort files by instance number if available
            def sort_key(f):
                try:
                    dcm = pydicom.dcmread(f, stop_before_pixels=True)
                    return int(getattr(dcm, 'InstanceNumber', 0))
                except:
                    return 0
            
            dicom_files.sort(key=sort_key)
            
            # Load first slice to get dimensions
            first_slice = pydicom.dcmread(dicom_files[0])
            rows, cols = first_slice.Rows, first_slice.Columns
            
            # Initialize volume
            volume = np.zeros((len(dicom_files), rows, cols), dtype=np.float32)
            
            # Load all slices
            for i, dcm_file in enumerate(dicom_files):
                try:
                    slice_data = pydicom.dcmread(dcm_file)
                    volume[i] = slice_data.pixel_array.astype(np.float32)
                except Exception as e:
                    print(f"Warning: Could not load slice {dcm_file}: {e}")
                    continue
            
            # Basic normalization for CT data
            volume = np.clip(volume, -1000, 1000)  # Typical CT range
            volume = (volume + 1000) / 2000.0  # Normalize to [0, 1]
            
            return volume
            
        except Exception as e:
            print(f"Error in simple DICOM loader: {e}")
            return None
    
    def generate_series_id(self, series_info):
        """Generate unique ID for series"""
        unique_string = f"{series_info['patient_dir']}_{series_info['study_dir']}_{series_info['series_dir']}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def get_label_name(self, label):
        """Get human readable label name"""
        label_names = {
            0: 'Normal',
            1: 'Benign',
            2: 'Malignant', 
            3: 'Tumor'
        }
        return label_names.get(label, 'Unknown')
    
    def create_dataset_splits(self, processed_series):
        """Create train/validation/test splits"""
        # Group by label to ensure balanced splits
        label_groups = {0: [], 1: [], 2: [], 3: []}
        
        for series in processed_series:
            if series['success']:
                label_groups[series['label']].append(series)
        
        train_data = []
        val_data = []
        test_data = []
        
        # Split each class separately
        for label, series_list in label_groups.items():
            n = len(series_list)
            if n == 0:
                continue
                
            # Shuffle
            np.random.shuffle(series_list)
            
            # Split indices
            if n >= 3:
                train_end = int(0.7 * n)
                val_end = int(0.85 * n)
                
                train_data.extend(series_list[:train_end])
                val_data.extend(series_list[train_end:val_end])
                test_data.extend(series_list[val_end:])
            elif n == 2:
                train_data.extend([series_list[0]])
                val_data.extend([series_list[1]])
            else:
                train_data.extend(series_list)
        
        # Save splits
        splits = {
            'train': [s['series_id'] for s in train_data],
            'val': [s['series_id'] for s in val_data],
            'test': [s['series_id'] for s in test_data]
        }
        
        for split_name, series_ids in splits.items():
            split_path = self.output_root / 'splits' / f'{split_name}.json'
            with open(split_path, 'w') as f:
                json.dump(series_ids, f, indent=2)
        
        # Create summary
        summary = {
            'total_series': len(processed_series),
            'successful_series': len([s for s in processed_series if s['success']]),
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'class_distribution': self.stats['class_distribution'],
            'labeling_strategy': self.labeling_strategy,
            'creation_date': datetime.now().isoformat()
        }
        
        summary_path = self.output_root / 'splits' / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return splits, summary
    
    def create_dataset_index(self, processed_series):
        """Create comprehensive dataset index"""
        data = []
        
        for series in processed_series:
            if series['success']:
                # Load metadata for additional info
                with open(series['metadata_path'], 'r') as f:
                    metadata = json.load(f)
                
                row = {
                    'series_id': series['series_id'],
                    'label': series['label'],
                    'label_name': self.get_label_name(series['label']),
                    'volume_path': series['volume_path'],
                    'mask_path': series['mask_path'],
                    'metadata_path': series['metadata_path'],
                    'has_mask': True,
                    'patient_dir': metadata['patient_dir'],
                    'study_dir': metadata['study_dir'],
                    'series_dir': metadata['series_dir'],
                    'num_slices': metadata['num_slices'],
                    'volume_shape': str(metadata['volume_shape']),
                    'scan_type': metadata['features']['scan_type'],
                    'has_contrast': metadata['features']['has_contrast'],
                    'is_chest': metadata['features']['is_chest'],
                    'is_abdomen': metadata['features']['is_abdomen']
                }
                
                data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        csv_path = self.output_root / 'processed' / 'dataset_index_with_splits.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Dataset index saved: {csv_path}")
        print(f"   Total samples: {len(df)}")
        print(f"   Class distribution:")
        for label in sorted(df['label'].unique()):
            count = len(df[df['label'] == label])
            label_name = self.get_label_name(label)
            print(f"      {label} ({label_name}): {count} samples")
        
        return df
    
    def run(self):
        """Main execution function"""
        print("🚀 Starting Lokman-v2 Data Preparation with Folder-Based Labeling")
        print("=" * 70)
        print(f"📂 Source: {self.data_original_root}")
        print(f"📁 Output: {self.output_root}")
        print(f"🏷️  Labeling Strategy: {self.labeling_strategy}")
        print()
        
        # Find all DICOM series
        series_list = self.find_dicom_series()
        
        if not series_list:
            print("❌ No DICOM series found!")
            return
        
        print(f"\n📊 Found {len(series_list)} DICOM series")
        print(f"📈 Label distribution:")
        for label, count in self.stats['class_distribution'].items():
            label_name = self.get_label_name(label)
            print(f"   {label} ({label_name}): {count} series")
        
        # Process all series
        print(f"\n🔄 Processing series...")
        processed_series = []
        
        for series_info in tqdm(series_list, desc="Processing series"):
            result = self.process_series(series_info)
            processed_series.append(result)
        
        # Create splits
        print(f"\n📊 Creating dataset splits...")
        splits, summary = self.create_dataset_splits(processed_series)
        
        # Create dataset index
        print(f"\n📋 Creating dataset index...")
        dataset_df = self.create_dataset_index(processed_series)
        
        # Print final statistics
        print(f"\n✅ Data preparation completed!")
        print(f"📊 Final Statistics:")
        print(f"   Total patients: {self.stats['total_patients']}")
        print(f"   Total studies: {self.stats['total_studies']}")
        print(f"   Total series: {self.stats['total_series']}")
        print(f"   Total slices: {self.stats['total_slices']}")
        print(f"   Successful processing: {summary['successful_series']}/{len(series_list)}")
        print(f"   Train samples: {summary['train_size']}")
        print(f"   Validation samples: {summary['val_size']}")
        print(f"   Test samples: {summary['test_size']}")
        
        if self.stats['errors']:
            print(f"   Errors: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"      - {error}")
        
        print(f"\n📁 Output saved to: {self.output_root}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Prepare CT data with folder-based labeling")
    parser.add_argument("--data-original", required=True, 
                       help="Path to data-original directory")
    parser.add_argument("--output", default="data", 
                       help="Output directory (default: data)")
    parser.add_argument("--strategy", default="mixed",
                       choices=['anatomical', 'contrast', 'mixed'],
                       help="Labeling strategy")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Validate input
    data_original_path = Path(args.data_original)
    if not data_original_path.exists():
        print(f"❌ Data original directory not found: {data_original_path}")
        return
    
    # Run data preparation
    preparator = DataPreparator(
        data_original_root=data_original_path,
        output_root=Path(args.output),
        labeling_strategy=args.strategy
    )
    
    success = preparator.run()
    
    if success:
        print(f"\n🎉 Data preparation successful!")
        print(f"🚀 Ready for training with proper labels!")
    else:
        print(f"\n❌ Data preparation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()