#!/usr/bin/env python3
"""
Dataset Augmentation Script - Expand dataset with rotations and mirroring
Creates additional training samples by applying geometric transformations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import argparse
import json
from tqdm import tqdm
import hashlib

def generate_series_id():
    """Generate a unique series ID"""
    return hashlib.md5(str(np.random.random()).encode()).hexdigest()[:12]

def augment_volume(volume, mask, augmentation_type):
    """Apply augmentation to volume and mask"""
    if augmentation_type == 'rotate_90':
        # Rotate 90 degrees clockwise around z-axis
        volume_aug = rotate(volume, -90, axes=(1, 2), reshape=False, order=1)
        mask_aug = rotate(mask, -90, axes=(1, 2), reshape=False, order=0)
    elif augmentation_type == 'rotate_180':
        # Rotate 180 degrees around z-axis
        volume_aug = rotate(volume, 180, axes=(1, 2), reshape=False, order=1)
        mask_aug = rotate(mask, 180, axes=(1, 2), reshape=False, order=0)
    elif augmentation_type == 'rotate_270':
        # Rotate 270 degrees clockwise around z-axis  
        volume_aug = rotate(volume, 90, axes=(1, 2), reshape=False, order=1)
        mask_aug = rotate(mask, 90, axes=(1, 2), reshape=False, order=0)
    elif augmentation_type == 'flip_horizontal':
        # Flip horizontally (left-right)
        volume_aug = np.flip(volume, axis=2)
        mask_aug = np.flip(mask, axis=2)
    elif augmentation_type == 'flip_vertical':
        # Flip vertically (up-down)
        volume_aug = np.flip(volume, axis=1)
        mask_aug = np.flip(mask, axis=1)
    elif augmentation_type == 'flip_depth':
        # Flip along depth axis
        volume_aug = np.flip(volume, axis=0)
        mask_aug = np.flip(mask, axis=0)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    return volume_aug.copy(), mask_aug.copy()

def create_augmented_metadata(original_metadata, augmentation_type, new_series_id):
    """Create metadata for augmented sample"""
    new_metadata = original_metadata.copy()
    new_metadata['series_id'] = new_series_id
    new_metadata['original_series_id'] = original_metadata.get('series_id', 'unknown')
    new_metadata['augmentation_type'] = augmentation_type
    new_metadata['is_augmented'] = True
    new_metadata['augmentation_timestamp'] = pd.Timestamp.now().isoformat()
    
    return new_metadata

def main():
    parser = argparse.ArgumentParser(description="Augment CT dataset with rotations and mirroring")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--data-root", help="Path to processed data (overrides config)")
    parser.add_argument("--augmentations", nargs='+', 
                       help="List of augmentations to apply (overrides config)")
    parser.add_argument("--splits", nargs='+', 
                       help="Which splits to augment (overrides config)")
    parser.add_argument("--preserve-ratios", action='store_true',
                       help="Maintain original train/val/test ratios (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    try:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"📝 Using config: {args.config}")
    except Exception as e:
        print(f"⚠️ Could not load config file {args.config}: {e}")
        print("⚠️ Using default settings")
        config = {
            'paths': {'data_root': 'data/processed'},
            'data': {'use_augmentation': True}
        }
    
    # Override config with command line arguments
    if args.data_root:
        data_root = args.data_root
    else:
        data_root = config['paths']['data_root']
        
    # Check if augmentation is enabled in config
    if not config.get('data', {}).get('use_augmentation', True):
        print("⚠️ Data augmentation is disabled in config.yaml (data.use_augmentation: false)")
        print("⚠️ Set data.use_augmentation: true to enable augmentation")
        return
    
    # Get augmentation settings from config or command line
    augmentation_config = config.get('data', {}).get('augmentation', {})
    
    augmentations = args.augmentations if args.augmentations else augmentation_config.get('types', ['rotate_90', 'flip_horizontal'])
    splits = args.splits if args.splits else augmentation_config.get('splits', ['train'])
    preserve_ratios = args.preserve_ratios if args.preserve_ratios else augmentation_config.get('preserve_ratios', True)
    
    print(f"📊 Augmentation settings:")
    print(f"   Types: {augmentations}")
    print(f"   Splits: {splits}")
    print(f"   Preserve ratios: {preserve_ratios}")
    
    data_root = Path(data_root)
    dataset_index_path = data_root / "dataset_index_with_splits.csv"
    
    # Load original dataset
    print("📊 Loading original dataset...")
    df_original = pd.read_csv(dataset_index_path)
    print(f"Original dataset: {len(df_original)} samples")
    
    # Create augmented samples
    augmented_rows = []
    
    for split in splits:
        split_df = df_original[df_original['split'] == split]
        print(f"\n🔄 Augmenting {split} split ({len(split_df)} samples)...")
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Augmenting {split}"):
            # Load original volume and mask
            try:
                volume = np.load(row['volume_path'])
                mask = np.load(row['mask_path']) if pd.notna(row['mask_path']) else None
                
                # Load original metadata
                if pd.notna(row['metadata_path']) and Path(row['metadata_path']).exists():
                    with open(row['metadata_path'], 'r') as f:
                        original_metadata = json.load(f)
                else:
                    original_metadata = {}
                
                # Apply each augmentation
                for aug_type in augmentations:
                    try:
                        # Generate new series ID
                        new_series_id = generate_series_id()
                        
                        # Apply augmentation
                        volume_aug, mask_aug = augment_volume(volume, mask, aug_type)
                        
                        # Create new file paths
                        volume_aug_path = data_root / "volumes" / f"{new_series_id}_volume.npy"
                        mask_aug_path = data_root / "masks" / f"{new_series_id}_mask.npy"
                        metadata_aug_path = data_root / "metadata" / f"{new_series_id}_metadata.json"
                        
                        # Save augmented volume and mask
                        np.save(volume_aug_path, volume_aug)
                        if mask is not None:
                            np.save(mask_aug_path, mask_aug)
                        
                        # Create and save augmented metadata
                        aug_metadata = create_augmented_metadata(original_metadata, aug_type, new_series_id)
                        with open(metadata_aug_path, 'w') as f:
                            json.dump(aug_metadata, f, indent=2)
                        
                        # Create new row for dataset index
                        new_row = row.copy()
                        new_row['series_id'] = new_series_id
                        new_row['volume_path'] = str(volume_aug_path)
                        new_row['mask_path'] = str(mask_aug_path) if mask is not None else None
                        new_row['metadata_path'] = str(metadata_aug_path)
                        new_row['has_mask'] = mask is not None
                        
                        # Add augmentation info
                        new_row['is_augmented'] = True
                        new_row['augmentation_type'] = aug_type
                        new_row['original_series_id'] = row['series_id']
                        
                        # Assign to same split by default, or distribute proportionally
                        if preserve_ratios:
                            new_row['split'] = split
                        else:
                            new_row['split'] = split
                        
                        augmented_rows.append(new_row)
                        
                    except Exception as e:
                        print(f"❌ Error augmenting {row['series_id']} with {aug_type}: {e}")
                        continue
                        
            except Exception as e:
                print(f"❌ Error loading {row['series_id']}: {e}")
                continue
    
    # Create augmented dataset
    df_augmented = pd.DataFrame(augmented_rows)
    df_combined = pd.concat([df_original, df_augmented], ignore_index=True)
    
    # Add is_augmented column to original data
    df_original['is_augmented'] = False
    df_combined.loc[:len(df_original)-1, 'is_augmented'] = False
    
    print(f"\n📈 Dataset expansion summary:")
    print(f"Original samples: {len(df_original)}")
    print(f"Augmented samples: {len(df_augmented)}")
    print(f"Total samples: {len(df_combined)}")
    print(f"Expansion factor: {len(df_combined) / len(df_original):.1f}x")
    
    # Print split distribution
    print(f"\n📊 Split distribution:")
    for split in ['train', 'val', 'test']:
        original_count = len(df_original[df_original['split'] == split])
        augmented_count = len(df_augmented[df_augmented['split'] == split])
        total_count = len(df_combined[df_combined['split'] == split])
        print(f"{split}: {original_count} → {total_count} (+{augmented_count})")
    
    # Print class distribution
    print(f"\n🏷️ Class distribution (total):")
    for label in sorted(df_combined['label'].unique()):
        original_count = len(df_original[df_original['label'] == label])
        total_count = len(df_combined[df_combined['label'] == label])
        augmented_count = total_count - original_count
        label_name = df_combined[df_combined['label'] == label]['label_name'].iloc[0]
        print(f"Class {label} ({label_name}): {original_count} → {total_count} (+{augmented_count})")
    
    # Save augmented dataset
    backup_path = dataset_index_path.with_suffix('.csv.backup')
    df_original.to_csv(backup_path, index=False)
    print(f"💾 Original dataset backed up to: {backup_path}")
    
    augmented_index_path = dataset_index_path.with_name("dataset_index_with_splits_augmented.csv")
    df_combined.to_csv(augmented_index_path, index=False)
    print(f"💾 Augmented dataset saved to: {augmented_index_path}")
    
    # Optionally replace original
    replace_original = input("\n🔄 Replace original dataset index with augmented version? (y/N): ").strip().lower()
    if replace_original == 'y':
        df_combined.to_csv(dataset_index_path, index=False)
        print(f"✅ Original dataset index updated with augmented data")
    else:
        print(f"ℹ️ Original dataset unchanged. Use augmented dataset with:")
        print(f"   --dataset-index {augmented_index_path}")

if __name__ == "__main__":
    main()