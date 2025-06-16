#!/usr/bin/env python3
"""
Data Quality Validation Script for Lokman-v2
Validates that data segmentation and labels are set up properly for accurate predictions
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse

def validate_data_quality(data_root="data"):
    """Comprehensive validation of data quality for model training"""
    
    data_path = Path(data_root)
    
    print("🔍 Lokman-v2 Data Quality Validation")
    print("=" * 50)
    
    # Load dataset index
    dataset_file = data_path / "processed" / "dataset_index_with_splits.csv"
    df = pd.read_csv(dataset_file)
    
    print(f"📊 Dataset Overview:")
    print(f"   Total samples: {len(df)}")
    print(f"   Labels distribution:")
    
    label_counts = df['label'].value_counts().sort_index()
    label_names = {0: 'Normal', 1: 'Benign', 2: 'Malignant', 3: 'Tumor'}
    
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"      {label} ({label_names[label]}): {count} samples ({percentage:.1f}%)")
    
    print()
    
    # Check data quality issues
    issues = []
    warnings = []
    
    # 1. Check for missing Normal class
    if 0 not in label_counts:
        warnings.append("⚠️  No Normal (class 0) samples - model may not learn normal anatomy")
    
    # 2. Check class imbalance
    max_class_count = label_counts.max()
    min_class_count = label_counts.min()
    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
    
    if imbalance_ratio > 3:
        warnings.append(f"⚠️  Significant class imbalance (ratio: {imbalance_ratio:.1f}:1)")
    
    # 3. Validate volume and mask data
    print("🔬 Validating Volume and Mask Data:")
    
    sample_count = min(5, len(df))  # Check first 5 samples
    
    for idx, row in df.head(sample_count).iterrows():
        series_id = row['series_id']
        label = row['label']
        label_name = row['label_name']
        
        print(f"   📁 Sample {idx+1}: {series_id} ({label_name})")
        
        # Load volume
        volume_path = Path(row['volume_path'])
        if not volume_path.exists():
            issues.append(f"❌ Volume file missing: {volume_path}")
            continue
            
        try:
            volume = np.load(volume_path)
            print(f"      Volume shape: {volume.shape}, dtype: {volume.dtype}")
            print(f"      Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
            
            # Check for valid data
            if np.all(volume == 0):
                issues.append(f"❌ Volume is all zeros: {series_id}")
            elif np.any(np.isnan(volume)):
                issues.append(f"❌ Volume contains NaN values: {series_id}")
            elif np.any(np.isinf(volume)):
                issues.append(f"❌ Volume contains infinite values: {series_id}")
                
        except Exception as e:
            issues.append(f"❌ Cannot load volume {series_id}: {e}")
            continue
        
        # Load mask
        mask_path = Path(row['mask_path'])
        if not mask_path.exists():
            issues.append(f"❌ Mask file missing: {mask_path}")
            continue
            
        try:
            mask = np.load(mask_path)
            print(f"      Mask shape: {mask.shape}, dtype: {mask.dtype}")
            
            # Analyze mask content
            unique_values = np.unique(mask)
            print(f"      Mask classes: {unique_values}")
            
            # Check mask quality
            if len(unique_values) == 1 and unique_values[0] == 0:
                warnings.append(f"⚠️  Mask is all background (no segmentation): {series_id}")
            elif len(unique_values) > 4:
                warnings.append(f"⚠️  Mask has unexpected classes: {series_id} - {unique_values}")
            
            # Calculate mask statistics
            total_voxels = mask.size
            background_voxels = np.sum(mask == 0)
            foreground_voxels = total_voxels - background_voxels
            
            if foreground_voxels > 0:
                fg_percentage = (foreground_voxels / total_voxels) * 100
                print(f"      Foreground: {fg_percentage:.1f}% of volume")
                
                # Check each class in mask
                for class_id in unique_values:
                    if class_id > 0:  # Skip background
                        class_voxels = np.sum(mask == class_id)
                        class_percentage = (class_voxels / total_voxels) * 100
                        print(f"         Class {class_id}: {class_percentage:.2f}% of volume")
            else:
                print(f"      No foreground segmentation")
                
        except Exception as e:
            issues.append(f"❌ Cannot load mask {series_id}: {e}")
            continue
        
        print()
    
    # 4. Check label consistency with folder structure
    print("🏷️  Validating Label Consistency:")
    
    for idx, row in df.head(sample_count).iterrows():
        series_id = row['series_id']
        label = row['label']
        study_dir = row['study_dir']
        scan_type = row['scan_type']
        has_contrast = row['has_contrast']
        
        # Check if labeling makes sense
        label_issues = []
        
        if label == 1 and has_contrast:  # Benign but with contrast
            label_issues.append("Benign classification but contrast study (may indicate screening)")
        elif label == 2 and not has_contrast:  # Malignant but no contrast
            label_issues.append("Malignant classification but no contrast (verify if correct)")
        elif label == 3 and not has_contrast:  # Tumor but no contrast
            label_issues.append("Tumor classification but no contrast (verify if correct)")
        
        if label_issues:
            for issue in label_issues:
                warnings.append(f"⚠️  {series_id}: {issue}")
    
    # 5. Check data consistency across splits
    print("📊 Validating Split Quality:")
    
    splits_dir = data_path / 'splits'
    split_files = ['train.json', 'val.json', 'test.json']
    
    for split_file in split_files:
        split_path = splits_dir / split_file
        if split_path.exists():
            with open(split_path) as f:
                split_ids = json.load(f)
            
            split_name = split_file.replace('.json', '')
            split_df = df[df['series_id'].isin(split_ids)]
            split_labels = split_df['label'].value_counts().sort_index()
            
            print(f"   {split_name.upper()} split:")
            for label in range(1, 4):  # Check classes 1, 2, 3
                count = split_labels.get(label, 0)
                print(f"      Class {label} ({label_names[label]}): {count} samples")
                
                if count == 0:
                    warnings.append(f"⚠️  {split_name} split missing class {label} ({label_names[label]})")
    
    print()
    
    # Summary and recommendations
    print("📋 Validation Summary:")
    print("=" * 30)
    
    if not issues and not warnings:
        print("✅ Data quality validation PASSED")
        print("   Your dataset is ready for training!")
    else:
        if issues:
            print("❌ CRITICAL ISSUES FOUND:")
            for issue in issues:
                print(f"   {issue}")
            print()
        
        if warnings:
            print("⚠️  WARNINGS (should be addressed):")
            for warning in warnings:
                print(f"   {warning}")
            print()
    
    # Recommendations
    print("💡 Recommendations for Better Training:")
    
    if 0 not in label_counts:
        print("   • Consider adding Normal/Healthy scans as class 0")
        print("   • This helps the model learn normal anatomy patterns")
    
    if imbalance_ratio > 2:
        print("   • Consider data augmentation for minority classes")
        print("   • Use class weights during training to handle imbalance")
    
    total_samples = len(df)
    if total_samples < 100:
        print(f"   • Small dataset ({total_samples} samples) - consider:")
        print("     - Transfer learning from pre-trained medical models")
        print("     - Extensive data augmentation")
        print("     - Cross-validation for robust evaluation")
    
    print("   • Verify that mask segmentations correspond to actual pathology")
    print("   • Consider expert medical review of challenging cases")
    
    return len(issues) == 0

def main():
    parser = argparse.ArgumentParser(description="Validate data quality for Lokman-v2")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    
    args = parser.parse_args()
    
    is_valid = validate_data_quality(args.data_root)
    
    if not is_valid:
        print("\n❌ Data validation failed! Please fix issues before training.")
        exit(1)
    else:
        print("\n✅ Data validation passed! Ready for training.")

if __name__ == "__main__":
    main()