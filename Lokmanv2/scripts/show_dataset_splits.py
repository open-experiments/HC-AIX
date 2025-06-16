#!/usr/bin/env python3
"""
Show Dataset Splits with Labels
Displays which data samples are in each split and their corresponding labels
"""

import json
import pandas as pd
from pathlib import Path

def show_dataset_splits(data_root="data"):
    """Show detailed breakdown of dataset splits with labels"""
    
    data_path = Path(data_root)
    
    # Load the dataset index
    dataset_file = data_path / "processed" / "dataset_index_with_splits.csv"
    if not dataset_file.exists():
        print(f"❌ Dataset index not found: {dataset_file}")
        return
    
    df = pd.read_csv(dataset_file)
    
    # Load splits
    splits_dir = data_path / "splits"
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f"{split_name}.json"
        if split_file.exists():
            with open(split_file) as f:
                splits[split_name] = json.load(f)
        else:
            splits[split_name] = []
    
    # Load summary
    summary_file = splits_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    else:
        summary = {}
    
    print("🏥 Lokman-v2 Dataset Splits Analysis")
    print("=" * 60)
    
    # Overall summary
    print(f"📊 Overall Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   Train: {len(splits['train'])}")
    print(f"   Validation: {len(splits['val'])}")
    print(f"   Test: {len(splits['test'])}")
    print()
    
    # Label mapping
    label_names = {0: 'Normal', 1: 'Benign', 2: 'Malignant', 3: 'Tumor'}
    
    # Show detailed breakdown for each split
    for split_name, series_ids in splits.items():
        if not series_ids:
            continue
            
        print(f"📋 {split_name.upper()} SET ({len(series_ids)} samples)")
        print("-" * 40)
        
        # Get data for this split
        split_data = df[df['series_id'].isin(series_ids)].copy()
        
        if len(split_data) == 0:
            print(f"   ⚠️  No data found for {split_name} split")
            continue
        
        # Label distribution
        label_counts = split_data['label'].value_counts().sort_index()
        print(f"   Label Distribution:")
        for label, count in label_counts.items():
            label_name = label_names.get(label, f"Unknown-{label}")
            percentage = (count / len(split_data)) * 100
            print(f"      {label} ({label_name}): {count} samples ({percentage:.1f}%)")
        print()
        
        # Show samples by label
        for label in sorted(split_data['label'].unique()):
            label_name = label_names.get(label, f"Unknown-{label}")
            label_samples = split_data[split_data['label'] == label]
            
            print(f"   🏷️  {label_name} Samples ({len(label_samples)}):")
            for _, row in label_samples.iterrows():
                patient = row['patient_dir']
                study = row['study_dir']
                series = row['series_dir']
                scan_type = row['scan_type']
                contrast = "with contrast" if row['has_contrast'] else "no contrast"
                
                print(f"      • {row['series_id'][:12]}... | {scan_type} | {contrast}")
                print(f"        Patient: {patient[:30]}...")
                print(f"        Study: {study}")
                print(f"        Series: {series[:50]}...")
                print()
        
        print()
    
    # Cross-validation check
    print("🔍 Data Integrity Check:")
    print("-" * 40)
    
    all_split_ids = set()
    for split_name, series_ids in splits.items():
        all_split_ids.update(series_ids)
    
    dataset_ids = set(df['series_id'].tolist())
    
    print(f"   Total unique IDs in splits: {len(all_split_ids)}")
    print(f"   Total IDs in dataset: {len(dataset_ids)}")
    
    missing_from_splits = dataset_ids - all_split_ids
    missing_from_dataset = all_split_ids - dataset_ids
    
    if missing_from_splits:
        print(f"   ⚠️  IDs in dataset but not in splits: {len(missing_from_splits)}")
        for missing_id in missing_from_splits:
            print(f"      - {missing_id}")
    
    if missing_from_dataset:
        print(f"   ⚠️  IDs in splits but not in dataset: {len(missing_from_dataset)}")
        for missing_id in missing_from_dataset:
            print(f"      - {missing_id}")
    
    if not missing_from_splits and not missing_from_dataset:
        print(f"   ✅ All data properly assigned to splits")
    
    print()
    
    # Label distribution across splits
    print("📈 Label Distribution Across Splits:")
    print("-" * 40)
    
    split_label_summary = {}
    for split_name, series_ids in splits.items():
        if series_ids:
            split_data = df[df['series_id'].isin(series_ids)]
            split_label_summary[split_name] = split_data['label'].value_counts().sort_index()
    
    # Create summary table
    print(f"{'Label':<12} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 48)
    
    for label in range(4):
        label_name = label_names[label]
        train_count = split_label_summary.get('train', {}).get(label, 0)
        val_count = split_label_summary.get('val', {}).get(label, 0)
        test_count = split_label_summary.get('test', {}).get(label, 0)
        total_count = train_count + val_count + test_count
        
        print(f"{label} ({label_name:<8}) {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}")
    
    print()
    print("✅ Dataset analysis complete!")

if __name__ == "__main__":
    show_dataset_splits()