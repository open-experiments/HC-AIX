#!/usr/bin/env python3
"""
Model Migration Script
Handles incompatible models from old architecture
"""

import shutil
from pathlib import Path
import argparse
from datetime import datetime

def backup_old_models(models_dir, backup_dir=None):
    """Backup old incompatible models"""
    
    models_dir = Path(models_dir)
    if backup_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = models_dir.parent / f"models_backup_{timestamp}"
    
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(exist_ok=True)
    
    print(f"🔄 Backing up old models from {models_dir} to {backup_dir}")
    
    model_files = list(models_dir.glob("*.pth"))
    checkpoint_files = list(models_dir.glob("checkpoints/*.pth"))
    
    backed_up = []
    
    # Backup main model files
    for model_file in model_files:
        backup_path = backup_dir / model_file.name
        shutil.copy2(model_file, backup_path)
        backed_up.append(str(model_file))
        print(f"   📦 Backed up: {model_file.name}")
    
    # Backup checkpoint files  
    if checkpoint_files:
        checkpoint_backup = backup_dir / "checkpoints"
        checkpoint_backup.mkdir(exist_ok=True)
        
        for checkpoint_file in checkpoint_files:
            backup_path = checkpoint_backup / checkpoint_file.name
            shutil.copy2(checkpoint_file, backup_path)
            backed_up.append(str(checkpoint_file))
            print(f"   📦 Backed up: checkpoints/{checkpoint_file.name}")
    
    return backed_up, backup_dir

def clean_incompatible_models(models_dir, dry_run=False):
    """Remove incompatible model files"""
    
    models_dir = Path(models_dir)
    
    model_files = list(models_dir.glob("*.pth"))
    checkpoint_files = list(models_dir.glob("checkpoints/*.pth"))
    
    all_files = model_files + checkpoint_files
    
    if not all_files:
        print("✅ No model files found to clean")
        return []
    
    if dry_run:
        print(f"🔍 DRY RUN: Would remove {len(all_files)} model files:")
        for f in all_files:
            print(f"   🗑️  Would remove: {f}")
        return []
    
    removed = []
    for model_file in all_files:
        try:
            model_file.unlink()
            removed.append(str(model_file))
            print(f"   🗑️  Removed: {model_file.name}")
        except Exception as e:
            print(f"   ❌ Failed to remove {model_file}: {e}")
    
    return removed

def main():
    parser = argparse.ArgumentParser(description="Migrate incompatible models")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--backup-dir", help="Backup directory (default: auto)")
    parser.add_argument("--clean", action="store_true", help="Clean old models after backup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    print("🏥 Lokman-v2 Model Migration")
    print("=" * 40)
    print(f"📁 Models directory: {models_dir}")
    
    # Check what models exist
    model_files = list(models_dir.glob("*.pth"))
    checkpoint_files = list(models_dir.glob("checkpoints/*.pth"))
    
    if not model_files and not checkpoint_files:
        print("✅ No model files found - nothing to migrate")
        return
    
    print(f"📊 Found {len(model_files)} model files and {len(checkpoint_files)} checkpoints")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be modified")
    
    # Backup models
    if model_files or checkpoint_files:
        if not args.force and not args.dry_run:
            response = input("📦 Backup existing models? (y/n): ")
            if response.lower() != 'y':
                print("❌ Migration cancelled")
                return
        
        if not args.dry_run:
            backed_up, backup_dir = backup_old_models(models_dir, args.backup_dir)
            print(f"✅ Backed up {len(backed_up)} files to {backup_dir}")
    
    # Clean old models
    if args.clean:
        if not args.force and not args.dry_run:
            response = input("🗑️  Remove old incompatible models? (y/n): ")
            if response.lower() != 'y':
                print("❌ Cleaning cancelled")
                return
        
        removed = clean_incompatible_models(models_dir, args.dry_run)
        if not args.dry_run:
            print(f"✅ Removed {len(removed)} incompatible model files")
    
    print("\n🎉 Migration complete!")
    print("💡 You can now train new models with the unified training script:")
    print("   python scripts/train.py --fast-test --model-mode classification")

if __name__ == "__main__":
    main()