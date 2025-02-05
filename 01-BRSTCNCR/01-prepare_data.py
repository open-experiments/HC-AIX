#!/usr/bin/env python3

import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MammogramPreprocessor:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.tar_path = self.base_dir / "images.tar.xz"
        self.image_dir = self.base_dir / "images"
        self.info_path = self.image_dir / "Info.txt"
        self.dataset_dir = self.base_dir / "processed_data"
        self.malignant_dir = self.dataset_dir / "malignant"
        self.non_malignant_dir = self.dataset_dir / "non_malignant"
        self.splits_path = self.base_dir / "dataset_splits.json"

    def extract_tarfile(self) -> None:
        """Extract the tar.xz archive containing images."""
        if not self.tar_path.exists():
            raise FileNotFoundError(f"Archive not found: {self.tar_path}")
        
        logger.info(f"Extracting {self.tar_path}...")
        with tarfile.open(self.tar_path, 'r:xz') as tar:
            tar.extractall(path=self.base_dir)
        logger.info("Extraction complete")

    def setup_directories(self) -> None:
        """Create necessary directories."""
        self.malignant_dir.mkdir(parents=True, exist_ok=True)
        self.non_malignant_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directories at {self.dataset_dir}")

    def parse_info_file(self) -> Dict[str, int]:
        """Parse Info.txt to determine image labels."""
        file_to_label = defaultdict(int)
        
        with open(self.info_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("mdb"):
                    continue
                    
                parts = line.split()
                mdb_ref = parts[0]
                abnormality = parts[2]
                
                if abnormality == "NORM":
                    label = 0
                else:
                    severity = parts[3]
                    label = 1 if severity == "M" else 0
                    
                file_to_label[mdb_ref] = max(file_to_label[mdb_ref], label)
        
        logger.info(f"Found {sum(file_to_label.values())} malignant and "
                   f"{len(file_to_label) - sum(file_to_label.values())} non-malignant cases")
        return dict(file_to_label)

    def copy_images(self, file_to_label: Dict[str, int]) -> Tuple[List[str], List[str]]:
        """Copy images to appropriate directories based on labels."""
        malignant_files = []
        non_malignant_files = []
        
        for mdb_ref, label in file_to_label.items():
            src = self.image_dir / f"{mdb_ref}.pgm"
            if not src.is_file():
                logger.warning(f"File not found: {src}")
                continue
                
            if label == 1:
                dest = self.malignant_dir / f"{mdb_ref}.pgm"
                malignant_files.append(str(dest))
            else:
                dest = self.non_malignant_dir / f"{mdb_ref}.pgm"
                non_malignant_files.append(str(dest))
                
            shutil.copy(src, dest)
        
        logger.info(f"Copied {len(malignant_files)} malignant and "
                   f"{len(non_malignant_files)} non-malignant images")
        return malignant_files, non_malignant_files

    def create_dataset_splits(self, malignant_files: List[str], 
                            non_malignant_files: List[str],
                            ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> None:
        """Create train/val/test splits maintaining class balance."""
        assert sum(ratios) == 1.0, "Split ratios must sum to 1.0"
        train_ratio, val_ratio, test_ratio = ratios
        
        n_malignant = len(malignant_files)
        train_size = int(n_malignant * train_ratio)
        val_size = int(n_malignant * val_ratio)
        test_size = n_malignant - train_size - val_size
        
        splits = {
            'train': {
                'files': (malignant_files[:train_size] + 
                         non_malignant_files[:train_size]),
                'labels': ([1] * train_size + [0] * train_size)
            },
            'validation': {
                'files': (malignant_files[train_size:train_size+val_size] +
                         non_malignant_files[train_size:train_size+val_size]),
                'labels': ([1] * val_size + [0] * val_size)
            },
            'test': {
                'files': (malignant_files[train_size+val_size:] +
                         non_malignant_files[train_size+val_size:train_size+val_size+test_size]),
                'labels': ([1] * test_size + [0] * test_size)
            }
        }
        
        with open(self.splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Created dataset splits at {self.splits_path}")
        for split, data in splits.items():
            logger.info(f"{split}: {len(data['files'])} images "
                       f"({sum(data['labels'])} malignant)")

    def process(self) -> None:
        """Run full preprocessing pipeline."""
        self.extract_tarfile()
        self.setup_directories()
        file_to_label = self.parse_info_file()
        malignant_files, non_malignant_files = self.copy_images(file_to_label)
        self.create_dataset_splits(malignant_files, non_malignant_files)

def main():
    preprocessor = MammogramPreprocessor()
    preprocessor.process()

if __name__ == "__main__":
    main()
