#!/usr/bin/env python3
"""
Data loading and preprocessing utilities for the mammography dataset.
"""

import os
import pandas as pd
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

def read_info_txt(file_path, bg_to_process, class_to_process):
    """
    Read and parse the MIAS dataset information file.
    
    Args:
        file_path: Path to the info.txt file
        bg_to_process: List of background tissue types to include
        class_to_process: List of abnormality classes to include
        
    Returns:
        pd_info: DataFrame containing the image information
        mmi: Dictionary with image reference as key and information as value
    """
    logger.info(f"Reading dataset info from: {file_path}")
    rows = []
    mmi = {}
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip non-data lines
            if not line.startswith("mdb"):
                continue
            
            parts = line.split()
            
            # Prepare a dict for each row
            row_dict = {
                "REF": None,        # e.g. "mdb001"
                "BG": None,         # background tissue: F/G/D
                "CLASS": None,      # abnormality class: CALC/CIRC/SPIC/MISC/ARCH/ASYM/NORM
                "SEVERITY": None,   # B or M, if present
                "X": None,
                "Y": None,
                "RADIUS": None
            }
            
            # Parse the line
            row_dict["REF"] = parts[0]       # Reference ID
            row_dict["BG"] = parts[1]        # Background tissue
            row_dict["CLASS"] = parts[2]     # Abnormality class
            
            # Skip if not in the requested bg_to_process or class_to_process
            if row_dict["BG"] not in bg_to_process:
                continue
            if row_dict["CLASS"] not in class_to_process:
                continue
            
            # If there's exactly 3 parts (e.g., "mdb003 D NORM")
            if len(parts) == 3:
                rows.append(row_dict)
                mmi[row_dict["REF"]] = row_dict
                continue
            
            # If there's a 4th part, it's typically severity (B/M)
            row_dict["SEVERITY"] = parts[3]
            
            if len(parts) == 4:
                rows.append(row_dict)
                mmi[row_dict["REF"]] = row_dict
                continue
            
            # If we have at least 7 parts, we have x,y,radius
            if len(parts) >= 7:
                row_dict["X"] = parts[4]
                row_dict["Y"] = parts[5]
                row_dict["RADIUS"] = parts[6]
            
            mmi[row_dict["REF"]] = row_dict
            rows.append(row_dict)
    
    # Create the DataFrame
    pd_info = pd.DataFrame(rows)
    logger.info(f"Dataset info loaded: {len(rows)} entries")
    return pd_info, mmi


def get_roi_coords(mmi, img):
    """
    Extract coordinates of region of interest from image info.
    
    Args:
        mmi: Dictionary with image information
        img: Image reference ID
        
    Returns:
        Tuple containing (class, severity, x, y, radius)
    """
    if mmi[img]['CLASS'] == 'NORM':
        return mmi[img]['CLASS'], None, None, None, None
    elif mmi[img]['CLASS'] == 'CIRC':
        if mmi[img]['X'] is not None:
            severity = mmi[img]['SEVERITY']
            x = int(mmi[img]['X'])
            y = int(mmi[img]['Y'])
            radius = int(mmi[img]['RADIUS'])
            return mmi[img]['CLASS'], severity, x, y, radius
        else:
            severity = mmi[img]['SEVERITY']
            return mmi[img]['CLASS'], severity, None, None, None
    
    return None, None, None, None, None


def read_labels(mmi, no_angles, angle_interval):
    """
    Create labels for each image and its augmented versions.
    
    Args:
        mmi: Dictionary with image information
        no_angles: Maximum rotation angle
        angle_interval: Rotation step size
        
    Returns:
        Dictionary with image reference as key and labels as values
    """
    logger.info("Reading and processing labels...")
    info = {}
    
    for key, value in mmi.items():
        img = key
        if mmi[img]['CLASS'] == 'NORM':
            info[img] = {angle: 2 for angle in range(0, no_angles, angle_interval)}  # Label "Normal" -> 2
        elif mmi[img]['SEVERITY'] == 'B':
            info[img] = {angle: 0 for angle in range(0, no_angles, angle_interval)}  # Label "Benign" -> 0
        elif mmi[img]['SEVERITY'] == 'M':
            info[img] = {angle: 1 for angle in range(0, no_angles, angle_interval)}  # Label "Malign" -> 1
    
    logger.info(f"Number of labels processed: {len(mmi)}")
    return info


def enhance_image(img):
    """
    Enhance image using CLAHE for better contrast.
    
    Args:
        img: Input image
        
    Returns:
        Enhanced image
    """
    # Apply CLAHE for contrast enhancement
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def crop_roi(img, x_center, y_center, radius, margin_factor=0.5):
    """
    Crop the region of interest (ROI) from the image.
    
    Args:
        img: Input image
        x_center, y_center: Center coordinates of the ROI
        radius: Radius of the ROI
        margin_factor: Factor to extend the ROI (e.g., 0.5 adds 50% margin)
        
    Returns:
        Cropped ROI image
    """
    margin = int(radius * margin_factor)
    x1 = max(x_center - radius - margin, 0)
    y1 = max(y_center - radius - margin, 0)
    x2 = min(x_center + radius + margin, img.shape[1])
    y2 = min(y_center + radius + margin, img.shape[0])
    return img[y1:y2, x1:x2]