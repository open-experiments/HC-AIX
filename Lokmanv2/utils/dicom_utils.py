"""
DICOM processing utilities for CT-based breast cancer detection.
Handles DICOM loading, preprocessing, and conversion for 3D analysis.
"""

import os
import pydicom
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)

class DICOMProcessor:
    """Advanced DICOM processing for CT breast cancer analysis"""
    
    def __init__(self, 
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_size: Optional[Tuple[int, int, int]] = None,
                 intensity_range: Tuple[int, int] = (-1000, 1000)):
        """
        Initialize DICOM processor
        
        Args:
            target_spacing: Target voxel spacing in mm (z, y, x)
            target_size: Target volume size (D, H, W)
            intensity_range: HU intensity range for CT normalization
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.intensity_range = intensity_range
        
    def load_dicom_series(self, dicom_dir: str) -> sitk.Image:
        """
        Load a complete DICOM series from directory
        
        Args:
            dicom_dir: Directory containing DICOM files
            
        Returns:
            SimpleITK image object
        """
        try:
            # Read DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
            
            if not dicom_names:
                raise ValueError(f"No DICOM series found in {dicom_dir}")
            
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            
            # Get metadata from first slice
            metadata = self._extract_metadata(dicom_names[0])
            
            return image, metadata
            
        except Exception as e:
            logger.error(f"Error loading DICOM series from {dicom_dir}: {e}")
            raise
    
    def _extract_metadata(self, dicom_file: str) -> Dict[str, Any]:
        """Extract relevant metadata from DICOM file"""
        try:
            ds = pydicom.dcmread(dicom_file)
            
            metadata = {
                'PatientID': getattr(ds, 'PatientID', 'Unknown'),
                'StudyDescription': getattr(ds, 'StudyDescription', 'Unknown'),
                'SeriesDescription': getattr(ds, 'SeriesDescription', 'Unknown'),
                'Modality': getattr(ds, 'Modality', 'Unknown'),
                'SliceThickness': float(getattr(ds, 'SliceThickness', 1.0)),
                'PixelSpacing': [float(x) for x in getattr(ds, 'PixelSpacing', [1.0, 1.0])],
                'ImageOrientationPatient': [float(x) for x in getattr(ds, 'ImageOrientationPatient', [1,0,0,0,1,0])],
                'ImagePositionPatient': [float(x) for x in getattr(ds, 'ImagePositionPatient', [0,0,0])],
                'RescaleIntercept': float(getattr(ds, 'RescaleIntercept', 0)),
                'RescaleSlope': float(getattr(ds, 'RescaleSlope', 1)),
                'WindowCenter': getattr(ds, 'WindowCenter', None),
                'WindowWidth': getattr(ds, 'WindowWidth', None),
            }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract metadata from {dicom_file}: {e}")
            return {}
    
    def preprocess_ct_volume(self, image: sitk.Image, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess CT volume for deep learning
        
        Args:
            image: SimpleITK image
            metadata: DICOM metadata
            
        Returns:
            Preprocessed numpy array
        """
        # Convert to numpy array
        volume = sitk.GetArrayFromImage(image)  # Shape: (Z, Y, X)
        
        # Apply HU conversion if needed
        volume = self._apply_hu_conversion(volume, metadata)
        
        # Resample to target spacing
        if self.target_spacing:
            volume = self._resample_volume(image, volume)
        
        # Normalize intensity
        volume = self._normalize_intensity(volume)
        
        # Resize to target size if specified
        if self.target_size:
            volume = self._resize_volume(volume, self.target_size)
        
        # Ensure correct data type
        volume = volume.astype(np.float32)
        
        return volume
    
    def _apply_hu_conversion(self, volume: np.ndarray, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """Apply Hounsfield Unit conversion"""
        intercept = metadata.get('RescaleIntercept', 0)
        slope = metadata.get('RescaleSlope', 1)
        
        # Convert to HU
        volume = volume * slope + intercept
        
        return volume
    
    def _resample_volume(self, image: sitk.Image, volume: np.ndarray) -> np.ndarray:
        """Resample volume to target spacing"""
        original_spacing = image.GetSpacing()[::-1]  # (Z, Y, X)
        original_size = image.GetSize()[::-1]  # (Z, Y, X)
        
        # Calculate new size based on target spacing
        new_size = [
            int(round(original_size[i] * original_spacing[i] / self.target_spacing[i]))
            for i in range(3)
        ]
        
        # Resample using SimpleITK
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing[::-1])  # (X, Y, Z)
        resampler.SetSize(new_size[::-1])  # (X, Y, Z)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        resampled_image = resampler.Execute(image)
        resampled_volume = sitk.GetArrayFromImage(resampled_image)
        
        return resampled_volume
    
    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normalize CT intensity values"""
        # Clip to intensity range
        volume = np.clip(volume, self.intensity_range[0], self.intensity_range[1])
        
        # Normalize to [0, 1]
        volume = (volume - self.intensity_range[0]) / (self.intensity_range[1] - self.intensity_range[0])
        
        return volume
    
    def _resize_volume(self, volume: np.ndarray, 
                      target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize volume to target size"""
        from scipy.ndimage import zoom
        
        current_size = volume.shape
        zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
        
        # Use order=1 for linear interpolation
        resized_volume = zoom(volume, zoom_factors, order=1)
        
        return resized_volume
    
    def load_segmentation_mask(self, seg_file: str, 
                              reference_image: sitk.Image) -> np.ndarray:
        """
        Load segmentation mask and align with reference image
        
        Args:
            seg_file: Path to segmentation file
            reference_image: Reference CT image
            
        Returns:
            Segmentation mask as numpy array
        """
        try:
            # Load segmentation
            seg_image = sitk.ReadImage(seg_file)
            
            # Resample to match reference image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(reference_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Preserve labels
            
            aligned_seg = resampler.Execute(seg_image)
            mask = sitk.GetArrayFromImage(aligned_seg)
            
            return mask.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error loading segmentation mask from {seg_file}: {e}")
            raise

class DICOMDatasetBuilder:
    """Build dataset from DICOM files with segmentation"""
    
    def __init__(self, data_root: str, processor: DICOMProcessor):
        """
        Initialize dataset builder
        
        Args:
            data_root: Root directory containing patient data
            processor: DICOM processor instance
        """
        self.data_root = Path(data_root)
        self.processor = processor
        
    def scan_dataset(self) -> pd.DataFrame:
        """
        Scan dataset and create index of all studies
        
        Returns:
            DataFrame with dataset information
        """
        studies = []
        
        # Scan all patient directories
        for patient_dir in self.data_root.iterdir():
            if not patient_dir.is_dir():
                continue
                
            patient_id = patient_dir.name
            
            # Scan study directories
            for study_dir in patient_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                    
                study_id = study_dir.name
                
                # Find DICOM series and segmentations
                dicom_dirs = []
                seg_files = []
                
                for item in study_dir.iterdir():
                    if item.is_dir():
                        # Check if directory contains DICOM files
                        dicom_files = list(item.glob("*.dcm"))
                        if dicom_files:
                            dicom_dirs.append(item)
                    elif item.suffix.lower() == '.dcm' and 'seg' in item.name.lower():
                        seg_files.append(item)
                
                # Create entry for each DICOM series
                for dicom_dir in dicom_dirs:
                    studies.append({
                        'patient_id': patient_id,
                        'study_id': study_id,
                        'series_name': dicom_dir.name,
                        'dicom_path': str(dicom_dir),
                        'seg_files': [str(f) for f in seg_files],
                        'num_seg_files': len(seg_files)
                    })
        
        dataset_df = pd.DataFrame(studies)
        logger.info(f"Found {len(dataset_df)} DICOM series from {dataset_df['patient_id'].nunique()} patients")
        
        return dataset_df
    
    def process_study(self, study_info: Dict[str, Any], 
                     output_dir: str) -> Dict[str, Any]:
        """
        Process single study (CT volume + segmentation)
        
        Args:
            study_info: Study information dictionary
            output_dir: Output directory for processed data
            
        Returns:
            Processing results
        """
        try:
            # Load CT volume
            ct_image, metadata = self.processor.load_dicom_series(study_info['dicom_path'])
            ct_volume = self.processor.preprocess_ct_volume(ct_image, metadata)
            
            # Load segmentation masks if available
            masks = []
            if study_info['seg_files']:
                for seg_file in study_info['seg_files']:
                    try:
                        mask = self.processor.load_segmentation_mask(seg_file, ct_image)
                        # Resize mask to match processed volume
                        if self.processor.target_size:
                            mask = self.processor._resize_volume(
                                mask.astype(np.float32), 
                                self.processor.target_size
                            ).astype(np.uint8)
                        masks.append(mask)
                    except Exception as e:
                        logger.warning(f"Could not load segmentation {seg_file}: {e}")
            
            # Combine multiple segmentation masks
            combined_mask = self._combine_segmentation_masks(masks)
            
            # Save processed data
            output_path = Path(output_dir) / f"{study_info['patient_id']}_{study_info['study_id']}_{study_info['series_name']}"
            output_path.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path / "volume.npy", ct_volume)
            if combined_mask is not None:
                np.save(output_path / "mask.npy", combined_mask)
            
            # Save metadata
            import json
            with open(output_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            result = {
                'success': True,
                'output_path': str(output_path),
                'volume_shape': ct_volume.shape,
                'has_segmentation': combined_mask is not None,
                'metadata': metadata
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing study {study_info}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _combine_segmentation_masks(self, masks: List[np.ndarray]) -> Optional[np.ndarray]:
        """Combine multiple segmentation masks into single multi-class mask"""
        if not masks:
            return None
        
        if len(masks) == 1:
            return masks[0]
        
        # For multiple masks, create combined mask
        # Assuming: 0=background, 1=normal, 2=malignant, 3=tumor
        combined = np.zeros_like(masks[0])
        
        for i, mask in enumerate(masks):
            # Simple combination strategy - can be improved based on actual labels
            combined = np.maximum(combined, mask * (i + 1))
        
        return combined
    
    def build_dataset(self, output_dir: str, max_workers: int = 4) -> pd.DataFrame:
        """
        Build complete dataset by processing all studies
        
        Args:
            output_dir: Output directory for processed data
            max_workers: Maximum number of worker threads
            
        Returns:
            DataFrame with processing results
        """
        # Scan dataset
        dataset_df = self.scan_dataset()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process studies in parallel
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for _, study_info in dataset_df.iterrows():
                future = executor.submit(self.process_study, study_info.to_dict(), output_dir)
                futures.append(future)
            
            for future in tqdm(futures, desc="Processing studies"):
                result = future.result()
                results.append(result)
        
        # Add results to dataframe
        results_df = pd.DataFrame(results)
        final_df = pd.concat([dataset_df.reset_index(drop=True), results_df], axis=1)
        
        # Save dataset index
        final_df.to_csv(Path(output_dir) / "dataset_index.csv", index=False)
        
        successful = final_df['success'].sum()
        total = len(final_df)
        logger.info(f"Successfully processed {successful}/{total} studies")
        
        return final_df