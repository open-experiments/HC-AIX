"""
Advanced inference engine for CT-based breast cancer metastasis detection.
Supports real-time prediction with uncertainty quantification and visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import time
from scipy.ndimage import zoom, gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import io
import base64

from utils.dicom_utils import DICOMProcessor
from core.model import AttentionUNet3D

logger = logging.getLogger(__name__)

class CTInferenceEngine:
    """Advanced inference engine for CT breast cancer detection"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration
            device: Device to run inference on
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                'model': {
                    'in_channels': 1,
                    'out_channels': 4,
                    'base_channels': 32,
                    'depth': 5,
                    'use_attention': True,
                    'use_deep_supervision': True,
                    'dropout_rate': 0.1,
                    'normalization': 'batch',
                    'activation': 'ReLU'
                },
                'data': {
                    'target_spacing': [1.0, 1.0, 1.0],
                    'target_size': [128, 128, 128],
                    'intensity_range': [-1000, 1000]
                },
                'inference': {
                    'confidence_threshold': 0.5,
                    'use_tta': True,
                    'tta_transforms': 8
                }
            }
        
        # Initialize DICOM processor
        self.processor = DICOMProcessor(
            target_spacing=tuple(self.config['data']['target_spacing']),
            target_size=tuple(self.config['data']['target_size']),
            intensity_range=tuple(self.config['data']['intensity_range'])
        )
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Class names and colors
        self.class_names = ["Background", "Normal", "Malignant", "Tumor"]
        self.class_colors = {
            0: [0, 0, 0, 0],        # Background - transparent
            1: [0, 255, 0, 128],    # Normal - green
            2: [255, 165, 0, 180],  # Malignant - orange  
            3: [255, 0, 0, 200]     # Tumor - red
        }
        
        logger.info(f"Inference engine initialized on {self.device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint"""
        # Create model from config
        from types import SimpleNamespace
        config_obj = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v 
                                       for k, v in self.config.items()})
        
        model = AttentionUNet3D(
            in_channels=self.config.get('model', {}).get('in_channels', 1),
            out_channels=self.config.get('model', {}).get('out_channels', 4),
            base_channels=self.config.get('model', {}).get('base_channels', 32),
            num_levels=self.config.get('model', {}).get('num_levels', 4),
            use_attention=self.config.get('model', {}).get('use_attention', True),
            use_deep_supervision=self.config.get('model', {}).get('use_deep_supervision', True)
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def predict_dicom_series(self, dicom_dir: str, 
                           return_probabilities: bool = False,
                           return_attention: bool = False) -> Dict:
        """
        Predict on DICOM series
        
        Args:
            dicom_dir: Directory containing DICOM files
            return_probabilities: Whether to return probability maps
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary with predictions and metadata
        """
        start_time = time.time()
        
        try:
            # Load and preprocess DICOM
            ct_image, metadata = self.processor.load_dicom_series(dicom_dir)
            volume = self.processor.preprocess_ct_volume(ct_image, metadata)
            
            # Convert to tensor
            volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            volume_tensor = volume_tensor.to(self.device)
            
            # Predict
            prediction_result = self._predict_volume(
                volume_tensor, 
                return_probabilities=return_probabilities,
                return_attention=return_attention
            )
            
            # Add metadata
            prediction_result.update({
                'original_shape': volume.shape,
                'original_spacing': ct_image.GetSpacing(),
                'original_origin': ct_image.GetOrigin(),
                'patient_metadata': metadata,
                'processing_time': time.time() - start_time
            })
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in DICOM prediction: {e}")
            raise
    
    def predict_volume_array(self, volume: np.ndarray,
                           return_probabilities: bool = False,
                           return_attention: bool = False) -> Dict:
        """
        Predict on preprocessed volume array
        
        Args:
            volume: Preprocessed volume array (D, H, W)
            return_probabilities: Whether to return probability maps
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary with predictions
        """
        start_time = time.time()
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        volume_tensor = volume_tensor.to(self.device)
        
        # Predict
        prediction_result = self._predict_volume(
            volume_tensor,
            return_probabilities=return_probabilities,
            return_attention=return_attention
        )
        
        prediction_result['processing_time'] = time.time() - start_time
        
        return prediction_result
    
    def _predict_volume(self, volume_tensor: torch.Tensor,
                       return_probabilities: bool = False,
                       return_attention: bool = False) -> Dict:
        """Core prediction function"""
        
        with torch.no_grad():
            # Test time augmentation if enabled
            if self.config['inference']['use_tta']:
                predictions, probabilities = self._predict_with_tta(volume_tensor)
            else:
                outputs = self.model(volume_tensor)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]  # Use main output
                
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
            
            # Extract attention maps if requested
            attention_maps = None
            if return_attention and hasattr(self.model, 'get_attention_maps'):
                _, attention_maps = self.model.get_attention_maps(volume_tensor)
                # Convert to numpy
                attention_maps = {k: v.cpu().numpy() for k, v in attention_maps.items()}
            
            # Convert to numpy
            predictions_np = predictions.squeeze().cpu().numpy()
            probabilities_np = probabilities.squeeze().cpu().numpy() if return_probabilities else None
            
            # Calculate confidence and uncertainty
            confidence_map = self._calculate_confidence(probabilities.squeeze())
            uncertainty_map = self._calculate_uncertainty(probabilities.squeeze())
            
            # Extract clinical metrics
            clinical_analysis = self._analyze_predictions(predictions_np, probabilities_np)
            
            result = {
                'predictions': predictions_np,
                'confidence_map': confidence_map.cpu().numpy(),
                'uncertainty_map': uncertainty_map.cpu().numpy(),
                'clinical_analysis': clinical_analysis
            }
            
            if return_probabilities:
                result['probabilities'] = probabilities_np
            
            if attention_maps:
                result['attention_maps'] = attention_maps
            
            return result
    
    def _predict_with_tta(self, volume_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Test time augmentation prediction"""
        
        all_predictions = []
        all_probabilities = []
        
        # Original prediction
        outputs = self.model(volume_tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        
        probabilities = F.softmax(outputs, dim=1)
        all_probabilities.append(probabilities)
        
        # Augmented predictions
        tta_transforms = [
            lambda x: torch.flip(x, dims=[2]),  # Flip depth
            lambda x: torch.flip(x, dims=[3]),  # Flip height
            lambda x: torch.flip(x, dims=[4]),  # Flip width
            lambda x: torch.flip(x, dims=[2, 3]),  # Flip depth + height
            lambda x: torch.flip(x, dims=[2, 4]),  # Flip depth + width
            lambda x: torch.flip(x, dims=[3, 4]),  # Flip height + width
            lambda x: torch.flip(x, dims=[2, 3, 4])  # Flip all
        ]
        
        reverse_transforms = [
            lambda x: torch.flip(x, dims=[2]),
            lambda x: torch.flip(x, dims=[3]),
            lambda x: torch.flip(x, dims=[4]),
            lambda x: torch.flip(x, dims=[2, 3]),
            lambda x: torch.flip(x, dims=[2, 4]),
            lambda x: torch.flip(x, dims=[3, 4]),
            lambda x: torch.flip(x, dims=[2, 3, 4])
        ]
        
        num_tta = min(len(tta_transforms), self.config['inference'].get('tta_transforms', 4))
        
        for i in range(num_tta):
            # Apply transform
            transformed_volume = tta_transforms[i](volume_tensor)
            
            # Predict
            outputs = self.model(transformed_volume)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            probabilities = F.softmax(outputs, dim=1)
            
            # Reverse transform
            probabilities = reverse_transforms[i](probabilities)
            all_probabilities.append(probabilities)
        
        # Average probabilities
        avg_probabilities = torch.stack(all_probabilities).mean(dim=0)
        avg_predictions = torch.argmax(avg_probabilities, dim=1)
        
        return avg_predictions, avg_probabilities
    
    def _calculate_confidence(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Calculate prediction confidence (max probability)"""
        return torch.max(probabilities, dim=0)[0]
    
    def _calculate_uncertainty(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Calculate prediction uncertainty (entropy)"""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = -torch.sum(probabilities * torch.log(probabilities + eps), dim=0)
        # Normalize by max entropy
        max_entropy = torch.log(torch.tensor(probabilities.shape[0], dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    
    def _analyze_predictions(self, predictions: np.ndarray, 
                           probabilities: Optional[np.ndarray] = None) -> Dict:
        """Analyze predictions for clinical insights"""
        
        analysis = {}
        
        # Volume analysis
        total_voxels = predictions.size
        class_volumes = {}
        class_percentages = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            volume = (predictions == class_idx).sum()
            percentage = (volume / total_voxels) * 100
            
            class_volumes[class_name] = int(volume)
            class_percentages[class_name] = float(percentage)
        
        analysis['volume_analysis'] = {
            'total_voxels': int(total_voxels),
            'class_volumes': class_volumes,
            'class_percentages': class_percentages
        }
        
        # Clinical risk assessment
        malignant_percentage = class_percentages.get('Malignant', 0)
        tumor_percentage = class_percentages.get('Tumor', 0)
        total_abnormal = malignant_percentage + tumor_percentage
        
        if total_abnormal > 5.0:
            risk_level = "High"
            risk_description = "Significant malignant/tumor tissue detected"
        elif total_abnormal > 1.0:
            risk_level = "Moderate"
            risk_description = "Some suspicious tissue detected"
        elif total_abnormal > 0.1:
            risk_level = "Low"
            risk_description = "Minimal suspicious findings"
        else:
            risk_level = "Very Low"
            risk_description = "No significant abnormalities detected"
        
        analysis['risk_assessment'] = {
            'risk_level': risk_level,
            'risk_description': risk_description,
            'abnormal_tissue_percentage': float(total_abnormal)
        }
        
        # Confidence analysis
        if probabilities is not None:
            max_probs = np.max(probabilities, axis=0)
            avg_confidence = float(np.mean(max_probs))
            min_confidence = float(np.min(max_probs))
            
            analysis['confidence_analysis'] = {
                'average_confidence': avg_confidence,
                'minimum_confidence': min_confidence,
                'high_confidence_percentage': float((max_probs > 0.8).sum() / max_probs.size * 100)
            }
        
        return analysis
    
    def create_visualization(self, volume: np.ndarray, predictions: np.ndarray,
                           confidence_map: Optional[np.ndarray] = None,
                           slice_indices: Optional[List[int]] = None) -> Dict[str, str]:
        """
        Create visualization of predictions
        
        Args:
            volume: Original CT volume
            predictions: Prediction mask
            confidence_map: Confidence map
            slice_indices: Specific slices to visualize
            
        Returns:
            Dictionary with base64 encoded images
        """
        
        if slice_indices is None:
            # Select representative slices
            depth = volume.shape[0]
            slice_indices = [depth//4, depth//2, 3*depth//4]
        
        visualizations = {}
        
        for i, slice_idx in enumerate(slice_indices):
            if slice_idx >= volume.shape[0]:
                continue
            
            # Create figure
            fig, axes = plt.subplots(1, 3 if confidence_map is not None else 2, 
                                   figsize=(15 if confidence_map is not None else 10, 5))
            
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            
            # Original slice
            vol_slice = volume[slice_idx]
            axes[0].imshow(vol_slice, cmap='gray', alpha=1.0)
            axes[0].set_title(f'Original CT - Slice {slice_idx}')
            axes[0].axis('off')
            
            # Prediction overlay
            pred_slice = predictions[slice_idx]
            axes[1].imshow(vol_slice, cmap='gray', alpha=0.7)
            
            # Create colored overlay for predictions
            overlay = np.zeros((*pred_slice.shape, 4))
            for class_idx in range(len(self.class_names)):
                mask = pred_slice == class_idx
                if mask.any():
                    color = np.array(self.class_colors[class_idx]) / 255.0
                    overlay[mask] = color
            
            axes[1].imshow(overlay)
            axes[1].set_title('Prediction Overlay')
            axes[1].axis('off')
            
            # Confidence map if available
            if confidence_map is not None and len(axes) > 2:
                conf_slice = confidence_map[slice_idx]
                im = axes[2].imshow(conf_slice, cmap='viridis', vmin=0, vmax=1)
                axes[2].set_title('Confidence Map')
                axes[2].axis('off')
                plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            visualizations[f'slice_{slice_idx}'] = image_base64
        
        return visualizations
    
    def create_3d_summary(self, predictions: np.ndarray) -> Dict[str, str]:
        """Create 3D summary visualizations"""
        
        # Maximum intensity projection for each axis
        projections = {
            'axial': np.max(predictions, axis=0),      # Z projection
            'coronal': np.max(predictions, axis=1),    # Y projection  
            'sagittal': np.max(predictions, axis=2)    # X projection
        }
        
        visualizations = {}
        
        for view_name, projection in projections.items():
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # Create colored projection
            overlay = np.zeros((*projection.shape, 4))
            for class_idx in range(len(self.class_names)):
                mask = projection == class_idx
                if mask.any():
                    color = np.array(self.class_colors[class_idx]) / 255.0
                    overlay[mask] = color
            
            ax.imshow(overlay)
            ax.set_title(f'Max Projection - {view_name.capitalize()} View')
            ax.axis('off')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            visualizations[f'projection_{view_name}'] = image_base64
        
        return visualizations

class BatchInferenceEngine:
    """Batch inference for multiple studies"""
    
    def __init__(self, inference_engine: CTInferenceEngine):
        self.inference_engine = inference_engine
    
    def process_multiple_studies(self, study_paths: List[str]) -> List[Dict]:
        """Process multiple studies in batch"""
        
        results = []
        
        for study_path in study_paths:
            try:
                result = self.inference_engine.predict_dicom_series(
                    study_path,
                    return_probabilities=True,
                    return_attention=False
                )
                
                result['study_path'] = study_path
                result['status'] = 'success'
                
            except Exception as e:
                result = {
                    'study_path': study_path,
                    'status': 'failed',
                    'error': str(e)
                }
                logger.error(f"Failed to process {study_path}: {e}")
            
            results.append(result)
        
        return results

# Utility functions for web interface
def prepare_inference_response(prediction_result: Dict, volume: np.ndarray) -> Dict:
    """Prepare inference result for web interface"""
    
    # Create visualizations
    engine = prediction_result.get('_engine')  # If engine is passed
    if engine:
        visualizations = engine.create_visualization(
            volume, prediction_result['predictions'], 
            prediction_result.get('confidence_map')
        )
        
        projections = engine.create_3d_summary(prediction_result['predictions'])
        
        visualizations.update(projections)
    else:
        visualizations = {}
    
    # Prepare response
    response = {
        'status': 'success',
        'clinical_analysis': prediction_result['clinical_analysis'],
        'visualizations': visualizations,
        'metadata': {
            'processing_time': prediction_result.get('processing_time', 0),
            'model_confidence': prediction_result['clinical_analysis'].get('confidence_analysis', {}),
            'volume_shape': prediction_result.get('original_shape', volume.shape)
        }
    }
    
    return response

# Test function
if __name__ == "__main__":
    # Test inference engine
    print("Testing CT Inference Engine...")
    
    # This would be run with actual model and data
    # engine = CTInferenceEngine("path/to/model.pth")
    # result = engine.predict_dicom_series("path/to/dicom/series")
    
    print("Inference engine implementation completed!")