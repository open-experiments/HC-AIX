"""
Comprehensive metrics for CT-based breast cancer metastasis detection.
Includes medical imaging specific metrics and visualization utilities.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
import logging

logger = logging.getLogger(__name__)

class SegmentationMetrics:
    """Comprehensive metrics for 3D segmentation"""
    
    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Medical imaging specific class names
        if num_classes == 4 and not class_names:
            self.class_names = ["Background", "Normal", "Malignant", "Tumor"]
    
    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor, 
                        class_idx: int, smooth: float = 1e-5) -> float:
        """Calculate Dice coefficient for specific class"""
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()
    
    def iou_score(self, pred: torch.Tensor, target: torch.Tensor, 
                  class_idx: int, smooth: float = 1e-5) -> float:
        """Calculate IoU (Jaccard) score for specific class"""
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    def sensitivity_specificity(self, pred: torch.Tensor, target: torch.Tensor, 
                              class_idx: int) -> Tuple[float, float]:
        """Calculate sensitivity and specificity for specific class"""
        pred_class = (pred == class_idx)
        target_class = (target == class_idx)
        
        # True positives, false positives, true negatives, false negatives
        tp = ((pred_class) & (target_class)).sum().float()
        fp = ((pred_class) & (~target_class)).sum().float()
        tn = ((~pred_class) & (~target_class)).sum().float()
        fn = ((~pred_class) & (target_class)).sum().float()
        
        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn + 1e-5)
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp + 1e-5)
        
        return sensitivity.item(), specificity.item()
    
    def hausdorff_distance(self, pred: torch.Tensor, target: torch.Tensor, 
                          class_idx: int) -> float:
        """Calculate Hausdorff distance for specific class"""
        try:
            pred_class = (pred == class_idx).cpu().numpy()
            target_class = (target == class_idx).cpu().numpy()
            
            # Get surface points
            pred_points = np.argwhere(pred_class)
            target_points = np.argwhere(target_class)
            
            if len(pred_points) == 0 or len(target_points) == 0:
                return float('inf')
            
            # Calculate bidirectional Hausdorff distance
            hd1 = directed_hausdorff(pred_points, target_points)[0]
            hd2 = directed_hausdorff(target_points, pred_points)[0]
            
            return max(hd1, hd2)
            
        except Exception as e:
            logger.warning(f"Could not calculate Hausdorff distance: {e}")
            return float('inf')
    
    def volume_similarity(self, pred: torch.Tensor, target: torch.Tensor, 
                         class_idx: int) -> float:
        """Calculate volume similarity for specific class"""
        pred_volume = (pred == class_idx).sum().float()
        target_volume = (target == class_idx).sum().float()
        
        if target_volume == 0 and pred_volume == 0:
            return 1.0
        elif target_volume == 0:
            return 0.0
        
        volume_sim = 1.0 - abs(pred_volume - target_volume) / target_volume
        return max(0.0, volume_sim.item())
    
    def calculate_batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics for a batch"""
        metrics = {}
        
        # Overall accuracy
        accuracy = (pred == target).float().mean().item()
        metrics['accuracy'] = accuracy
        
        # Class-wise metrics
        dice_scores = []
        iou_scores = []
        
        for class_idx in range(self.num_classes):
            # Skip background for medical metrics
            if class_idx == 0:
                continue
                
            dice = self.dice_coefficient(pred, target, class_idx)
            iou = self.iou_score(pred, target, class_idx)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            
            metrics[f'dice_class_{self.class_names[class_idx]}'] = dice
            metrics[f'iou_class_{self.class_names[class_idx]}'] = iou
        
        # Average metrics (excluding background)
        metrics['dice_score'] = np.mean(dice_scores) if dice_scores else 0.0
        metrics['iou_score'] = np.mean(iou_scores) if iou_scores else 0.0
        
        return metrics
    
    def calculate_comprehensive_metrics(self, pred: torch.Tensor, 
                                      target: torch.Tensor) -> Dict[str, float]:
        """Calculate comprehensive metrics for evaluation"""
        metrics = {}
        
        # Convert to numpy for sklearn metrics
        pred_np = pred.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()
        
        # Overall metrics
        accuracy = accuracy_score(target_np, pred_np)
        metrics['accuracy'] = accuracy
        
        # Class-wise detailed metrics
        class_metrics = {
            'dice_scores': [],
            'iou_scores': [],
            'sensitivity': [],
            'specificity': [],
            'volume_similarity': []
        }
        
        for class_idx in range(self.num_classes):
            # Calculate all metrics for this class
            dice = self.dice_coefficient(pred, target, class_idx)
            iou = self.iou_score(pred, target, class_idx)
            sens, spec = self.sensitivity_specificity(pred, target, class_idx)
            vol_sim = self.volume_similarity(pred, target, class_idx)
            
            # Store class-specific metrics
            metrics[f'dice_{self.class_names[class_idx]}'] = dice
            metrics[f'iou_{self.class_names[class_idx]}'] = iou
            metrics[f'sensitivity_{self.class_names[class_idx]}'] = sens
            metrics[f'specificity_{self.class_names[class_idx]}'] = spec
            metrics[f'volume_sim_{self.class_names[class_idx]}'] = vol_sim
            
            # Collect for averaging (skip background)
            if class_idx > 0:
                class_metrics['dice_scores'].append(dice)
                class_metrics['iou_scores'].append(iou)
                class_metrics['sensitivity'].append(sens)
                class_metrics['specificity'].append(spec)
                class_metrics['volume_similarity'].append(vol_sim)
        
        # Average metrics (medical classes only)
        metrics['dice_score'] = np.mean(class_metrics['dice_scores'])
        metrics['iou_score'] = np.mean(class_metrics['iou_scores'])
        metrics['sensitivity'] = np.mean(class_metrics['sensitivity'])
        metrics['specificity'] = np.mean(class_metrics['specificity'])
        metrics['volume_similarity'] = np.mean(class_metrics['volume_similarity'])
        
        # Multi-class precision, recall, F1
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                target_np, pred_np, average='weighted', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
        except Exception as e:
            logger.warning(f"Could not calculate precision/recall/F1: {e}")
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1_score'] = 0.0
        
        # Clinical metrics for breast cancer detection
        metrics.update(self._calculate_clinical_metrics(pred, target))
        
        return metrics
    
    def _calculate_clinical_metrics(self, pred: torch.Tensor, 
                                   target: torch.Tensor) -> Dict[str, float]:
        """Calculate clinical metrics specific to breast cancer detection"""
        clinical_metrics = {}
        
        # Focus on malignant and tumor classes (indices 2 and 3)
        malignant_idx = 2 if self.num_classes > 2 else 1
        tumor_idx = 3 if self.num_classes > 3 else malignant_idx
        
        # Combine malignant and tumor as "cancer positive"
        if self.num_classes > 3:
            pred_cancer = ((pred == malignant_idx) | (pred == tumor_idx)).long()
            target_cancer = ((target == malignant_idx) | (target == tumor_idx)).long()
        else:
            pred_cancer = (pred == malignant_idx).long()
            target_cancer = (target == malignant_idx).long()
        
        # Clinical sensitivity/specificity for cancer detection
        cancer_sens, cancer_spec = self.sensitivity_specificity(
            pred_cancer, target_cancer, 1
        )
        
        clinical_metrics['cancer_sensitivity'] = cancer_sens
        clinical_metrics['cancer_specificity'] = cancer_spec
        
        # Positive/Negative Predictive Values
        tp = ((pred_cancer == 1) & (target_cancer == 1)).sum().float()
        fp = ((pred_cancer == 1) & (target_cancer == 0)).sum().float()
        tn = ((pred_cancer == 0) & (target_cancer == 0)).sum().float()
        fn = ((pred_cancer == 0) & (target_cancer == 1)).sum().float()
        
        ppv = tp / (tp + fp + 1e-5)  # Positive Predictive Value
        npv = tn / (tn + fn + 1e-5)  # Negative Predictive Value
        
        clinical_metrics['ppv'] = ppv.item()
        clinical_metrics['npv'] = npv.item()
        
        # Balanced accuracy for imbalanced classes
        balanced_acc = (cancer_sens + cancer_spec) / 2
        clinical_metrics['balanced_accuracy'] = balanced_acc
        
        return clinical_metrics

class MetricsVisualizer:
    """Visualize metrics and predictions"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ["Background", "Normal", "Malignant", "Tumor"]
    
    def plot_confusion_matrix(self, pred: torch.Tensor, target: torch.Tensor, 
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix"""
        pred_np = pred.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()
        
        cm = confusion_matrix(target_np, pred_np)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_class_distribution(self, target: torch.Tensor, 
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot class distribution"""
        target_np = target.cpu().numpy().flatten()
        unique, counts = np.unique(target_np, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar([self.class_names[i] for i in unique], counts)
        ax.set_title('Class Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Voxels')
        
        # Add percentage labels
        total = counts.sum()
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({100*count/total:.1f}%)',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, float], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot metrics comparison"""
        # Select key metrics for visualization
        key_metrics = {
            'Accuracy': metrics_dict.get('accuracy', 0),
            'Dice Score': metrics_dict.get('dice_score', 0),
            'IoU Score': metrics_dict.get('iou_score', 0),
            'Sensitivity': metrics_dict.get('sensitivity', 0),
            'Specificity': metrics_dict.get('specificity', 0),
            'Cancer Sensitivity': metrics_dict.get('cancer_sensitivity', 0),
            'Cancer Specificity': metrics_dict.get('cancer_specificity', 0)
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics_names = list(key_metrics.keys())
        metrics_values = list(key_metrics.values())
        
        bars = ax.bar(metrics_names, metrics_values)
        ax.set_title('Model Performance Metrics')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_volume_slice_comparison(self, volume: torch.Tensor, pred: torch.Tensor, 
                                   target: torch.Tensor, slice_idx: Optional[int] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot volume slice with prediction and ground truth"""
        if slice_idx is None:
            slice_idx = volume.shape[0] // 2  # Middle slice
        
        # Extract slices
        vol_slice = volume[slice_idx].cpu().numpy()
        pred_slice = pred[slice_idx].cpu().numpy()
        target_slice = target[slice_idx].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original volume
        axes[0].imshow(vol_slice, cmap='gray')
        axes[0].set_title('Original CT Slice')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(vol_slice, cmap='gray', alpha=0.7)
        axes[1].imshow(pred_slice, cmap='jet', alpha=0.5)
        axes[1].set_title('Prediction Overlay')
        axes[1].axis('off')
        
        # Ground truth
        axes[2].imshow(vol_slice, cmap='gray', alpha=0.7)
        axes[2].imshow(target_slice, cmap='jet', alpha=0.5)
        axes[2].set_title('Ground Truth Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def create_evaluation_report(metrics: Dict[str, float], 
                           save_path: Optional[str] = None) -> str:
    """Create comprehensive evaluation report"""
    
    report = "="*60 + "\n"
    report += "CT BREAST CANCER METASTASIS DETECTION - EVALUATION REPORT\n"
    report += "="*60 + "\n\n"
    
    # Overall Performance
    report += "OVERALL PERFORMANCE:\n"
    report += "-"*30 + "\n"
    report += f"Accuracy:              {metrics.get('accuracy', 0):.4f}\n"
    report += f"Dice Score:            {metrics.get('dice_score', 0):.4f}\n"
    report += f"IoU Score:             {metrics.get('iou_score', 0):.4f}\n"
    report += f"F1 Score:              {metrics.get('f1_score', 0):.4f}\n\n"
    
    # Medical Performance
    report += "MEDICAL PERFORMANCE:\n"
    report += "-"*30 + "\n"
    report += f"Sensitivity:           {metrics.get('sensitivity', 0):.4f}\n"
    report += f"Specificity:           {metrics.get('specificity', 0):.4f}\n"
    report += f"Cancer Sensitivity:    {metrics.get('cancer_sensitivity', 0):.4f}\n"
    report += f"Cancer Specificity:    {metrics.get('cancer_specificity', 0):.4f}\n"
    report += f"Positive Pred. Value:  {metrics.get('ppv', 0):.4f}\n"
    report += f"Negative Pred. Value:  {metrics.get('npv', 0):.4f}\n"
    report += f"Balanced Accuracy:     {metrics.get('balanced_accuracy', 0):.4f}\n\n"
    
    # Class-wise Performance
    report += "CLASS-WISE PERFORMANCE:\n"
    report += "-"*30 + "\n"
    class_names = ["Background", "Normal", "Malignant", "Tumor"]
    
    for class_name in class_names:
        dice = metrics.get(f'dice_{class_name}', 0)
        iou = metrics.get(f'iou_{class_name}', 0)
        sens = metrics.get(f'sensitivity_{class_name}', 0)
        spec = metrics.get(f'specificity_{class_name}', 0)
        
        report += f"{class_name}:\n"
        report += f"  Dice: {dice:.4f}, IoU: {iou:.4f}\n"
        report += f"  Sens: {sens:.4f}, Spec: {spec:.4f}\n\n"
    
    # Clinical Interpretation
    report += "CLINICAL INTERPRETATION:\n"
    report += "-"*30 + "\n"
    
    cancer_sens = metrics.get('cancer_sensitivity', 0)
    cancer_spec = metrics.get('cancer_specificity', 0)
    
    if cancer_sens >= 0.9:
        report += "✓ Excellent cancer detection sensitivity\n"
    elif cancer_sens >= 0.8:
        report += "✓ Good cancer detection sensitivity\n"
    else:
        report += "⚠ Cancer detection sensitivity needs improvement\n"
    
    if cancer_spec >= 0.9:
        report += "✓ Excellent cancer detection specificity\n"
    elif cancer_spec >= 0.8:
        report += "✓ Good cancer detection specificity\n"
    else:
        report += "⚠ Cancer detection specificity needs improvement\n"
    
    balanced_acc = metrics.get('balanced_accuracy', 0)
    if balanced_acc >= 0.85:
        report += "✓ Overall model performance is excellent\n"
    elif balanced_acc >= 0.75:
        report += "✓ Overall model performance is good\n"
    else:
        report += "⚠ Overall model performance needs improvement\n"
    
    report += "\n" + "="*60 + "\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report

# Test functions
if __name__ == "__main__":
    # Test metrics calculation
    print("Testing segmentation metrics...")
    
    # Create dummy data
    pred = torch.randint(0, 4, (2, 64, 64, 64))
    target = torch.randint(0, 4, (2, 64, 64, 64))
    
    # Initialize metrics
    metrics_calc = SegmentationMetrics(num_classes=4)
    
    # Calculate metrics
    batch_metrics = metrics_calc.calculate_batch_metrics(pred, target)
    comprehensive_metrics = metrics_calc.calculate_comprehensive_metrics(pred, target)
    
    print("Batch metrics:", batch_metrics)
    print("Comprehensive metrics keys:", list(comprehensive_metrics.keys()))
    
    # Test visualizer
    visualizer = MetricsVisualizer()
    
    # Create sample volume for visualization
    volume = torch.randn(64, 64, 64)
    
    # Test confusion matrix plot
    fig_cm = visualizer.plot_confusion_matrix(pred, target)
    plt.close(fig_cm)
    
    # Test metrics comparison plot
    fig_metrics = visualizer.plot_metrics_comparison(comprehensive_metrics)
    plt.close(fig_metrics)
    
    # Test evaluation report
    report = create_evaluation_report(comprehensive_metrics)
    print("\nEvaluation Report:")
    print(report)
    
    print("Metrics testing completed successfully!")