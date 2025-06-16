# Lokman-v2: Unified CT Breast Cancer Detection System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

🏥 **Advanced AI system for detecting breast cancer metastases in CT scans using unified 3D deep learning architecture**

## 🎯 System Overview

- **Current Architecture**: Unified SimpleAttentionUNet3D with dual-mode capability
- **Training Modes**: Classification (volume-level) and Segmentation (pixel-level)
- **Dataset**: 79 CT volumes with train(69)/val(5)/test(5) splits
- **Performance**: Optimized for both CPU testing and GPU production

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (optional, CPU mode available)
- 16GB+ RAM recommended
- 20GB+ free disk space

### Installation
```bash
# Clone and setup
git clone https://github.com/open-experiments/HC-AIX.git
cd Lokmanv2

# Install dependencies
pip install -r requirements.txt
```

### ⚡ Fastest Way to Test the System
```bash
# 1. Clean any old incompatible models
python scripts/migrate_models.py --clean

# 2. Train a new model (fast test mode)
python scripts/train.py --fast-test --model-mode classification

# 3. Evaluate the model
python scripts/evaluate_model.py --fast-test

# 4. Run the web interface
python web/ocp_app.py --model models/best_model.pth
```

## 📋 When to Use Which Script

### 🔄 **Data Pipeline Scripts** (Use in Order)

#### 1. **Starting with Raw Data**
```bash
# When: You have raw DICOM files and need to process them
# Purpose: Convert DICOM to training-ready format
python scripts/prep_data.py --data-root data/raw --output data/processed

# When: You need to check clinical information first
# Purpose: Understand patient data structure
python scripts/read_clinical_data.py --excel-file "data/original/MEME HASTA EXCEL LİSTESİ.xlsx"

# When: You need to remove patient names/PII
# Purpose: Anonymize data for HIPAA compliance
python scripts/anonymize_pii.py --data-root data/processed
```

#### 2. **Data Quality & Validation**
```bash
# When: After data preparation, before training
# Purpose: Validate data integrity and check for issues
python scripts/validate_data_quality.py --data-root data/

# When: You want to understand your dataset splits
# Purpose: Analyze train/val/test distribution and labels
python scripts/show_dataset_splits.py --data-root data/
```

#### 3. **Dataset Expansion**
```bash
# When: You have small dataset (< 100 samples)
# Purpose: Expand dataset with geometric augmentations
python scripts/augment_dataset.py --data-root data/processed

# When: You want custom augmentations
# Purpose: Apply specific transformations
python scripts/augment_dataset.py --data-root data/processed --augmentations rotate_90 flip_horizontal
```

### 🎯 **Training Scripts** (Choose One)

#### **Main Training Script: `train.py`** ⭐
```bash
# When: Starting new project or need fast testing
# Purpose: Quick CPU testing with small volumes
python scripts/train.py --fast-test --model-mode classification

# When: You have GPU and want standard training
# Purpose: Full training with 160³ volumes
python scripts/train.py --data-root data/ --model-mode classification

# When: You need segmentation instead of classification  
# Purpose: Pixel-level analysis
python scripts/train.py --data-root data/ --model-mode segmentation

# When: You have memory constraints
# Purpose: Custom volume size for your hardware
python scripts/train.py --target-size 96 96 96 --batch-size 2

# When: You want to use optimized config
# Purpose: Use pre-configured fast testing setup
python scripts/train.py --config configs/config_fast_test.yaml --model-mode classification
```

### 📊 **Evaluation Scripts**

#### **Model Evaluation: `evaluate_model.py`** 
```bash
# When: After training, want to test model performance
# Purpose: Quick evaluation with small volumes (CPU friendly)
python scripts/evaluate_model.py --fast-test

# When: You want comprehensive evaluation
# Purpose: Full evaluation with detailed metrics and plots
python scripts/evaluate_model.py --data-root data/

# When: You have specific model to evaluate
# Purpose: Evaluate custom model file
python scripts/evaluate_model.py --model models/custom_model.pth --data-root data/

# When: Model was trained with custom volume size
# Purpose: Match evaluation to training configuration
python scripts/evaluate_model.py --target-size 96 96 96 --data-root data/
```

### 🔧 **Maintenance Scripts**

#### **Model Migration: `migrate_models.py`**
```bash
# When: Upgrading from old system or getting architecture errors
# Purpose: Check what models would be affected (safe preview)
python scripts/migrate_models.py --dry-run

# When: You have old incompatible models
# Purpose: Backup and remove old models
python scripts/migrate_models.py --clean

# When: You want to keep old models but clean workspace
# Purpose: Backup only, don't delete
python scripts/migrate_models.py

# When: Automating migration
# Purpose: Force migration without user prompts
python scripts/migrate_models.py --clean --force
```

#### **System Testing: `demo.py`**
```bash
# When: Setting up system for first time
# Purpose: Quick test that all components work
python scripts/demo.py

# When: Troubleshooting installation issues
# Purpose: Validate core functionality
python scripts/demo.py
```

## 🔀 Complete Workflows

### 📚 **Workflow 1: Starting from Scratch (New Dataset)**
```bash
# Step 1: Prepare raw DICOM data
python scripts/prep_data.py --data-root data/raw --output data/processed

# Step 2: Anonymize patient data
python scripts/anonymize_pii.py --data-root data/processed

# Step 3: Validate data quality
python scripts/validate_data_quality.py --data-root data/

# Step 4: Expand dataset (if small)
python scripts/augment_dataset.py --data-root data/processed

# Step 5: Train model
python scripts/train.py --fast-test --model-mode classification

# Step 6: Evaluate model
python scripts/evaluate_model.py --fast-test

# Step 7: Deploy web interface
python web/ocp_app.py --model models/best_model.pth
```

### ⚡ **Workflow 2: Quick Testing (Existing Processed Data)**
```bash
# Step 1: Clean old models (if any errors)
python scripts/migrate_models.py --clean

# Step 2: Fast training test
python scripts/train.py --fast-test --model-mode classification

# Step 3: Quick evaluation
python scripts/evaluate_model.py --fast-test

# Step 4: Test web interface
python web/ocp_app.py --model models/best_model.pth
```

### 🏭 **Workflow 3: Production Training (GPU Environment)**
```bash
# Step 1: Validate data
python scripts/validate_data_quality.py --data-root data/

# Step 2: Check dataset splits
python scripts/show_dataset_splits.py --data-root data/

# Step 3: Production training
python scripts/train.py --data-root data/ --model-mode classification --epochs 100

# Step 4: Comprehensive evaluation
python scripts/evaluate_model.py --data-root data/

# Step 5: Deploy for production
python web/ocp_app.py --host 0.0.0.0 --port 30080 --model models/best_model.pth
```

### 🔄 **Workflow 4: Troubleshooting & Maintenance**
```bash
# Check system setup
python scripts/demo.py

# Analyze data issues
python scripts/validate_data_quality.py --data-root data/
python scripts/show_dataset_splits.py --data-root data/

# Handle model compatibility issues
python scripts/migrate_models.py --dry-run
python scripts/migrate_models.py --clean

# Test with minimal resources
python scripts/train.py --fast-test --model-mode classification
```

## 🎛️ **Script Selection Guide**

### **Based on Your Situation:**

| **Your Situation** | **Scripts to Use** | **Commands** |
|-------------------|-------------------|-------------|
| 🆕 **New User/Testing** | `migrate_models.py` → `train.py` → `evaluate_model.py` | `--clean` → `--fast-test` → `--fast-test` |
| 💻 **CPU Only** | `train.py` → `evaluate_model.py` | `--fast-test` → `--fast-test` |
| 🚀 **GPU Available** | `train.py` → `evaluate_model.py` | Standard mode → Standard mode |
| 📊 **Raw DICOM Data** | `prep_data.py` → `validate_data_quality.py` → `train.py` | Full pipeline |
| 🔍 **Small Dataset** | `augment_dataset.py` → `train.py` | Expand then train |
| ❌ **Model Errors** | `migrate_models.py` → `train.py` | `--clean` → `--fast-test` |
| 🔧 **Troubleshooting** | `demo.py` → `validate_data_quality.py` | Check system |
| 🏥 **Production** | `train.py` → `evaluate_model.py` → `web/ocp_app.py` | Full pipeline |

### **Based on Hardware:**

| **Hardware** | **Recommended Commands** |
|-------------|-------------------------|
| **CPU Only** | `--fast-test --target-size 64 64 64 --batch-size 1` |
| **Low Memory GPU** | `--target-size 96 96 96 --batch-size 2` |
| **Standard GPU** | `--target-size 128 128 128 --batch-size 4` |
| **High-end GPU** | `--target-size 160 160 160 --batch-size 4` |

## 📁 Project Structure

```
lokman-v2/
├── scripts/                  # 🔧 Executable Scripts
│   ├── train.py             # ⭐ Main: Unified training (classification/segmentation)
│   ├── evaluate_model.py    # 📊 Main: Unified evaluation with auto-detection
│   ├── migrate_models.py    # 🔄 Migration: Handle old incompatible models
│   ├── prep_data.py         # 📥 Data: DICOM to training format
│   ├── augment_dataset.py   # 📈 Data: Expand dataset with augmentations
│   ├── validate_data_quality.py # ✅ QA: Check data integrity
│   ├── show_dataset_splits.py   # 📋 Analysis: View data distribution
│   ├── read_clinical_data.py    # 🏥 Clinical: Process medical records
│   ├── anonymize_pii.py         # 🔐 Privacy: Remove patient information
│   └── demo.py                  # 🎮 Testing: Quick system validation
├── configs/                 # ⚙️ Configuration Files
│   ├── config.yaml         # Standard configuration
│   └── config_fast_test.yaml # Optimized for CPU testing
├── core/                    # 🧠 Core Components
├── data/                    # 📊 Dataset
├── models/                  # 🤖 Trained Models & Results
├── web/                     # 🌐 Web Interface
└── utils/                   # 🔧 Utilities
```

## ⚙️ **Configuration Options**

### **Fast Test Mode** (CPU-friendly)
- Volume size: 64³ (1.5 MB per sample)
- Batch size: 1
- Epochs: 2
- Mixed precision: Disabled

### **Standard Mode** (GPU recommended)
- Volume size: 160³ (23.4 MB per sample)
- Batch size: 4
- Epochs: 100
- Mixed precision: Enabled

### **Custom Configuration**
Edit `configs/config.yaml` or use command line overrides:
```bash
python scripts/train.py --target-size 128 128 128 --batch-size 2 --epochs 50
```

## 🚀 **OpenShift Deployment**

```bash
# Deploy to OpenShift
oc new-project lokman-v2
oc apply -f openshift/
oc expose svc/lokman-v2

# Get application URL
oc get route lokman-v2
```

## 🔧 **Troubleshooting**

### **Common Issues & Solutions**

| **Issue** | **Solution** | **Command** |
|-----------|-------------|-------------|
| Model loading errors | Migrate old models | `python scripts/migrate_models.py --clean` |
| Slow training | Use fast test mode | `python scripts/train.py --fast-test` |
| Memory errors | Reduce batch/volume size | `--batch-size 1 --target-size 64 64 64` |
| No model found | Train new model | `python scripts/train.py --fast-test --model-mode classification` |
| Data validation fails | Check data quality | `python scripts/validate_data_quality.py` |

### **Getting Help**
1. 📖 Check `TROUBLESHOOTING.md` for detailed solutions
2. 🧪 Run `python scripts/demo.py` to test system setup
3. 📊 Use `python scripts/validate_data_quality.py` to check data
4. ⚡ Try fast test mode first: `--fast-test`

## 📚 **Key Features**

✅ **Unified Architecture** - Single training script for all modes  
✅ **Fast Testing** - CPU-friendly mode for quick iteration  
✅ **Auto-Detection** - Automatic model/data compatibility checking  
✅ **Migration Support** - Handle old incompatible models  
✅ **HIPAA Compliant** - Patient data anonymization  
✅ **Production Ready** - OpenShift deployment support  
✅ **Comprehensive Evaluation** - Detailed metrics and visualizations  

---

**Note**: This system is for research and development purposes. Clinical deployment requires proper validation and regulatory approval.

🔗 **Quick Links**
- [Configuration Details](configs/config.yaml)
