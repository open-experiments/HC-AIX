# Cancer Detection using InceptionV3

An optimized deep learning system for cancer detection from mammography images, classifying them as Benign, Malignant, or Normal.

Authors: Fatih E. NAR (Red Hat), Sedat Kaymaz (Microsoft)

## Overview

This project implements a transfer learning approach using the InceptionV3 architecture, optimized for multi-GPU performance. The system processes mammography images from the MIAS dataset and classifies them into three categories: Benign, Malignant, or Normal.

The code follows 12-factor app principles for better scalability, maintainability, and reliability.

## Key Features

- Transfer learning with InceptionV3 and MobileNetV2 architectures
- Multi-GPU support for distributed training
- K-fold cross-validation
- Ensemble model prediction
- Advanced data augmentation
- Imbalanced class handling
- Comprehensive evaluation metrics
- Visualization tools
- Environment-based configuration

## Project Structure

```
breast_cancer_detection/
├── config/               # Configuration management
├── data/                 # Directory for storing data
├── logs/                 # Directory for logs
├── models/               # Model architecture definitions
├── preprocessing/        # Data loading and augmentation
├── training/             # Training functionality
├── evaluation/           # Metrics and visualization
├── utils/                # Utility functions
├── main.py               # Main entry point
├── requirements.txt      # Dependencies
├── .env.example          # Environment variables example
└── README.md             # Project documentation
```

## Requirements

- Python 3.7+
- TensorFlow 2.7+
- CUDA 11.2+ (for GPU acceleration)
- See requirements.txt for full dependencies

## Installation

1. Clone the repository:

```bash
git clone https://github.com/open-experiments/HC-AIX.git
cd HC-AIX/02-PROMODELING
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment:

```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Data ETL
Get Data Uncompressed and Distributed:
```bash
python 01-prepare_data.py 
```

## Training

### PRO-MODELING & INFERENCE
Train a model using default settings:

```bash
python 02-main.py --mode train --batch-size 128 --xla --amp
```

Use cross-validation:

```bash
python 02-main.py --mode train --cross-validation
```

Use lightweight model (MobileNetV2):

```bash
python 02-main.py --mode train --lightweight
```

### QUICK & DIRTY (LOTR)
```bash
python 03-LOTR.py
```

### Evaluation

Evaluate a saved model:

```bash
python main.py --mode evaluate --model-path data/best_weights.weights.h5
```

### Prediction

Make predictions on new data:

```bash
python main.py --mode predict --model-path data/best_weights.weights.h5 --data-dir path/to/data
```

### Additional Options

- `--batch-size`: Set batch size for training/evaluation
- `--epochs`: Number of training epochs
- `--learning-rate`: Initial learning rate
- `--data-dir`: Directory containing dataset
- `--results-dir`: Directory to save results
- `--visualize`: Enable visualization of samples and results
- `--debug`: Enable debug logging
- `--ensemble`: Use ensemble prediction (with multiple models)

## Dataset

This project is designed to work with the Mamm
