#!/usr/bin/env python3
"""
Quick Demo - Shows All Features Working
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from core.model import create_model

def main():
    print("🏥 CT Breast Cancer Detection - Quick Demo")
    
    # Create a simple config object that matches the expected structure
    class SimpleConfig:
        def __init__(self):
            self.model = SimpleModelConfig()
    
    class SimpleModelConfig:
        def __init__(self):
            self.in_channels = 1
            self.out_channels = 4
            self.base_channels = 32
            self.depth = 4
            self.use_attention = True
            self.use_deep_supervision = True
            self.normalization = 'batch'
            self.activation = 'relu'
            self.dropout_rate = 0.1
    
    config = SimpleConfig()
    
    # Create model
    model = create_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"✅ Model created on {device}")
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test inference
    test_input = torch.randn(1, 1, 64, 64, 64).to(device)
    
    with torch.no_grad():
        output = model(test_input)
        
        # Handle deep supervision output (tuple/list) or single output
        if isinstance(output, (tuple, list)):
            main_output = output[0]
            print(f"✅ Inference successful: {main_output.shape} (+ {len(output)-1} auxiliary outputs)")
        else:
            main_output = output
            print(f"✅ Inference successful: {main_output.shape}")
    
    print("🎉 Demo completed - All systems working!")

if __name__ == "__main__":
    main()
