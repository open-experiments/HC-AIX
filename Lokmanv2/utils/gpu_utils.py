"""
GPU Acceleration Optimizer for Maximum Performance
Implements comprehensive GPU optimization strategies across the pipeline
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
import numpy as np
from typing import Optional, List, Tuple, Dict
import logging
import os
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

class GPUOptimizer:
    """Advanced GPU optimization for medical imaging pipeline"""
    
    def __init__(self, enable_fp16: bool = True, 
                 enable_channels_last: bool = True,
                 enable_cudnn_benchmark: bool = True,
                 enable_tf32: bool = True):
        """
        Initialize GPU optimizer with performance flags
        
        Args:
            enable_fp16: Use automatic mixed precision
            enable_channels_last: Use memory-efficient tensor format
            enable_cudnn_benchmark: Enable cuDNN autotuner
            enable_tf32: Enable TensorFloat-32 for Ampere GPUs
        """
        self.enable_fp16 = enable_fp16
        self.enable_channels_last = enable_channels_last
        self.enable_cudnn_benchmark = enable_cudnn_benchmark
        self.enable_tf32 = enable_tf32
        
        # Configure global settings
        self._configure_gpu_settings()
        
        # Log GPU information
        self._log_gpu_info()
    
    def _configure_gpu_settings(self):
        """Configure optimal GPU settings"""
        if torch.cuda.is_available():
            # Enable cuDNN autotuner for optimal convolution algorithms
            if self.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                logger.info("✓ cuDNN autotuner enabled")
            
            # Enable TensorFloat-32 for Ampere GPUs (RTX 30xx, A100)
            if self.enable_tf32 and torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✓ TensorFloat-32 enabled for Ampere GPU")
            
            # Set memory allocator settings
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            # Enable CUDA graphs for static models (inference)
            if hasattr(torch.cuda, 'set_sync_debug_mode'):
                torch.cuda.set_sync_debug_mode(0)
    
    def _log_gpu_info(self):
        """Log detailed GPU information"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Memory: {props.total_memory / 1e9:.1f} GB")
                logger.info(f"  Compute Capability: {props.major}.{props.minor}")
                logger.info(f"  Multi-processors: {props.multi_processor_count}")
    
    @contextmanager
    def optimize_model(self, model: nn.Module, distributed: bool = False):
        """
        Context manager for model optimization
        
        Args:
            model: PyTorch model to optimize
            distributed: Use distributed data parallel
        """
        if torch.cuda.is_available():
            # Move model to GPU
            model = model.cuda()
            
            # Convert to channels_last format for better performance
            if self.enable_channels_last:
                model = model.to(memory_format=torch.channels_last_3d)
                logger.info("✓ Model converted to channels_last format")
            
            # Compile model with torch.compile (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("✓ Model compiled with torch.compile")
            
            # Wrap with DataParallel or DistributedDataParallel
            if distributed and dist.is_initialized():
                model = DistributedDataParallel(model)
                logger.info("✓ Using DistributedDataParallel")
            elif torch.cuda.device_count() > 1:
                model = DataParallel(model)
                logger.info(f"✓ Using DataParallel on {torch.cuda.device_count()} GPUs")
        
        try:
            yield model
        finally:
            # Cleanup
            torch.cuda.empty_cache()
    
    def optimize_data_loading(self, dataloader):
        """Optimize data loading for GPU"""
        if torch.cuda.is_available():
            # Prefetch data to GPU
            dataloader.pin_memory = True
            dataloader.num_workers = min(os.cpu_count(), 8)
            dataloader.persistent_workers = True
            dataloader.prefetch_factor = 2
            
            logger.info("✓ Data loading optimized for GPU")
        
        return dataloader
    
    @contextmanager
    def mixed_precision_context(self):
        """Context manager for mixed precision training"""
        if self.enable_fp16 and torch.cuda.is_available():
            scaler = amp.GradScaler()
            
            @contextmanager
            def autocast_context():
                with amp.autocast(device_type='cuda', dtype=torch.float16):
                    yield
            
            yield scaler, autocast_context
        else:
            # Dummy context for CPU
            @contextmanager
            def dummy_context():
                yield
            
            class DummyScaler:
                def scale(self, loss): return loss
                def step(self, optimizer): optimizer.step()
                def update(self): pass
                def unscale_(self, optimizer): pass
            
            yield DummyScaler(), dummy_context
    
    def optimize_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference"""
        if torch.cuda.is_available():
            model.eval()
            
            # Enable inference mode optimizations
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d)):
                    module.track_running_stats = False
            
            # Use torch.jit.script for static models
            try:
                example_input = torch.randn(1, 1, 128, 128, 128).cuda()
                if self.enable_channels_last:
                    example_input = example_input.to(memory_format=torch.channels_last_3d)
                
                with torch.no_grad():
                    traced_model = torch.jit.trace(model, example_input)
                logger.info("✓ Model traced with TorchScript")
                return traced_model
            except Exception as e:
                logger.warning(f"Could not trace model: {e}")
                return model
        
        return model
    
    def profile_performance(self, model: nn.Module, input_shape: Tuple[int, ...]):
        """Profile model performance"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available for profiling")
            return
        
        # Ensure model is on GPU
        model = model.cuda()
        model.eval()
        input_tensor = torch.randn(input_shape).cuda()
        
        if self.enable_channels_last:
            input_tensor = input_tensor.to(memory_format=torch.channels_last_3d)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        torch.cuda.synchronize()
        
        # Profile
        times = []
        for _ in range(100):
            start = time.time()
            with torch.no_grad():
                _ = model(input_tensor)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        mean_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        throughput = 1000 / mean_time  # Images per second
        
        logger.info(f"Performance Profile:")
        logger.info(f"  Mean inference time: {mean_time:.2f} ± {std_time:.2f} ms")
        logger.info(f"  Throughput: {throughput:.1f} images/second")
        logger.info(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

class GPUMemoryManager:
    """Intelligent GPU memory management"""
    
    def __init__(self, reserved_memory_mb: int = 1024):
        """
        Initialize memory manager
        
        Args:
            reserved_memory_mb: Memory to keep free (MB)
        """
        self.reserved_memory_mb = reserved_memory_mb
    
    def get_optimal_batch_size(self, model: nn.Module, 
                             input_shape: Tuple[int, ...],
                             max_batch_size: int = 32) -> int:
        """Find optimal batch size for available GPU memory"""
        if not torch.cuda.is_available():
            return 1
        
        # Ensure model is on GPU
        model = model.cuda()
        model.eval()
        
        # Binary search for optimal batch size
        left, right = 1, max_batch_size
        optimal_batch_size = 1
        
        while left <= right:
            mid = (left + right) // 2
            
            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Try batch size
                test_input = torch.randn(mid, *input_shape[1:]).cuda()
                with torch.no_grad():
                    _ = model(test_input)
                
                # Check memory usage
                memory_used = torch.cuda.max_memory_allocated() / 1e6  # MB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e6
                memory_free = memory_total - memory_used - self.reserved_memory_mb
                
                if memory_free > 0:
                    optimal_batch_size = mid
                    left = mid + 1
                else:
                    right = mid - 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    right = mid - 1
                else:
                    raise
            finally:
                torch.cuda.empty_cache()
        
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    @contextmanager
    def efficient_inference(self, clear_cache_every: int = 10):
        """Context manager for memory-efficient inference"""
        try:
            yield
        finally:
            torch.cuda.empty_cache()
    
    def log_memory_usage(self):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                memory_reserved = torch.cuda.memory_reserved(i) / 1e9
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                
                logger.info(f"GPU {i} Memory:")
                logger.info(f"  Allocated: {memory_allocated:.2f} GB")
                logger.info(f"  Reserved: {memory_reserved:.2f} GB")
                logger.info(f"  Total: {memory_total:.2f} GB")
                logger.info(f"  Free: {memory_total - memory_allocated:.2f} GB")

class CUDAGraphOptimizer:
    """CUDA Graphs for static model acceleration"""
    
    def __init__(self):
        self.graphs = {}
        self.static_inputs = {}
        self.static_outputs = {}
    
    def capture_graph(self, model: nn.Module, input_shape: Tuple[int, ...], 
                     graph_name: str = "default"):
        """Capture CUDA graph for static model"""
        if not torch.cuda.is_available() or not hasattr(torch.cuda, 'CUDAGraph'):
            logger.warning("CUDA Graphs not available")
            return model
        
        model.eval()
        
        # Create static input/output
        static_input = torch.randn(input_shape).cuda()
        self.static_inputs[graph_name] = static_input
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(static_input)
        
        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = model(static_input)
        
        self.graphs[graph_name] = graph
        self.static_outputs[graph_name] = static_output
        
        logger.info(f"✓ CUDA Graph captured: {graph_name}")
        
        def graph_replay(input_tensor):
            self.static_inputs[graph_name].copy_(input_tensor)
            self.graphs[graph_name].replay()
            return self.static_outputs[graph_name].clone()
        
        return graph_replay

# Utility functions for easy integration
def auto_gpu_optimize(model: nn.Module, config) -> Tuple[nn.Module, GPUOptimizer]:
    """Automatically optimize model for GPU"""
    optimizer = GPUOptimizer(
        enable_fp16=config.training.use_mixed_precision,
        enable_channels_last=True,
        enable_cudnn_benchmark=True,
        enable_tf32=True
    )
    
    with optimizer.optimize_model(model) as optimized_model:
        return optimized_model, optimizer

def profile_model_performance(model: nn.Module, input_shape: Tuple[int, ...]):
    """Profile model performance with GPU optimization"""
    # Ensure model is on GPU
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = GPUOptimizer()
    optimizer.profile_performance(model, input_shape)
    
    # Memory profiling
    memory_manager = GPUMemoryManager()
    optimal_batch_size = memory_manager.get_optimal_batch_size(model, input_shape)
    memory_manager.log_memory_usage()
    
    return optimal_batch_size

# Integration with training pipeline
class GPUAcceleratedTrainer:
    """GPU-optimized trainer wrapper"""
    
    def __init__(self, base_trainer, gpu_optimizer: GPUOptimizer):
        self.base_trainer = base_trainer
        self.gpu_optimizer = gpu_optimizer
        self.memory_manager = GPUMemoryManager()
    
    def fit(self, train_loader, val_loader):
        """GPU-accelerated training"""
        # Optimize data loaders
        train_loader = self.gpu_optimizer.optimize_data_loading(train_loader)
        val_loader = self.gpu_optimizer.optimize_data_loading(val_loader)
        
        # Log initial memory
        self.memory_manager.log_memory_usage()
        
        # Train with base trainer
        with self.gpu_optimizer.optimize_model(self.base_trainer.model) as model:
            self.base_trainer.model = model
            self.base_trainer.fit(train_loader, val_loader)
        
        # Log final memory
        self.memory_manager.log_memory_usage()

if __name__ == "__main__":
    # Test GPU optimization
    print("Testing GPU Optimization...")
    
    if torch.cuda.is_available():
        # Create dummy model
        from models.attention_unet_3d import AttentionUNet3D
        model = AttentionUNet3D(in_channels=1, out_channels=4, base_channels=32, depth=4)
        
        # Test optimization
        optimizer = GPUOptimizer()
        
        # Profile performance
        print("\nProfiling model performance...")
        profile_model_performance(model, (1, 1, 128, 128, 128))
        
        # Test memory manager
        print("\nTesting memory management...")
        memory_manager = GPUMemoryManager()
        optimal_batch = memory_manager.get_optimal_batch_size(model, (1, 1, 128, 128, 128))
        print(f"Optimal batch size: {optimal_batch}")
        
        print("\nGPU optimization tests completed!")
    else:
        print("CUDA not available - skipping GPU tests")