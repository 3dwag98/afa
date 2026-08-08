"""Device utilities for GPU/CPU selection in PyTorch."""

import torch


def get_device(config_device: str = "auto") -> torch.device:
    """
    Get the appropriate device for PyTorch computations.

    Args:
        config_device: Device specification string. Options:
            - "auto": Automatically select best available device
            - "cuda": Use CUDA GPU if available
            - "mps": Use Apple Metal Performance Shaders
            - "cpu": Force CPU usage

    Returns:
        torch.device: The selected device

    Prints:
        Selected device and GPU memory info if CUDA is selected.
    """
    if config_device == "auto":
        # Check for CUDA first (highest priority for performance)
        if torch.cuda.is_available():
            device_str = "cuda"
        # Check for Apple MPS
        elif torch.backends.mps.is_available():
            device_str = "mps"
        # Fall back to CPU
        else:
            device_str = "cpu"
    else:
        device_str = config_device

    device = torch.device(device_str)

    # Print device information
    print(f"Selected device: {device}")

    if device.type == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**2)
            print(f"GPU: {gpu_name}")
            print(f"Total GPU Memory: {total_memory:.2f} GB")
            print(f"Allocated GPU Memory: {allocated_memory:.2f} MB")
        else:
            print("Warning: CUDA requested but not available")

    return device
