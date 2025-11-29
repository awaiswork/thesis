"""Device detection utilities."""

import torch


def get_device() -> torch.device:
    """
    Detect and return the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device for the best available hardware.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU backend.")
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU.")

    print(f"PyTorch version: {torch.__version__}")
    return device


def get_device_string() -> str:
    """
    Get device as a string (for YOLO compatibility).

    Returns:
        Device name as string: "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


