"""
Quick check: is PyTorch seeing your NVIDIA GPU?
If this says "CUDA: No", you have the CPU-only build of PyTorch.
Reinstall PyTorch with CUDA (see commands below).
"""
import sys

def main():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        return 1

    print("PyTorch version:", torch.__version__)
    print("CUDA built into this PyTorch:", torch.version.cuda or "No (CPU-only build)")
    print("CUDA available now:", "Yes" if torch.cuda.is_available() else "No")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        return 0

    print()
    if torch.version.cuda is None:
        print("Your PyTorch was installed without CUDA, so it cannot use your RTX 3080.")
        print()
        print("Fix: reinstall PyTorch with CUDA support (pick one, then restart the app):")
        print()
        print("  # CUDA 12.1 (recommended for recent drivers):")
        print("  pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("  # CUDA 11.8:")
        print("  pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("Then run this script again to confirm.")
    else:
        print("PyTorch has CUDA but the GPU is not visible. Check:")
        print("  - NVIDIA drivers installed (nvidia-smi in a terminal)")
        print("  - No other process holding the GPU exclusively")
    return 1

if __name__ == "__main__":
    sys.exit(main())
