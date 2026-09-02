import torch
import torchaudio
import numpy as np
import scipy
import torch_pruning
import onnx
import onnxruntime


def test_environment():
    print("=== Environment Verification ===")
    print(f"PyTorch: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    print(f"Torchaudio: {torchaudio.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Torch-Pruning: {torch_pruning.__version__}")
    print(f"ONNX: {onnx.__version__}")
    print(f"ONNX Runtime: {onnxruntime.__version__}")

    x = torch.randn(2, 64, 21)
    assert x.shape == (2, 64, 21), "Tensor shape mismatch"
    print("[PASS] Environment is ready!")


if __name__ == "__main__":
    test_environment()
