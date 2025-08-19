import torch

def test_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

def test_pytorch():
    print(f"PyTorch version: {torch.__version__}")

def test_gpu_compatibility():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = torch.randn(3, 3).to(device)
        print("Tensor on GPU:", tensor)
    else:
        print("No compatible GPU found.")

if __name__ == "__main__":
    test_cuda()
    test_pytorch()
    test_gpu_compatibility()