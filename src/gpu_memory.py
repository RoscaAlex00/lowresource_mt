import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:")
            print(f"Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"Used memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"Free memory: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1024**3:.2f} GB")
            print('-' * 40)
    else:
        print("CUDA is not available.")


print_gpu_memory()