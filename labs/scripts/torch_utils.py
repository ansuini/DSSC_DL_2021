import torch

def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"