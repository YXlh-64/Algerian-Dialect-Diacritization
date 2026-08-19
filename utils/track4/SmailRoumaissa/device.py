import torch


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected.")
    return device
