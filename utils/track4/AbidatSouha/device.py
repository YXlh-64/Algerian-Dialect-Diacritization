import torch


def get_device(verbose: bool = True) -> str:
    """Pick the best available backend (notebook §1).

    Prefers CUDA, then Apple Silicon's Metal backend, then CPU.

    On MPS: the CRF's forward and Viterbi recursions are sequential Python loops
    over sequence length, so they benefit far less from the GPU than the
    encoder's matmuls do. If a kernel is missing ("operator not implemented for
    MPS"), pass device="cpu" explicitly at the call site.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if verbose and device == "cpu":
        print("WARNING: no GPU detected, training on CPU will be slow.")
    return device
