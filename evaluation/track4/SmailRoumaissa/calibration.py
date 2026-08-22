import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """logits: (N,16) raw (pre-softmax) scores pooled over all valid dev
    characters. labels: (N,) gold class ids, both already filtered to
    remove -100 / ignored positions."""
    T = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=0.05, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().clamp_min(1e-3).item())
