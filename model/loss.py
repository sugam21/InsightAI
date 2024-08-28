import torch.nn.functional as F
import torch


def cross_entropy_loss(output, target):
    """Takes output and target tensors and compute Cross Entropy Loss."""
    return F.cross_entropy(output, target)


if __name__ == "__main__":
    # Example of target with class indices
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    loss = cross_entropy_loss(input, target)
    loss.backward()
    print(loss.item())