import torch


def accuracy(output, target) -> float:
    """Takes output and target tensor and returns the accuracy"""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        assert prediction.shape[0] == target.shape[0]
        correct = torch.sum(prediction == target).item()
    return correct / len(target)
