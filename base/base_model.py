import torch.nn as nn
from abc import abstractmethod
import numpy as np


class BaseModel(nn.Module):
    """Represents base class for all models."""

    @abstractmethod
    def forward(self, *inputs):
        """Forward logic for model"""
        raise NotImplementedError

    def __str__(self) -> str:
        """Prints the total number of parameters in the model."""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
