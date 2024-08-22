import torch
from torchvision.models import squeezenet1_1, mobilenet_v3_large
import torch.nn as nn


class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = squeezenet1_1(weights="SqueezeNet1_1_Weights.DEFAULT", progress=True)
        self.model.classifier[1] = nn.Conv2d(512, 17, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, image):
        model_output = self.model(image)
        return nn.Softmax(dim=1)(model_output)


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v3_large(weights="DEFAULT", progress=True)
        self.model.classifier[3] = nn.Linear(1280, 17, bias=True)

    def forward(self, image):
        model_output = self.model(image)
        return nn.Softmax(dim=1)(model_output)
