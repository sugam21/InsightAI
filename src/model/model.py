from torchvision.models import squeezenet1_1, mobilenet_v3_large
import torch.nn as nn
from src.base import BaseModel
from src.utils import get_logger

LOG = get_logger("model")


class SqueezeNet(BaseModel):

    def __init__(self, out_feature: int):
        super().__init__()
        self.model = squeezenet1_1(weights="SqueezeNet1_1_Weights.DEFAULT",
                                   progress=True)
        self.model.classifier[1] = nn.Conv2d(512,
                                             out_feature,
                                             kernel_size=(1, 1),
                                             stride=(1, 1))
        LOG.debug("Successfully loaded the model.✔")

    def forward(self, image):
        model_output = self.model(image)
        return nn.Softmax(dim=1)(model_output)


class MobileNet(nn.Module):

    def __init__(self, out_feature: int):
        super().__init__()
        self.model = mobilenet_v3_large(weights="DEFAULT", progress=True)
        self.model.classifier[3] = nn.Linear(1280, out_feature, bias=True)
        LOG.debug("Successfully loaded the model.✔")

    def forward(self, image):
        model_output = self.model(image)
        return nn.Softmax(dim=1)(model_output)


if __name__ == "__main__":
    model = SqueezeNet(17)
    print(model.test())
