import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision.models import mobilenet_v3_large, squeezenet1_1, vgg16


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes=17, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        logger.info("Saving Hyper parameters")
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)

        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1score = F1Score(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x

    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("train_f1_score", f1, on_step=True, on_epoch=True, logger=True)
        self.log("train_precision", precision, on_step=True, on_epoch=True, logger=True)
        self.log("train_recall", recall, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1_score", f1, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# VGGMODULE
class VGGModel(pl.LightningModule):
    def __init__(self, num_classes=17, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        logger.info("Saving Hyper parameters")
        self.learning_rate = learning_rate
        self.model = vgg16(weights="DEFAULT", progress=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes, bias=True)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1score = F1Score(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)

    # will be used during inference
    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("train_f1_score", f1, on_step=True, on_epoch=True, logger=True)
        self.log("train_precision", precision, on_step=True, on_epoch=True, logger=True)
        self.log("train_recall", recall, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1_score", f1, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# SQUEEZE-NET
class SqueezeNetModel(pl.LightningModule):
    def __init__(self, num_classes=17, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        logger.info("Saving Hyper parameters")
        self.learning_rate = learning_rate
        self.model = squeezenet1_1(
            weights="SqueezeNet1_1_Weights.DEFAULT", progress=True
        )
        self.model.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1score = F1Score(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)

    # will be used during inference
    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("train_f1_score", f1, on_step=True, on_epoch=True, logger=True)
        self.log("train_precision", precision, on_step=True, on_epoch=True, logger=True)
        self.log("train_recall", recall, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1_score", f1, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# MOBILE NET
class MobileNetModel(pl.LightningModule):
    def __init__(self, num_classes=17, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        logger.info("Saving Hyper parameters")
        self.learning_rate = learning_rate
        self.model = mobilenet_v3_large(weights="DEFAULT", progress=True)
        self.model.classifier[3] = nn.Linear(1280, num_classes, bias=True)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1score = F1Score(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)

    # will be used during inference
    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("train_f1_score", f1, on_step=True, on_epoch=True, logger=True)
        self.log("train_precision", precision, on_step=True, on_epoch=True, logger=True)
        self.log("train_recall", recall, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1_score", f1, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    mobile_net = MobileNetModel()
    tensor = torch.rand([1, 3, 224, 224])
    print(mobile_net.forward(tensor))
