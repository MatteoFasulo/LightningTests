import os

import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torchmetrics.classification import Accuracy

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateFinder, ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.cli import LightningCLI

# Function for setting the seed
SEED = 42
L.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ConvBnRelu(L.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Model(L.LightningModule):
    def __init__(self, num_classes: int = 10, lr: float = 1e-3, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.layer_1 = ConvBnRelu(1, 16, 3, 1, 1)
        self.layer_2 = ConvBnRelu(16, 32, 3, 1, 1)
        self.layer_3 = ConvBnRelu(32, 64, 3, 1, 1)
        self.classifier = torch.nn.Linear(64, num_classes)
        self.dropout = torch.nn.Dropout(0.2)
        self.example_input_array = torch.Tensor(1, 1, 28, 28)
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(self.dropout(x))
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)  # Get raw logits
        loss = self.loss_module(logits, labels)  # Compute loss with logits
        preds = logits.argmax(dim=-1)  # Convert logits to class indices
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)  # Get raw logits
        preds = logits.argmax(dim=-1)  # Convert logits to class indices
        acc = (preds == labels).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.test_step_outputs.append({"test_acc": acc})

    def on_test_epoch_end(self):
        avg_acc = torch.stack([x["test_acc"] for x in self.test_step_outputs]).mean()
        self.log("test_acc", avg_acc, prog_bar=True)
        return {"test_acc": avg_acc}
        
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        imgs, labels = batch
        logits = self(imgs)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        return {"preds": preds, "labels": labels, "acc": acc}
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    def prepare_data(self):
        """
        Here ops that are shared since setup is applied on all processes (thus there will be data duplication)
        """
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(SEED))

        if stage == "validate":
            self.mnist_val = MNIST(self.data_dir, train=False, transform=self.transform)
        
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        
        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

def cli_main():
    cli = LightningCLI(Model, MNISTDataModule)

if __name__ == '__main__':
    trainer = L.Trainer(
        accelerator="gpu",
        precision=32,  # use 32-bit precision,
        max_epochs=10,
        accumulate_grad_batches=7,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        callbacks=[
            #EarlyStopping(
            #    monitor="val_loss", 
            #    mode="min", 
            #    patience=3,
            #    check_on_train_epoch_end=False,
            #),
            StochasticWeightAveraging(
                swa_lrs=1e-2,
            ),
            DeviceStatsMonitor(), # monitor GPU usage
            LearningRateFinder( # find optimal learning rate
                min_lr=1e-6,
                max_lr=1e-2,
                mode="exponential"
            ),
            ModelCheckpoint( # save the best model
                dirpath="models",
                save_weights_only=True,
                monitor="val_loss",
                mode="min", 
                save_top_k=1
            ), 
            LearningRateMonitor("epoch"), # log learning rate
        ],
        profiler=AdvancedProfiler(dirpath="./logs", filename="perf_logs"),
    )
    trainer.logger._log_graph = True

    # Setup the data
    mnist = MNISTDataModule()

    # Setup the model
    autoencoder = Model()

    # Train the model (uncomment to train)
    #trainer.fit(model=autoencoder, datamodule=mnist)

    # Save the model (uncomment to save)
    #trainer.save_checkpoint("models/model.ckpt")

    # load the model
    autoencoder = Model.load_from_checkpoint(checkpoint_path='models/model.ckpt')

    # Test the model
    trainer.test(autoencoder, mnist)