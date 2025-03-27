import os

import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.cli import LightningCLI

# Function for setting the seed
SEED = 42
L.seed_everything(SEED)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LiteAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder, lr: float = 1e-3, w_decay: float = 1e-5, *args, **kwargs):
        super().__init__()

        self.lr = lr
        self.w_decay = w_decay

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )

        self.example_input_array = torch.Tensor(1, 28*28)
        self.save_hyperparameters()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.w_decay)
        return optimizer

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = os.getcwd(), batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

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
    cli = LightningCLI(LiteAutoEncoder, MNISTDataModule)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
#profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
#trainer = L.Trainer(
#    max_epochs=10,
#    callbacks=[
#        EarlyStopping(monitor="val_loss", mode="min", patience=3),
#        DeviceStatsMonitor(),
#    ],
#    profiler=profiler,
#)
#trainer.fit(autoencoder, mnist)

# test the model
#trainer.test(autoencoder, mnist)

# save the model
#trainer.save_checkpoint("autoencoder.ckpt")

if __name__ == '__main__':
    cli_main()