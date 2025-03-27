import os

import matplotlib.pyplot as plt

import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import lightning as L

# Function for setting the seed
L.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

encoder = nn.Sequential(
    nn.Linear(28*28, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

decoder = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 28*28)
)

class LiteAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder, *args, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# setup data
train_dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
test_dataset = MNIST(os.getcwd(), download=True, train=False, transform=ToTensor())
val_dataset, test_dataset = utils.data.random_split(test_dataset, [0.5, 0.5])
train_loader = utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = utils.data.DataLoader(val_dataset, batch_size=32, pin_memory=True)
test_loader = utils.data.DataLoader(test_dataset, batch_size=32)

# load the model
autoencoder = LiteAutoEncoder.load_from_checkpoint(checkpoint_path="autoencoder.ckpt", encoder=encoder, decoder=decoder)

# predict
autoencoder.eval()

x, _ = next(iter(test_loader))
x = x.view(x.size(0), -1)
z = autoencoder.encoder(x)
x_hat = autoencoder.decoder(z)

# visualize
n = 5
plt.figure(figsize=(20, 4))

for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x[i].detach().numpy().reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_hat[i].detach().numpy().reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()