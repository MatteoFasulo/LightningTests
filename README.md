# LightningTests

This repo is a collection of tests for PyTorch Lightning which might be useful for testing and debugging in the future. There will be some python scripts and (maybe) some notebooks for testing.

Right now there are two examples, one using the Lightning CLI version and the other using explicit procedural code for training.

The CLI version is just a simple Autoencoder trying to reconstuct MNIST digits and the procedural version is a simple Convolutional Neural Network for classifying MNIST digits.

For the CLI version:

* Running training for 3 epochs with custom learning rate and weight decay

```bash
python test_cli.py fit --model.lr 0.0001 --model.w_decay 0.001 --trainer.max_epochs 3
```

Run with custom config file

```bash
python test_cli.py fit --config config.yaml
```

## Issues

For the procedural version there are some issues with the Windows file system permissions. Model checkpoints are unable to be saved due to folder permissions. Except for that, it should work and predict with 95% accuracy on the MNIST dataset using Conv-BatchNorm-ReLU architecture with adaptive pooling.
