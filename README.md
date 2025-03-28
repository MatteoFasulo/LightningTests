# LightningTests: A Collection of PyTorch Lightning Test Cases

This repository provides a collection of tests and examples for PyTorch Lightning which will help to understand the framework better and experiment with various configurations. While this repository is intended for my own use, feel free to explore, suggest improvements, or propose new ideas for experiments.

## Current Examples

This repository contains two primary examples:

1. **CLI Version:**  
   A simple **Autoencoder** tasked with reconstructing MNIST digits, leveraging the power of PyTorch Lightning’s Command Line Interface (CLI). This allows for rapid experimentation using various configurations and hyperparameters without writing additional code.

2. **Procedural Version:**  
   A basic **Convolutional Neural Network (CNN)** designed for MNIST digit classification. This example uses explicit procedural code for training, providing more granular control over the process.

### 1. CLI Version - Autoencoder for MNIST Reconstruction

In this version, the PyTorch Lightning CLI is used for training a simple Autoencoder to reconstruct MNIST digits. Here’s how you can experiment with custom configurations:

#### Example: Train for 3 epochs with custom learning rate and weight decay

```bash
python test_cli.py fit --model.lr 0.0001 --model.w_decay 0.001 --trainer.max_epochs 3
```

#### Run with a custom configuration file (that you can modify)

```bash
python test_cli.py fit --config config.yaml
```

### 2. Procedural Version - CNN for MNIST Classification

This version demonstrates how to train a **Convolutional Neural Network** for classifying MNIST digits with PyTorch Lightning, providing more explicit control over the training loop and model configuration.

- The model achieves **95% accuracy** on the MNIST dataset, using a Conv-BatchNorm-ReLU architecture with adaptive pooling.
  
### Known Issues

#### Windows File System Permissions:

- When running the procedural version, **Windows file system permissions** might prevent model checkpoints from being saved due to folder access restrictions.
- Aside from this, the model should function correctly, and predictions on the MNIST dataset should be accurate.

## Suggestions

Though this repository is mainly intended for my personal use, I welcome suggestions or ideas for improvements, new test cases, or any other ideas you think might be valuable. Feel free to reach out if you have any suggestions!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.