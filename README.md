# LightningTests
PyTorch Lightning Tests

Running training for 3 epochs with custom learning rate and weight decay

```bash
python main.py fit --model.lr 0.0001 --model.w_decay 0.001 --trainer.max_epochs 3
```